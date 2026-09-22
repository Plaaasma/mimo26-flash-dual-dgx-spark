#!/usr/bin/env python3
"""ops_check.py: value-level checks of vLLM's compiled (_C / _moe_C) ops the MiMo forward uses, vs torch references,
on this GPU. Ops built for sm_120 run on an sm_121 GB10 but could still compute wrong; this catches that."""
import torch, math
import vllm._custom_ops as ops
torch.manual_seed(0); dev = "cuda"
def rep(name, a, b, tol=2e-2):
    a = a.float(); b = b.float(); d = (a - b).abs().max().item(); scale = b.abs().max().item() or 1.0
    ok = torch.isfinite(a).all().item() and d / scale < tol
    print(f"{name:34s} max|diff| {d:9.4g}  rel {d / scale:.4f}  {'OK' if ok else 'BAD'}")
T, H = 512, 4096
x = torch.randn(T, H, device=dev, dtype=torch.bfloat16); w = (torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5)
# rms_norm
out = torch.empty_like(x); ops.rms_norm(out, x, w, 1e-6)
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
rep("rms_norm", out, ref)
# fused_add_rms_norm (in-place residual)
x2 = x.clone(); r2 = torch.randn_like(x); r_ref = r2.clone()
ops.fused_add_rms_norm(x2, r2, w, 1e-6)
h = x.float() + r_ref.float(); ref = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
rep("fused_add_rms_norm out", x2, ref); rep("fused_add_rms_norm residual", r2, h)
# silu_and_mul
y = torch.randn(T, 2 * 2048, device=dev, dtype=torch.bfloat16); out = torch.empty(T, 2048, device=dev, dtype=torch.bfloat16)
torch.ops._C.silu_and_mul(out, y); rep("silu_and_mul", out, torch.nn.functional.silu(y[:, :2048].float()) * y[:, 2048:].float())
# rotary_embedding (neox style, partial rotary 64 of head_dim 192, like MiMo)
nh, nkv, hd, rot = 32, 2, 192, 64
pos = torch.arange(T, device=dev)
q = torch.randn(T, nh * hd, device=dev, dtype=torch.bfloat16); k = torch.randn(T, nkv * hd, device=dev, dtype=torch.bfloat16)
inv = 1.0 / (1e7 ** (torch.arange(0, rot, 2, device=dev).float() / rot)); freqs = torch.outer(pos.float(), inv)
cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1).to(torch.bfloat16)  # [T, rot] (cos | sin), neox layout cache
def ref_rope(t, heads):
    t = t.float().view(T, heads, hd); r = t[..., :rot]; x1, x2 = r[..., : rot // 2], r[..., rot // 2 :]
    c = freqs.cos()[:, None, :]; s = freqs.sin()[:, None, :]
    rr = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1); return torch.cat([rr, t[..., rot:]], -1).view(T, heads * hd)
qr, kr = ref_rope(q, nh), ref_rope(k, nkv)
q1, k1 = q.clone(), k.clone(); ops.rotary_embedding(pos, q1, k1, hd, cos_sin, True)
rep("rotary_embedding q (neox, partial)", q1, qr); rep("rotary_embedding k", k1, kr)
# scaled_fp8_quant per-tensor + per-token vs torch
xq, s = ops.scaled_fp8_quant(x); rep("scaled_fp8_quant per-tensor", xq.float() * s, x, tol=8e-2)
xq, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True); rep("scaled_fp8_quant per-token", xq.float() * s, x, tol=8e-2)
# MoE routing: grouped_topk noaux_tc sigmoid like MiMo (n_group=1, topk_group=1, top8, renormalize, bias)
E, K = 256, 8
logits = torch.randn(T, E, device=dev, dtype=torch.float32); bias = torch.randn(E, device=dev) * 0.1
try:
    tw, ti = ops.grouped_topk(logits, K, True, 1, 1, "sigmoid", bias, 1.0)
    sc = torch.sigmoid(logits); sel = torch.topk(sc + bias, K, dim=-1).indices
    w_ref = torch.gather(sc, 1, sel); w_ref = w_ref / w_ref.sum(-1, keepdim=True)
    same = (torch.sort(ti.long(), -1).values == torch.sort(sel, -1).values).all(-1).float().mean().item()
    tw_sorted = torch.gather(tw.float(), 1, torch.argsort(ti.long(), -1)); w_ref_sorted = torch.gather(w_ref, 1, torch.argsort(sel, -1))
    print(f"{'grouped_topk noaux_tc':34s} expert-set match {same:.4f}  weight max|diff| {(tw_sorted - w_ref_sorted).abs().max().item():.4g}  {'OK' if same > 0.99 else 'BAD'}")
except Exception as e:
    print("grouped_topk FAIL:", str(e)[:160])
# topk_softmax (softmax routing)
try:
    tw = torch.empty(T, K, device=dev); ti = torch.empty(T, K, dtype=torch.int32, device=dev); ops.topk_softmax(tw, ti, logits)
    sel = torch.topk(torch.softmax(logits, -1), K, -1)
    same = (torch.sort(ti.long(), -1).values == torch.sort(sel.indices, -1).values).all(-1).float().mean().item()
    print(f"{'topk_softmax':34s} expert-set match {same:.4f}  {'OK' if same > 0.99 else 'BAD'}")
except Exception as e:
    print("topk_softmax FAIL:", str(e)[:160])
# moe_sum
a = torch.randn(T, K, H, device=dev, dtype=torch.bfloat16); out = torch.empty(T, H, device=dev, dtype=torch.bfloat16); ops.moe_sum(a, out); rep("moe_sum", out, a.float().sum(1))
# bf16 GEMM sanity (cuBLAS) at o_proj shape
A = torch.randn(T, 32 * 128, device=dev, dtype=torch.bfloat16); Wm = torch.randn(H, 32 * 128, device=dev, dtype=torch.bfloat16) * 0.02
rep("cublas bf16 matmul (o_proj shape)", A @ Wm.t(), A.float() @ Wm.float().t())
