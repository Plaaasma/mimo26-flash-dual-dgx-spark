#!/usr/bin/env python3
"""attn_check.py: vLLM's DiffKV cache write (triton_reshape_and_cache_flash_diffkv) + attention kernel on a paged
bf16 cache vs a naive torch reference, at MiMo's TP2 shapes: full layer 32q/2kv, SWA layer 32q/4kv with sinks and
window 128, K 192 / V 128, block size 16 (the layout vLLM allocates: [blocks, block, heads, 320], viewed transposed)."""
import torch, math
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash_diffkv
from vllm.v1.attention.ops.triton_unified_attention_diffkv import unified_attention_diffkv
torch.manual_seed(0); dev = "cuda"; DK, DV, BS = 192, 128, 16
def run(name, HQ, HKV, T, window, sinks):
    q = torch.randn(T, HQ, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(T, HKV, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(T, HKV, DV, device=dev, dtype=torch.bfloat16)
    nblk = (T + BS - 1) // BS + 2
    kv_cache = torch.zeros(nblk, BS, HKV, DK + DV, device=dev, dtype=torch.bfloat16)   # kernel layout [blocks, block, heads, D] (vLLM allocates [blocks, heads, block, D] and transposes)
    perm = torch.randperm(nblk, device=dev)[: (T + BS - 1) // BS]
    slot = torch.tensor([int(perm[i // BS]) * BS + i % BS for i in range(T)], device=dev, dtype=torch.int64)
    one = torch.tensor(1.0, device=dev)
    triton_reshape_and_cache_flash_diffkv(k, v, kv_cache, slot, "auto", one, one)
    key_cache, value_cache = kv_cache[..., :DK], kv_cache[..., DK:DK + DV]
    bt = perm.to(torch.int32).view(1, -1); cu = torch.tensor([0, T], device=dev, dtype=torch.int32); sl = torch.tensor([T], device=dev, dtype=torch.int32)
    scale = 1.0 / math.sqrt(DK)
    sink_t = (torch.randn(HQ, device=dev, dtype=torch.float32) if sinks else None)
    out = torch.empty(T, HQ, DV, device=dev, dtype=torch.bfloat16)
    segs = 16; thr = 64
    unified_attention_diffkv(q=q, k=key_cache, v=value_cache, out=out, cu_seqlens_q=cu, seqused_k=sl, softmax_scale=scale, causal=True,
        alibi_slopes=None, use_alibi_sqrt=False, window_size=(window - 1, -1) if window else (-1, -1), block_table=bt, softcap=0.0,
        sinks=sink_t, max_seqlen_q=T, seq_threshold_3D=thr, num_par_softmax_segments=segs,
        softmax_segm_output=torch.empty(thr, HQ, segs, 128, device=dev, dtype=torch.float32),
        softmax_segm_max=torch.empty(thr, HQ, segs, device=dev, dtype=torch.float32),
        softmax_segm_expsum=torch.empty(thr, HQ, segs, device=dev, dtype=torch.float32))
    torch.cuda.synchronize()
    # naive reference (fp32): GQA, causal, optional sliding window, optional sinks (extra logit per head, HF semantics)
    g = HQ // HKV
    kf = k.float().repeat_interleave(g, dim=1); vf = v.float().repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale
    i = torch.arange(T, device=dev); mask = i[None, :] > i[:, None]
    if window: mask |= (i[:, None] - i[None, :]) >= window
    s = s.masked_fill(mask[None], float("-inf"))
    if sinks: s = torch.cat([s, sink_t.float().view(HQ, 1, 1).expand(HQ, T, 1)], dim=-1)
    p = torch.softmax(s, dim=-1)
    if sinks: p = p[..., :T]
    ref = torch.einsum("hqk,khd->qhd", p, vf)
    d = (out.float() - ref).abs(); rel = d.max().item() / ref.abs().max().item()
    print(f"{name:36s} T={T:4d} max|diff| {d.max().item():.4f} rel {rel:.4f} mean-rel {(d.mean() / ref.abs().mean()).item():.4f} {'OK' if rel < 0.03 else 'BAD'}")
run("full 32/2 no sinks", 32, 2, 6, None, False)
run("full 32/2 no sinks", 32, 2, 200, None, False)
run("SWA 32/4 sinks win128", 32, 4, 6, 128, True)
run("SWA 32/4 sinks win128", 32, 4, 300, 128, True)
run("SWA 32/4 NO sinks win128", 32, 4, 300, 128, False)
