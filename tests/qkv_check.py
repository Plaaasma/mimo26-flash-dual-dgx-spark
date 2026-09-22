#!/usr/bin/env python3
"""qkv_check.py <shard0.safetensors>: does the TP2 fused-QKV path produce finite, correct weights, and do the fp8
dense GEMM kernels (Marlin W8A16 block-scaled, Triton W8A8 block-scaled) compute qkv_proj correctly on this GPU?
Uses the real layer-0 (full attention) and layer-1 (SWA) tensors of the checkpoint."""
import sys, math, torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
_ctx = set_current_vllm_config(VllmConfig()); _ctx.__enter__()
from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mf
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
dev = "cuda"; B = 128; torch.manual_seed(0)
f = safe_open(sys.argv[1], framework="pt", device="cpu")
def cdiv(a, b): return -(-a // b)
def dequant_grid(w, s, chunks):
    """dequantize an fp8 weight whose scale rows are tiled per checkpoint chunk (rows_per_chunk rows each)."""
    rows, K = w.shape; rpc = rows // chunks; csr = cdiv(rpc, B)
    r = torch.arange(rows, device=w.device); si = (r // rpc) * csr + (r % rpc) // B
    return w.float() * s[si].repeat_interleave(B, dim=1)[:, :K]
def stats(name, t):
    tf = t.float(); fin = torch.isfinite(tf).all().item()
    print(f"  {name:28s} {str(t.dtype):22s} {tuple(t.shape)}  finite={fin}  absmax={tf.abs().max().item():.4g}  min={tf.min().item():.4g}")
    return fin
for layer, nkv in ((0, 4), (1, 8)):
    w = f.get_tensor(f"model.layers.{layer}.self_attn.qkv_proj.weight").to(dev)
    s = f.get_tensor(f"model.layers.{layer}.self_attn.qkv_proj.weight_scale_inv").to(dev)
    print(f"== layer {layer}: num_kv_heads={nkv}, ckpt_tp=4, weight {tuple(w.shape)} scales {tuple(s.shape)}")
    W_deq = dequant_grid(w, s, 4)                       # reference dequant of the full checkpoint weight
    for r in (0, 1):
        wr, sr = _shard_fp8_qkv_proj(w, s, num_heads=64, num_kv_heads=nkv, head_dim=192, v_head_dim=128, tp_rank=r, tp_size=2, ckpt_tp=4)
        ok = stats(f"rank{r} w_rank", wr) & stats(f"rank{r} s_rank", sr)
        N, K = wr.shape
        # reference rows for this rank, same head order as the loader: Q heads r*32.., K heads r*nkv/2.., V heads
        qh = range(r * 32, (r + 1) * 32); kvh = range(r * (nkv // 2), (r + 1) * (nkv // 2))
        rpc = w.shape[0] // 4; qpc = 16 * 192; kpc = (nkv // 4) * 192; vpc = (nkv // 4) * 128
        idx = []
        for h in qh: c = h // 16; idx.append(c * rpc + (h % 16) * 192 + torch.arange(192))
        for h in kvh: c = h // (nkv // 4); idx.append(c * rpc + qpc + (h % (nkv // 4)) * 192 + torch.arange(192))
        for h in kvh: c = h // (nkv // 4); idx.append(c * rpc + qpc + kpc + (h % (nkv // 4)) * 128 + torch.arange(128))
        idx = torch.cat(idx).to(dev); ref_w = W_deq[idx]
        wr_deq = wr.float() * sr.repeat_interleave(B, dim=0)[:N].repeat_interleave(B, dim=1)[:, :K]
        rel = ((wr_deq - ref_w).abs().max() / ref_w.abs().max()).item()
        print(f"  rank{r} requantized weight vs reference rows: max rel err {rel:.4f} {'OK' if rel < 0.05 else 'BAD'}")
        import os
        M_ = int(os.environ.get("QKV_M", "64")); xscale = float(os.environ.get("QKV_XSCALE", "1"))
        x = torch.randn(M_, K, device=dev, dtype=torch.bfloat16) * xscale
        ref_y = x.float() @ ref_w.t()
        # Marlin fp8 W8A16 with block scales (what --linear-backend marlin runs for qkv_proj)
        try:
            layer_m = torch.nn.Module()
            layer_m.weight = torch.nn.Parameter(wr.clone(), requires_grad=False)
            layer_m.weight_scale_inv = torch.nn.Parameter(sr.clone(), requires_grad=False)
            layer_m.input_size_per_partition = K; layer_m.output_size_per_partition = N
            layer_m.weight_block_size = [B, B]; layer_m.orig_dtype = torch.bfloat16
            mf.prepare_fp8_layer_for_marlin(layer_m, size_k_first=False)
            ws_ = marlin_make_workspace_new(x.device)
            sc = layer_m.weight_scale if hasattr(layer_m, "weight_scale") else layer_m.weight_scale_inv
            for rep in range(3):   # repeated calls: workspace/lock reuse
                y = mf.apply_fp8_marlin_linear(x, layer_m.weight, sc, ws_, N, K, None)
                d = (y.float() - ref_y).abs(); print(f"  rank{r} MARLIN fp8 W8A16 M={M_} x{xscale} call{rep}: finite={torch.isfinite(y).all().item()} max|diff|={d.max().item():.4g} rel-mean={(d.mean() / ref_y.abs().mean()).item():.4f}")
            y2 = mf.apply_fp8_marlin_linear(x, layer_m.weight, sc, ws_, N, K, None, use_fp32_reduce=False)
            d = (y2.float() - ref_y).abs(); print(f"  rank{r} MARLIN fp8 W8A16 M={M_} no-fp32-reduce: finite={torch.isfinite(y2).all().item()} rel-mean={(d.mean() / ref_y.abs().mean()).item():.4f}")
        except Exception as e:
            print(f"  rank{r} MARLIN fp8 FAIL: {str(e)[:160]}")
        # Triton fp8 W8A8 block-scaled (what --linear-backend triton runs)
        try:
            q = QuantFP8(static=False, group_shape=GroupShape(1, 128), column_major_scales=True); xq, xs = q(x)
            y = w8a8_triton_block_scaled_mm(xq, wr, xs, sr, [B, B], torch.bfloat16)
            d = (y.float() - ref_y).abs(); print(f"  rank{r} TRITON fp8 W8A8: finite={torch.isfinite(y).all().item()} max|diff|={d.max().item():.4g} rel-mean={(d.mean() / ref_y.abs().mean()).item():.4f}")
        except Exception as e:
            print(f"  rank{r} TRITON fp8 FAIL: {str(e)[:160]}")
