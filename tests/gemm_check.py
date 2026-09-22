#!/usr/bin/env python3
"""gemm_check.py: numerical sanity of the fp8 GEMM paths vLLM can pick for MiMo's dense fp8 linears on this GPU.
Compares each kernel against a float32 reference built from the dequantized weights (M=256, K=4096, N=6784 = the
TP2 fused-qkv shape). Run inside the kit image with the GPU: tests/gemm_check.sh"""
import inspect, torch
from vllm.config import VllmConfig, set_current_vllm_config
_cfg = VllmConfig(); _ctx = set_current_vllm_config(_cfg); _ctx.__enter__()
torch.manual_seed(0)
dev = "cuda"; M, K, N, B = 256, 4096, 6784, 128
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = (torch.randn(N, K, device=dev) * 0.02)
# block-quantize the weight (128x128 blocks) like the checkpoint
wb = w.view(N // B, B, K // B, B)
ws = wb.abs().amax(dim=(1, 3), keepdim=True).float() / 448.0
wq = (wb / ws).to(torch.float8_e4m3fn)
w_deq = (wq.float() * ws).view(N, K)
wq = wq.view(N, K); ws = ws.view(N // B, K // B)
ref = x.float() @ w_deq.t()
def report(name, out, ref=ref):
    d = (out.float() - ref).abs(); err = d.max().item(); rel = err / ref.abs().max().item(); mean = d.mean().item() / ref.abs().mean().item()
    nan = torch.isnan(out).any().item()
    print(f"{name:40s} max|diff| {err:9.4f}  rel-max {rel:.4f}  rel-mean {mean:.4f}  nan={nan}  {'OK' if (rel < 0.05 and not nan) else 'BAD'}")
# 1. Triton fp8 block-scaled W8A8 (what --linear-backend triton uses)
from vllm.model_executor.layers.quantization.utils.fp8_utils import w8a8_triton_block_scaled_mm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
q = QuantFP8(static=False, group_shape=GroupShape(1, 128), column_major_scales=True)
xq, xs = q(x)
print("w8a8_triton_block_scaled_mm signature:", inspect.signature(w8a8_triton_block_scaled_mm))
# reference on the SAME quantized activations (isolates the GEMM kernel from the activation-quant error)
x_deq = (xq.float().view(M, K // B, B) * xs.t().contiguous().view(M, K // B, 1)).view(M, K) if xs.shape[0] == K // B else (xq.float().view(M, K // B, B) * xs.view(M, K // B, 1)).view(M, K)
ref_q = x_deq @ w_deq.t()
try:
    out = w8a8_triton_block_scaled_mm(xq, wq, xs, ws, [B, B], torch.bfloat16)
    report("triton w8a8 block-scaled vs bf16-act ref", out)
    report("triton w8a8 block-scaled vs fp8-act ref", out, ref_q)
except Exception as e:
    print("triton w8a8 block-scaled FAIL:", str(e)[:200])
# Marlin fp8 W8A16 with block scales (what --linear-backend marlin uses)
try:
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mf
    import types
    layer = types.SimpleNamespace(weight=torch.nn.Parameter(wq.clone(), requires_grad=False),
                                  weight_scale_inv=torch.nn.Parameter(ws.clone(), requires_grad=False),
                                  weight_scale=torch.nn.Parameter(ws.clone(), requires_grad=False),
                                  input_size_per_partition=K, output_size_per_partition=N, weight_block_size=[B, B], orig_dtype=torch.bfloat16)
    print("marlin_utils_fp8 helpers:", [n for n in dir(mf) if "prepare" in n or "apply" in n])
except Exception as e:
    print("marlin fp8 probe FAIL:", str(e)[:200])
# 2. bf16 reference through torch on this GPU (sanity of the reference itself)
report("torch bf16 matmul (dequant weight)", x @ w_deq.to(torch.bfloat16).t())
# 3. CUTLASS block-scaled (what auto picked): expected to raise on sm_121
try:
    import vllm._custom_ops as ops
    out = ops.cutlass_scaled_mm(xq, wq.t(), xs, ws, torch.bfloat16) if hasattr(ops, "cutlass_scaled_mm") else None
    if out is not None: report("cutlass_scaled_mm block", out)
except Exception as e:
    print("cutlass_scaled_mm FAIL:", str(e).splitlines()[0][:120])
