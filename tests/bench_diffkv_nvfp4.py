"""Timing of the DiffKV attention kernel with an NVFP4 cache vs an fp8 cache at MiMo TP2 shapes, plus the compiled
kernels' register / spill counts. Cases: prefill chunk 4096 @ 26K (global 32q/2kv), 512 @ 8K, SWA prefill 4096 @ 30K
(32q/4kv, window 128, sinks), decode q=1 @ 46K and spec verify q=8 @ 46K (3D split-KV)."""
import itertools, time
import torch
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m
from vllm.v1.attention.ops.nvfp4_diffkv import reshape_and_cache_nvfp4_diffkv, nvfp4_row_bytes, dequant_nvfp4_blocks

torch.manual_seed(0)
dev = "cuda"; HQ, DK, DV, BS = 32, 192, 128, 16
SEGS, THR = 16, 64
ROW = nvfp4_row_bytes(DK, DV)
F8 = torch.float8_e4m3fn


def make(ctx, ql, hkv):
    s = ctx + ql; n = (s + BS - 1) // BS; nb = n + 4
    kv = torch.randn(nb, BS, hkv, DK + DV, device=dev, dtype=torch.bfloat16)
    bt = torch.randperm(nb, device=dev)[:n].to(torch.int32).view(1, -1)
    q = torch.randn(ql, HQ, DK, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, ql], dtype=torch.int32, device=dev); sl = torch.tensor([s], dtype=torch.int32, device=dev)
    kv8 = kv.to(F8)
    nv = torch.zeros(nb, hkv, BS, ROW, device=dev, dtype=torch.uint8)
    flat = kv.reshape(nb * BS, hkv, DK + DV)
    reshape_and_cache_nvfp4_diffkv(flat[..., :DK].contiguous(), flat[..., DK:].contiguous(), nv,
                                   torch.arange(nb * BS, device=dev, dtype=torch.int64))
    return kv8, nv, bt, q, cu, sl


def launch(mode, kv8, nv, bt, q, cu, sl, max_q, window, sinks):
    out = torch.empty(q.shape[0], HQ, DV, device=dev, dtype=torch.bfloat16)
    kw = dict(seq_threshold_3D=THR, num_par_softmax_segments=SEGS,
              softmax_segm_output=torch.empty(THR, HQ, SEGS, DV, device=dev),
              softmax_segm_max=torch.empty(THR, HQ, SEGS, device=dev),
              softmax_segm_expsum=torch.empty(THR, HQ, SEGS, device=dev))
    one = torch.ones(1, device=dev)
    if mode == "dq":   # dequant the touched blocks to bf16 once, then the bf16 kernel path (timed together)
        def f():
            scratch = dequant_nvfp4_blocks(nv, bt.reshape(-1), DK, DV)
            nbt = torch.arange(bt.numel(), device=dev, dtype=torch.int32).view(bt.shape)
            m.unified_attention_diffkv(q=q, k=scratch[..., :DK], v=scratch[..., DK:], out=out, cu_seqlens_q=cu, seqused_k=sl,
                                       softmax_scale=DK ** -0.5, causal=True, window_size=(window, -1), block_table=nbt,
                                       softcap=0, max_seqlen_q=max_q, sinks=sinks, **kw)
        return f
    if mode == "fp8":
        f = lambda: m.unified_attention_diffkv(q=q, k=kv8[..., :DK], v=kv8[..., DK:], out=out, cu_seqlens_q=cu, seqused_k=sl,
                                               softmax_scale=DK ** -0.5, causal=True, window_size=(window, -1), block_table=bt,
                                               softcap=0, max_seqlen_q=max_q, sinks=sinks, k_descale=one, v_descale=one, **kw)
    else:
        f = lambda: m.unified_attention_diffkv(q=q, k=nv, v=nv, out=out, cu_seqlens_q=cu, seqused_k=sl,
                                               softmax_scale=DK ** -0.5, causal=True, window_size=(window, -1), block_table=bt,
                                               softcap=0, max_seqlen_q=max_q, sinks=sinks, nvfp4_cache=nv.transpose(1, 2),
                                               head_size_v=DV, **kw)
    return f


def timeit(f, n=5):
    f(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3


cases = [  # label, ctx, ql, hkv, window, sinks
    ("prefill 4096 @ 26K global", 26000, 4096, 2, -1, False),
    ("prefill 512 @ 8K global", 8000, 512, 2, -1, False),
    ("SWA prefill 4096 @ 30K", 30000, 4096, 4, 127, True),
    ("decode q=1 @ 46K global", 46000, 1, 2, -1, False),
    ("spec verify q=8 @ 46K", 46000, 8, 2, -1, False),
]
for label, ctx, ql, hkv, win, sk in cases:
    kv8, nv, bt, q, cu, sl = make(ctx, ql, hkv)
    sinks = torch.randn(HQ, device=dev) if sk else None
    t8 = timeit(launch("fp8", kv8, nv, bt, q, cu, sl, ql, win, sinks))
    t4 = timeit(launch("nvfp4", kv8, nv, bt, q, cu, sl, ql, win, sinks))
    tq = timeit(launch("dq", kv8, nv, bt, q, cu, sl, ql, win, sinks)) if ql >= 64 and win < 0 else float("nan")
    print(f"{label:28s} fp8 {t8:8.2f} ms | nvfp4 in-kernel {t4:8.2f} ms ({t4 / t8:4.1f}x) | nvfp4 dequant+bf16 {tq:8.2f} ms ({tq / t8:4.1f}x)", flush=True)
    del kv8, nv; torch.cuda.empty_cache()

print("\ncompiled variants (regs / spills / warps):")
for dev_key, cache in m.kernel_unified_attention_diffkv.device_caches.items() if hasattr(m.kernel_unified_attention_diffkv, "device_caches") else []:
    kcache = cache[0] if isinstance(cache, tuple) else cache
    for key, k in kcache.items():
        md = getattr(k, "metadata", None)
        print(f"  regs {getattr(k, 'n_regs', '?'):>4} spills {getattr(k, 'n_spills', '?'):>6} warps {getattr(md, 'num_warps', '?')} "
              f"stages {getattr(md, 'num_stages', '?')} smem {getattr(md, 'shared', '?')}")
