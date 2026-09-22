"""NVFP4 KV for the DiffKV kernel: write kernel (reshape_and_cache_nvfp4_diffkv) + attention on the packed cache vs
(a) an fp32 reference on the DEQUANTIZED cache (kernel correctness) and (b) the same reference on the bf16 cache
(quantization error), on the fp8 test's cases: decode q=1 (3D split-KV), spec verify q=8 (3D), prefill 512 (2D,
BLOCK_M 128), SWA + sinks q=8, SWA prefill + sinks. Shapes: MiMo TP2, K 192 / V 128, block 16."""
import itertools
import torch
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m
from vllm.v1.attention.ops.nvfp4_diffkv import reshape_and_cache_nvfp4_diffkv, dequant_nvfp4_cache, nvfp4_row_bytes, dequant_nvfp4_blocks

torch.manual_seed(0)
dev = "cuda"; HQ, DK, DV, BS = 32, 192, 128, 16
SEGS, THR = 16, 64
ROW = nvfp4_row_bytes(DK, DV)


def make(ctx_lens, q_lens, hkv):
    """bf16 paged cache [nb, BS, hkv, DK+DV] + the same content written through the NVFP4 kernel."""
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens)]
    nblk = [(s + BS - 1) // BS for s in seq_lens]; nb = sum(nblk) + 4
    kv = torch.randn(nb, BS, hkv, DK + DV, device=dev, dtype=torch.bfloat16) * 1.5
    perm = torch.randperm(nb, device=dev); bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[o:o + n].to(torch.int32); o += n
    q = torch.randn(sum(q_lens), HQ, DK, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(itertools.accumulate(q_lens)), dtype=torch.int32, device=dev)
    # NVFP4 cache in vLLM's allocation layout [nb, hkv, BS, ROW]; fill every (block, token) slot from kv
    nv = torch.zeros(nb, hkv, BS, ROW, device=dev, dtype=torch.uint8)
    flat = kv.reshape(nb * BS, hkv, DK + DV)
    slots = torch.arange(nb * BS, device=dev, dtype=torch.int64)
    reshape_and_cache_nvfp4_diffkv(flat[..., :DK].contiguous(), flat[..., DK:].contiguous(), nv, slots)
    return kv, nv, bt, q, cu, torch.tensor(seq_lens, dtype=torch.int32, device=dev)


def ref(kv, bt, q, cu, sl, hkv, sinks=None, window=-1):
    outs = []
    for i, s in enumerate(sl.tolist()):
        qs, qe = cu[i].item(), cu[i + 1].item(); ql = qe - qs; n = (s + BS - 1) // BS
        blk = kv[bt[i, :n].long()].reshape(n * BS, hkv, DK + DV)[:s].float()
        Kx = blk[..., :DK].repeat_interleave(HQ // hkv, 1); Vx = blk[..., DK:].repeat_interleave(HQ // hkv, 1)
        S = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), Kx) * DK ** -0.5
        qp = torch.arange(s - ql, s, device=dev)[:, None]; kp = torch.arange(s, device=dev)[None, :]
        mask = kp <= qp
        if window > 0: mask &= kp > qp - (window + 1)
        S = S.masked_fill(~mask[None], float("-inf"))
        P = (torch.softmax(torch.cat([S, sinks[:, None, None].expand(HQ, ql, 1)], -1), -1)[..., :-1]
             if sinks is not None else torch.softmax(S, -1))
        outs.append(torch.einsum("hqk,khd->qhd", P, Vx))
    return torch.cat(outs)


def run(nv, bt, q, cu, sl, max_q, sinks=None, window=-1, use3d=True):
    out = torch.empty(q.shape[0], HQ, DV, device=dev, dtype=torch.bfloat16)
    kw = {}
    if use3d:
        kw = dict(seq_threshold_3D=THR, num_par_softmax_segments=SEGS,
                  softmax_segm_output=torch.empty(THR, HQ, SEGS, DV, device=dev),
                  softmax_segm_max=torch.empty(THR, HQ, SEGS, device=dev),
                  softmax_segm_expsum=torch.empty(THR, HQ, SEGS, device=dev))
    m.unified_attention_diffkv(q=q, k=nv, v=nv, out=out, cu_seqlens_q=cu, seqused_k=sl, softmax_scale=DK ** -0.5,
                               causal=True, window_size=(window, -1), block_table=bt, softcap=0, max_seqlen_q=max_q,
                               sinks=sinks, nvfp4_cache=nv.transpose(1, 2), head_size_v=DV, **kw)
    torch.cuda.synchronize(); return out


ok = True
cases = [  # label, ctx, qlens, hkv, window, sinks, max_q
    ("decode q=1 (3D)", [46000], [1], 2, -1, False, 1),
    ("spec verify q=8 (3D)", [46000, 9000], [8, 8], 2, -1, False, 8),
    ("prefill chunk (BLOCK_M 128)", [8000, 30], [512, 1], 2, -1, False, 512),
    ("SWA + sinks q=8", [20000], [8], 4, 127, True, 8),
    ("SWA prefill + sinks", [3000], [300], 4, 127, True, 300),
    ("mixed small", [5, 33, 100], [3, 1, 17], 2, -1, False, 17),
]
for label, ctx, ql, hkv, win, sk, mq in cases:
    kv, nv, bt, q, cu, sl = make(ctx, ql, hkv)
    sinks = torch.randn(HQ, device=dev) if sk else None
    deq = dequant_nvfp4_cache(nv, DK, DV).transpose(1, 2).to(torch.bfloat16)   # [nb, BS, hkv, DK+DV] like kv
    r_bf16 = ref(kv, bt, q, cu, sl, hkv, sinks, win)
    r_deq = ref(deq, bt, q, cu, sl, hkv, sinks, win)
    o = run(nv, bt, q, cu, sl, mq, sinks=sinks, window=win)
    e_kernel = (o.float() - r_deq).abs().max().item()
    e_quant = (o.float() - r_bf16).abs()
    # write-kernel check: dequantized cache vs source within NVFP4 step (max rel step 1/6 of the group amax)
    src = kv.float(); dq = deq.float()
    w_err = ((dq - src).abs() / (src.abs().amax(-1, keepdim=True) + 1e-6)).max().item()
    good = (not torch.isnan(o).any()) and e_kernel <= 2.5e-2 and w_err <= 0.2
    ok &= good
    print(f"{label:30s}: kernel err vs dequant ref {e_kernel:.4f} | quant err vs bf16 ref max {e_quant.max().item():.4f} "
          f"mean {e_quant.mean().item():.5f} | write err (rel amax) {w_err:.3f} {'OK' if good else 'FAIL'}")
# prefill dequant kernel: blocks -> bf16 scratch must equal the reference dequant exactly (NVFP4 values are exact in bf16)
kv, nv, bt, q, cu, sl = make([3000], [64], 2)
ref = dequant_nvfp4_cache(nv, DK, DV)                      # [nb, H, BS, 320] f32
blocks = bt.reshape(-1).to(torch.int64)
got = dequant_nvfp4_blocks(nv, blocks, DK, DV).float()     # [n, BS, H, 320]
exp = ref[blocks].transpose(1, 2)
dq_ok = torch.equal(got, exp)
ok &= dq_ok
print(f"dequant_nvfp4_blocks vs reference: {'exact' if dq_ok else 'MISMATCH max ' + str((got - exp).abs().max().item())}")
print("ALL OK" if ok else "SOME FAILED")
