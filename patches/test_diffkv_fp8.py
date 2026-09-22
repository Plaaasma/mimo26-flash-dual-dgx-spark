"""fp8 (E4M3, per-tensor) KV for the patched diffkv kernel: every launch path vs (a) fp32 reference on the DEQUANTIZED
cache (kernel correctness) and (b) fp32 reference on the original bf16 cache (quantization error), at scale 1.0
(vLLM default without checkpoint scales) and amax/448. MiMo TP2 shapes: global 32q/2kv, SWA 32q/4kv, K 192 / V 128."""
import itertools
import torch
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m

torch.manual_seed(0)
dev = "cuda"; HQ, DK, DV, BS = 32, 192, 128, 64
SEGS, THR = 16, 64
F8 = torch.float8_e4m3fn


def make(ctx_lens, q_lens, hkv, kscale_amp=1.5):
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens)]
    nblk = [(s + BS - 1) // BS for s in seq_lens]; nb = sum(nblk) + 4
    kv = torch.randn(nb, BS, hkv, DK + DV, device=dev, dtype=torch.bfloat16) * kscale_amp
    perm = torch.randperm(nb, device=dev); bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[o:o + n].to(torch.int32); o += n
    q = torch.randn(sum(q_lens), HQ, DK, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(itertools.accumulate(q_lens)), dtype=torch.int32, device=dev)
    return kv, bt, q, cu, torch.tensor(seq_lens, dtype=torch.int32, device=dev)


def quant(kv, ks, vs):
    k8 = (kv[..., :DK].float() / ks).to(F8); v8 = (kv[..., DK:].float() / vs).to(F8)
    kv8 = torch.cat([k8.view(torch.uint8), v8.view(torch.uint8)], -1).view(F8)
    deq = torch.cat([(k8.float() * ks), (v8.float() * vs)], -1).to(torch.bfloat16)
    return kv8, deq


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


def run(kv, bt, q, cu, sl, max_q, ks=None, vs=None, sinks=None, window=-1, use3d=True):
    out = torch.empty(q.shape[0], HQ, DV, device=dev, dtype=torch.bfloat16)
    kw = {}
    if use3d:
        kw = dict(seq_threshold_3D=THR, num_par_softmax_segments=SEGS,
                  softmax_segm_output=torch.empty(THR, HQ, SEGS, DV, device=dev),
                  softmax_segm_max=torch.empty(THR, HQ, SEGS, device=dev),
                  softmax_segm_expsum=torch.empty(THR, HQ, SEGS, device=dev))
    m.unified_attention_diffkv(q=q, k=kv[..., :DK], v=kv[..., DK:], out=out, cu_seqlens_q=cu, seqused_k=sl,
                               softmax_scale=DK ** -0.5, causal=True, window_size=(window, -1), block_table=bt,
                               softcap=0, max_seqlen_q=max_q, sinks=sinks, k_descale=ks, v_descale=vs, **kw)
    torch.cuda.synchronize(); return out


ok = True
cases = [  # label, ctx, qlens, hkv, window, sinks, max_q
    ("decode q=1 (3D)", [46000], [1], 2, -1, False, 1),
    ("spec verify q=8 (patched 3D)", [46000, 9000], [8, 8], 2, -1, False, 8),
    ("prefill chunk (BLOCK_M 128)", [8000, 30], [512, 1], 2, -1, False, 512),
    ("SWA + sinks q=8", [20000], [8], 4, 127, True, 8),
    ("SWA prefill + sinks", [3000], [300], 4, 127, True, 300),
]
for label, ctx, ql, hkv, win, sk, mq in cases:
    kv, bt, q, cu, sl = make(ctx, ql, hkv)
    sinks = torch.randn(HQ, device=dev) if sk else None
    r_bf16 = ref(kv, bt, q, cu, sl, hkv, sinks, win)
    base = run(kv, bt, q, cu, sl, mq, sinks=sinks, window=win)  # bf16 cache through the same (patched) launcher
    e_base = (base.float() - r_bf16).abs().max().item()
    for sname, ks_v, vs_v in [("scale 1.0", 1.0, 1.0),
                              ("amax/448", kv[..., :DK].abs().max().item() / 448, kv[..., DK:].abs().max().item() / 448)]:
        ks = torch.tensor([ks_v], device=dev, dtype=torch.float32); vs = torch.tensor([vs_v], device=dev, dtype=torch.float32)
        kv8, deq = quant(kv, ks_v, vs_v)
        o8 = run(kv8, bt, q, cu, sl, mq, ks, vs, sinks, win)
        e_kernel = (o8.float() - ref(deq, bt, q, cu, sl, hkv, sinks, win)).abs().max().item()
        e_quant = (o8.float() - r_bf16).abs()
        good = not torch.isnan(o8).any() and e_kernel <= max(2 * e_base, 1e-2)
        ok &= good
        print(f"{label:30s} {sname:9s}: kernel err vs dequant ref {e_kernel:.4f} (bf16-cache err {e_base:.4f}) | "
              f"fp8 quant err vs bf16 ref max {e_quant.max().item():.4f} mean {e_quant.mean().item():.5f} {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED")
