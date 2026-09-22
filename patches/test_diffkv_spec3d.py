"""Numerics + timing for the diffkv split-KV spec-decode patch: 3D (patched) vs 2D (stock path) vs fp32 reference.
MiMo global-layer shapes at TP2: 32 q heads, 2 kv heads, K 192 / V 128, bf16 packed cache."""
import time
import torch
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m

torch.manual_seed(0)
dev = "cuda"
HQ, HKV, DK, DV, BS = 32, 2, 192, 128, 64
SEGS, THR = 16, 64


def make(seq_lens, qlen):
    nseq = len(seq_lens)
    nblk_per = [(s + BS - 1) // BS for s in seq_lens]
    nblocks = sum(nblk_per) + 4
    kv = (torch.randn(nblocks, BS, HKV, DK + DV, device=dev, dtype=torch.bfloat16) * 1.5)
    perm = torch.randperm(nblocks, device=dev)
    bt = torch.zeros(nseq, max(nblk_per), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nblk_per):
        bt[i, :n] = perm[o:o + n].to(torch.int32); o += n
    q = torch.randn(nseq * qlen, HQ, DK, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(0, nseq * qlen + 1, qlen, dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    return kv, bt, q, cu, sl


def run(kv, bt, q, cu, sl, qlen, max_q, sinks=None, window=-1):
    m._SPEC_3D_MAX_Q = max_q
    out = torch.empty(q.shape[0], HQ, DV, device=dev, dtype=torch.bfloat16)
    segm_o = torch.empty(THR, HQ, SEGS, DV, device=dev, dtype=torch.float32)
    segm_m = torch.empty(THR, HQ, SEGS, device=dev, dtype=torch.float32)
    segm_e = torch.empty(THR, HQ, SEGS, device=dev, dtype=torch.float32)
    kc, vc = kv[..., :DK], kv[..., DK:]
    args = dict(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, seqused_k=sl, softmax_scale=DK ** -0.5, causal=True,
                window_size=(window, -1), block_table=bt, softcap=0, max_seqlen_q=qlen, sinks=sinks,
                seq_threshold_3D=THR, num_par_softmax_segments=SEGS, softmax_segm_output=segm_o,
                softmax_segm_max=segm_m, softmax_segm_expsum=segm_e)
    m.unified_attention_diffkv(**args)
    torch.cuda.synchronize()
    return out, args


def ref(kv, bt, q, sl, qlen, sinks=None, window=-1):
    outs = []
    for i, s in enumerate(sl.tolist()):
        n = (s + BS - 1) // BS
        blk = kv[bt[i, :n].long()].reshape(n * BS, HKV, DK + DV)[:s].float()
        K, V = blk[..., :DK], blk[..., DK:]
        Q = q[i * qlen:(i + 1) * qlen].float()
        Kx = K.repeat_interleave(HQ // HKV, dim=1); Vx = V.repeat_interleave(HQ // HKV, dim=1)
        S = torch.einsum("qhd,khd->hqk", Q, Kx) * DK ** -0.5
        qpos = torch.arange(s - qlen, s, device=dev)[:, None]; kpos = torch.arange(s, device=dev)[None, :]
        mask = kpos <= qpos
        if window > 0:
            mask &= kpos > qpos - (window + 1)
        S = S.masked_fill(~mask[None], float("-inf"))
        if sinks is not None:
            S = torch.cat([S, sinks.float()[:, None, None].expand(HQ, qlen, 1)], dim=-1)
            P = torch.softmax(S, -1)[..., :-1]
        else:
            P = torch.softmax(S, -1)
        outs.append(torch.einsum("hqk,khd->qhd", P, Vx))
    return torch.cat(outs)


def timeit(args, max_q, n=50):
    m._SPEC_3D_MAX_Q = max_q
    for _ in range(3): m.unified_attention_diffkv(**args)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): m.unified_attention_diffkv(**args)
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3


ok = True
cases = [([46000], 8, None, -1), ([46000, 30011, 1000, 17], 8, None, -1), ([12345, 777], 4, None, -1),
         ([46000], 8, "sinks", -1), ([46000, 12], 8, None, 127), ([46000, 12], 8, "sinks", 127), ([9], 8, None, -1), ([2000] * 8, 8, None, -1), ([2163], 8, None, -1), ([2167, 2161, 30000, 5000], 8, None, -1)]
for seq_lens, qlen, sk, win in cases:
    kv, bt, q, cu, sl = make(seq_lens, qlen)
    sinks = torch.randn(HQ, device=dev, dtype=torch.float32) if sk else None
    o3, args = run(kv, bt, q, cu, sl, qlen, 16, sinks, win)
    o3 = o3.clone()
    o2, _ = run(kv, bt, q, cu, sl, qlen, 0, sinks, win)
    r = ref(kv, bt, q, sl, qlen, sinks, win)
    e3 = (o3.float() - r).abs().max().item(); e2 = (o2.float() - r).abs().max().item()
    nan = bool(torch.isnan(o3).any())
    good = (not nan) and e3 < 2e-2 and e3 <= 2 * e2 + 5e-3
    ok &= good
    t3 = timeit(args, 16); t2 = timeit(args, 0)
    print(f"lens={seq_lens} q={qlen} sinks={bool(sk)} win={win}: max|3D-ref|={e3:.4f} max|2D-ref|={e2:.4f} nan={nan} "
          f"| 3D {t3:.3f} ms vs 2D {t2:.3f} ms ({t2/t3:.1f}x) {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED")
