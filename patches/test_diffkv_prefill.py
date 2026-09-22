"""Prefill-launch sweep for the diffkv patch: BLOCK_M / num_warps / tile on MiMo global + SWA layer shapes (TP2).
Correctness vs fp32 reference on a small chunk; timing on a 4096-token chunk at 30K context (and a mixed batch)."""
import itertools, time
import torch
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m

torch.manual_seed(0)
dev = "cuda"
HQ, HKV, DK, DV, BS = 32, 2, 192, 128, 64
HKV_SWA = 4  # SWA layers: 8 kv heads / TP2 (same 64 q heads -> group 8)


def make(ctx_lens, q_lens, hkv):
    nseq = len(ctx_lens)
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens)]
    nblk_per = [(s + BS - 1) // BS for s in seq_lens]
    nblocks = sum(nblk_per) + 4
    kv = torch.randn(nblocks, BS, hkv, DK + DV, device=dev, dtype=torch.bfloat16) * 1.5
    perm = torch.randperm(nblocks, device=dev)
    bt = torch.zeros(nseq, max(nblk_per), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nblk_per):
        bt[i, :n] = perm[o:o + n].to(torch.int32); o += n
    q = torch.randn(sum(q_lens), HQ, DK, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(itertools.accumulate(q_lens)), dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    return kv, bt, q, cu, sl


def call(kv, bt, q, cu, sl, max_q, cfg, sinks=None, window=-1):
    bm, nw, ns, tile = cfg
    m._PREFILL_BLOCK_M, m._PREFILL_NUM_WARPS, m._PREFILL_NUM_STAGES, m._PREFILL_TILE = bm, nw, ns, tile
    out = torch.empty(q.shape[0], HQ, DV, device=dev, dtype=torch.bfloat16)
    m.unified_attention_diffkv(q=q, k=kv[..., :DK], v=kv[..., DK:], out=out, cu_seqlens_q=cu, seqused_k=sl,
                               softmax_scale=DK ** -0.5, causal=True, window_size=(window, -1), block_table=bt,
                               softcap=0, max_seqlen_q=max_q, sinks=sinks)
    return out


def ref(kv, bt, q, cu, sl, hkv, sinks=None, window=-1):
    outs = []
    for i, s in enumerate(sl.tolist()):
        qs, qe = cu[i].item(), cu[i + 1].item(); ql = qe - qs
        n = (s + BS - 1) // BS
        blk = kv[bt[i, :n].long()].reshape(n * BS, hkv, DK + DV)[:s].float()
        Kx = blk[..., :DK].repeat_interleave(HQ // hkv, 1); Vx = blk[..., DK:].repeat_interleave(HQ // hkv, 1)
        S = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), Kx) * DK ** -0.5
        qpos = torch.arange(s - ql, s, device=dev)[:, None]; kpos = torch.arange(s, device=dev)[None, :]
        mask = kpos <= qpos
        if window > 0: mask &= kpos > qpos - (window + 1)
        S = S.masked_fill(~mask[None], float("-inf"))
        if sinks is not None:
            P = torch.softmax(torch.cat([S, sinks[:, None, None].expand(HQ, ql, 1)], -1), -1)[..., :-1]
        else:
            P = torch.softmax(S, -1)
        outs.append(torch.einsum("hqk,khd->qhd", P, Vx))
    return torch.cat(outs)


def bench(fn, n=10):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3


STOCK = (16, 4, 0, 32)
CFGS = [STOCK, (32, 4, 0, 32), (64, 4, 0, 32), (64, 8, 0, 32), (64, 4, 0, 64), (64, 8, 0, 64),
        (128, 8, 0, 32), (128, 8, 0, 64), (64, 4, 2, 64), (128, 8, 2, 32)]
ok = True
# correctness: small chunked-prefill batch mixing a prefill chunk with decode/verify rows
for hkv, win, sk in [(HKV, -1, False), (HKV_SWA, 127, True)]:
    kv, bt, q, cu, sl = make([1500, 700, 40], [300, 8, 1], hkv)
    sinks = torch.randn(HQ, device=dev) if sk else None
    r = ref(kv, bt, q, cu, sl, hkv, sinks, win)
    for cfg in CFGS:
        try:
            o = call(kv, bt, q, cu, sl, 300, cfg, sinks, win).float()
            e = (o - r).abs().max().item(); good = e < 2e-2 and not torch.isnan(o).any()
        except Exception as ex:  # noqa: BLE001
            e, good = float("nan"), False; print("  ERR", cfg, type(ex).__name__, str(ex)[:120])
        ok &= good
        print(f"correctness hkv={hkv} win={win} cfg(BM,warps,stages,tile)={cfg}: max err {e:.4f} {'OK' if good else 'FAIL'}")
# timing: one 4096-token chunk at 26K..30K context (the 46K prompt's middle chunk), global and SWA layers
for hkv, win, sk, label in [(HKV, -1, False, "global"), (HKV_SWA, 127, True, "swa")]:
    kv, bt, q, cu, sl = make([26000], [4096], hkv)
    sinks = torch.randn(HQ, device=dev) if sk else None
    base = None
    for cfg in CFGS:
        try:
            o = call(kv, bt, q, cu, sl, 4096, cfg, sinks, win)
            t = bench(lambda: call(kv, bt, q, cu, sl, 4096, cfg, sinks, win))
            if base is None: base, o0 = t, o.clone()
            d = (o.float() - o0.float()).abs().max().item()
            print(f"{label} 4096q@30K cfg={cfg}: {t:7.2f} ms  ({base/t:4.2f}x vs stock)  max|diff vs stock| {d:.4f}")
        except Exception as ex:  # noqa: BLE001
            print(f"{label} cfg={cfg}: ERR {type(ex).__name__} {str(ex)[:120]}")
print("ALL OK" if ok else "SOME FAILED")
