"""Correctness (vs fp32 reference) + timing (vs our patched Triton diffkv kernel) for prefill_attn.cu.
MiMo global layer at TP2: 32 q heads / 2 kv heads, qk 192 / v 128, packed paged cache (fp8 or bf16)."""
import os, sys, time, torch
from torch.utils.cpp_extension import load

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
PS = int(args.get("page", "16"))
dev = "cuda"; torch.manual_seed(0)
HQ, HKV, DK, DV = 32, 2, 192, 128
here = os.path.dirname(os.path.abspath(__file__))
ext = load(name="prefill_attn_ext", sources=[os.environ.get("ATTN_SRC", f"{here}/prefill_attn.cu")], verbose=False,
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + os.environ.get("ATTN_CUSTOM_GENCODE", "-gencode=arch=compute_121a,code=sm_121a").split(),
           build_directory=os.environ.get("EXT_BUILD", "/tmp"))
import vllm.v1.attention.ops.triton_unified_attention_diffkv as m
F8 = torch.float8_e4m3fn


def make(ctxs, qls, dtype):
    seq_lens = [c + q for c, q in zip(ctxs, qls)]
    nblk = [(s + PS - 1) // PS for s in seq_lens]; NP = sum(nblk) + 3
    kv = (torch.randn(NP, HKV, PS, DK + DV, device=dev) * 0.7).to(dtype)
    perm = torch.randperm(NP, device=dev)
    bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32, device=dev); o = 0
    for i, n in enumerate(nblk): bt[i, :n] = perm[o:o + n].to(torch.int32); o += n
    q = (torch.randn(sum(qls), HQ, DK, device=dev) * 0.7).bfloat16()
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(qls)), dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    return kv, bt, q, cu, sl


def qblocks(cu, tok=8):  # static-shape q-block metadata: per block (seq, first global row)
    seqs, starts = [], []
    for i in range(len(cu) - 1):
        for r in range(int(cu[i]), int(cu[i + 1]), tok): seqs.append(i); starts.append(r)
    return torch.tensor(seqs, dtype=torch.int32, device=dev), torch.tensor(starts, dtype=torch.int32, device=dev)


def ref(kv, bt, q, cu, sl):
    out = torch.empty(q.shape[0], HQ, DV, device=dev)
    for i, s in enumerate(sl.tolist()):
        qs, qe = int(cu[i]), int(cu[i + 1]); ql = qe - qs; n = (s + PS - 1) // PS
        blk = kv[bt[i, :n].long()].permute(0, 2, 1, 3).reshape(n * PS, HKV, DK + DV)[:s].float()
        for h in range(HQ):
            K, V = blk[:, h // (HQ // HKV), :DK], blk[:, h // (HQ // HKV), DK:]
            sc = (q[qs:qe, h].float() @ K.T) * DK ** -0.5
            sc = sc.masked_fill(torch.arange(s, device=dev)[None, :] > torch.arange(s - ql, s, device=dev)[:, None], float("-inf"))
            out[qs:qe, h] = torch.softmax(sc, -1) @ V
    return out


def ours(kv, bt, q, cu, sl, out):
    qs, qst = qblocks(cu)
    ext.prefill_attn(q, kv, bt, cu, sl, qs, qst, DK ** -0.5, 1.0, 1.0, out)


def triton(kv, bt, q, cu, sl, out):
    k, v = kv[..., :DK].transpose(1, 2), kv[..., DK:].transpose(1, 2)
    kw = {} if kv.dtype != F8 else dict(k_descale=torch.ones(1, device=dev), v_descale=torch.ones(1, device=dev))
    m.unified_attention_diffkv(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, seqused_k=sl, softmax_scale=DK ** -0.5, causal=True,
                               window_size=(-1, -1), block_table=bt, softcap=0, max_seqlen_q=int((cu[1:] - cu[:-1]).max()), **kw)


def bench(fn, n=10):
    fn(); torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n


rel = lambda o, r: ((o.float() - r).norm() / r.norm()).item()
ok = True
for label, ctxs, qls in [("small mixed", [300, 0, 77], [64, 100, 13]), ("edge: 1-token seqs", [0, 5, 4095], [1, 3, 1]),
                         ("chunk 512 @ 8K", [8192], [512]), ("chunk 4096 @ 26K", [26000], [4096]), ("4 seqs 1024 @ 20K", [20000] * 4, [1024] * 4)]:
    for dtype in (torch.bfloat16, F8):
        kv, bt, q, cu, sl = make(ctxs, qls, dtype)
        out = torch.empty(q.shape[0], HQ, DV, device=dev, dtype=torch.bfloat16)
        r = ref(kv, bt, q, cu, sl)
        ours(kv, bt, q, cu, sl, out); torch.cuda.synchronize(); e_o = rel(out, r); nan = bool(torch.isnan(out).any())
        triton(kv, bt, q, cu, sl, out); torch.cuda.synchronize(); e_t = rel(out, r)
        good = (not nan) and e_o < 0.02 and e_o <= 1.5 * e_t + 2e-3
        ok &= good
        line = f"{label:20s} {str(dtype)[6:]:14s}: ours rel {e_o:.4f} | triton rel {e_t:.4f} | {'OK' if good else 'FAIL'}"
        if sum(qls) >= 512:
            to, tt = bench(lambda: ours(kv, bt, q, cu, sl, out)), bench(lambda: triton(kv, bt, q, cu, sl, out))
            line += f" | ours {to:7.2f} ms  triton {tt:7.2f} ms  ({tt/to:.2f}x)"
        print(line, flush=True)
print("ALL OK" if ok else "SOME FAILED")
