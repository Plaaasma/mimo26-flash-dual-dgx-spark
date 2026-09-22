#!/usr/bin/env python3
"""it_check_all.py <snapshot_dir>: one InstantTensor session over ALL weight shards (what vLLM does), streaming
per-tensor comparison against safetensors CPU reads. Exercises the real io_depth / buffer scheduling."""
import sys, os, glob, time, torch, instanttensor
from safetensors import safe_open
D = sys.argv[1]
files = sorted(glob.glob(os.path.join(D, "model_pp0_ep*_shard0.safetensors"))) + [os.path.join(D, "model_mtp.safetensors")]
files = [os.path.realpath(f) for f in files]
print(f"{len(files)} files", flush=True)
handles = {f: safe_open(f, framework="pt", device="cpu") for f in files}
owner = {k: f for f, h in handles.items() for k in h.keys()}
n = bad = 0; t0 = time.time(); per_file = {}
with instanttensor.safe_open(files, framework="pt", device=torch.device("cuda:0"), process_group=None, copy=True) as it:
    for name, t in it.tensors():
        n += 1; f = owner.get(name)
        if f is None:
            print("EXTRA", name, flush=True); bad += 1; continue
        ref = handles[f].get_tensor(name)
        a = t.contiguous().cpu(); a = a.view(torch.uint8) if a.dtype != torch.bool else a
        b = ref.view(torch.uint8) if ref.dtype != torch.bool else ref
        if a.shape != b.shape or not torch.equal(a, b):
            bad += 1; per_file[f] = per_file.get(f, 0) + 1
            if bad <= 8: print(f"DIFF {name} {t.dtype} {tuple(t.shape)} in {os.path.basename(f)}", flush=True)
        del t
        if n % 10000 == 0: print(f"  {n} tensors, {bad} bad, {time.time() - t0:.0f}s", flush=True)
print(f"{n} tensors over {len(files)} files in {time.time() - t0:.0f}s: {bad} mismatches", {os.path.basename(k): v for k, v in per_file.items()}, "->", "ALL IDENTICAL" if bad == 0 else "CORRUPTION AT SCALE")
