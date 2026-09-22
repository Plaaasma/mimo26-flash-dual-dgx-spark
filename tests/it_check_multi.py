#!/usr/bin/env python3
"""it_check_multi.py <shard1> <shard2> ...: open several files in ONE InstantTensor session (as vLLM does with all 65
shards), keep every yielded tensor alive until the end (as vLLM's fused-QKV loader keeps a weight until its scale
arrives), then compare all of them to safetensors CPU reads. Detects buffer reuse across tensors/files."""
import sys, torch, instanttensor
from safetensors import safe_open
files = sys.argv[1:]; held = []; bad = 0; n = 0
with instanttensor.safe_open(files, framework="pt", device=torch.device("cuda:0"), process_group=None, copy=True) as it:
    for name, t in it.tensors():
        held.append((name, t)); n += 1
torch.cuda.synchronize()
refs = {}; handles = [safe_open(f, framework="pt", device="cpu") for f in files]
for f, sf in zip(files, handles):
    for k in sf.keys(): refs[k] = (f, sf)
first_bad = None
for i, (name, t) in enumerate(held):
    f, sf = refs[name]; ref = sf.get_tensor(name)
    a = t.contiguous().cpu(); a = a.view(torch.uint8) if a.dtype != torch.bool else a
    b = ref.view(torch.uint8) if ref.dtype != torch.bool else ref
    if a.shape != b.shape or not torch.equal(a, b):
        bad += 1
        if first_bad is None: first_bad = i
        if bad <= 5: print(f"DIFF #{i} {name} {t.dtype} {tuple(t.shape)} from {f.split('/')[-1]}")
print(f"{n} tensors held from {len(files)} files; {bad} mismatches; first bad index {first_bad} ->", "ALL IDENTICAL" if bad == 0 else "BUFFER REUSE / CORRUPTION")
