#!/usr/bin/env python3
"""it_check_race.py <shard>: snapshot each InstantTensor tensor the instant it is yielded (device clone on the default
stream, no sync — what vLLM's param.copy_ does), then compare the snapshots to safetensors reads. A mismatch here
with identical bytes in it_check.py means the yielded tensor is still being filled asynchronously (stream race)."""
import sys, torch, instanttensor
from safetensors import safe_open
f = sys.argv[1]; snaps = []; n = 0
with instanttensor.safe_open([f], framework="pt", device=torch.device("cuda:0"), process_group=None, copy=True) as it:
    for name, t in it.tensors():
        snaps.append((name, t.clone(), t)); n += 1          # clone immediately, no synchronization
torch.cuda.synchronize()
bad_snap = bad_late = 0; first = None
with safe_open(f, framework="pt", device="cpu") as sf:
    for i, (name, snap, t) in enumerate(snaps):
        ref = sf.get_tensor(name); b = ref.view(torch.uint8) if ref.dtype != torch.bool else ref
        a = snap.cpu(); a = a.view(torch.uint8) if a.dtype != torch.bool else a
        c = t.cpu(); c = c.view(torch.uint8) if c.dtype != torch.bool else c
        if not torch.equal(a, b):
            bad_snap += 1
            if first is None: first = (i, name, tuple(snap.shape))
        if not torch.equal(c, b): bad_late += 1
print(f"{n} tensors: immediate-clone mismatches {bad_snap}, late mismatches {bad_late}, first bad {first} ->",
      "STREAM RACE" if bad_snap and not bad_late else ("ALL IDENTICAL" if not bad_snap and not bad_late else "CORRUPTION"))
