#!/usr/bin/env python3
"""it_check.py <shard.safetensors>: every tensor read through InstantTensor (device, rank-local, copy=True — the kit's
loader path) must be bit-identical to the safetensors CPU read of the same file. Streams tensor by tensor (a 13 GB
shard fits next to a running server); also reports strides/contiguity and the tensor order/count."""
import sys, torch, instanttensor
from safetensors import safe_open
f = sys.argv[1]; bad = 0; n = 0; bytes_ = 0; noncontig = 0; names_it = []
sf = safe_open(f, framework="pt", device="cpu"); keys = set(sf.keys())
with instanttensor.safe_open([f], framework="pt", device=torch.device("cuda:0"), process_group=None, copy=True) as it:
    for name, t in it.tensors():
        names_it.append(name); n += 1
        if name not in keys:
            print("EXTRA", name); bad += 1; continue
        ref = sf.get_tensor(name); bytes_ += ref.numel() * ref.element_size()
        if t.dtype != ref.dtype or tuple(t.shape) != tuple(ref.shape):
            print("SHAPE/DTYPE", name, t.dtype, tuple(t.shape), "vs", ref.dtype, tuple(ref.shape)); bad += 1; continue
        if not t.is_contiguous(): noncontig += 1
        a = t.contiguous().cpu(); a = a.view(torch.uint8) if a.dtype != torch.bool else a
        b = ref.view(torch.uint8) if ref.dtype != torch.bool else ref
        if not torch.equal(a, b):
            print("DIFF", name, t.dtype, tuple(t.shape)); bad += 1
        del t
missing = keys - set(names_it)
for m in sorted(missing)[:10]: print("MISSING", m)
print(f"{n} tensors from InstantTensor ({len(keys)} in file), {bytes_ / 2**30:.2f} GiB compared, {bad} mismatches, {len(missing)} missing, {noncontig} non-contiguous ->", "ALL IDENTICAL" if bad == 0 and not missing else "MISMATCH")
sys.exit(1 if (bad or missing) else 0)
