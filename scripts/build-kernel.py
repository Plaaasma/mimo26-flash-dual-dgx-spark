#!/usr/bin/env python3
"""Prebuild the custom DiffKV prefill-attention kernel (kernels/attention/prefill_attn.cu) at image build time.

torch.utils.cpp_extension.load() is called with exactly the arguments the patched
triton_attn_diffkv.py uses at runtime, so the serving container finds an up-to-date
ninja build and never compiles in-process (no JIT stall on the first prefill).
Needs nvcc only, no GPU. Keep TORCH_CUDA_ARCH_LIST identical at build and run time.
"""
import os
import sys

from torch.utils.cpp_extension import load

build = os.environ.get("ATTN_CUSTOM_BUILD", "/opt/mimo26/attn-build")
src = os.environ.get("ATTN_CUSTOM_SRC", "/opt/mimo26/kernels/attention/prefill_attn.cu")
gencode = os.environ.get("ATTN_CUSTOM_GENCODE", "-gencode=arch=compute_121a,code=sm_121a").split()
os.makedirs(build, exist_ok=True)
ext = load(
    name="prefill_attn_ext",
    sources=[src],
    extra_cuda_cflags=["-O3", "--use_fast_math"] + gencode,
    build_directory=build,
    verbose=True,
)
print("prefill_attn_ext built:", ext.__file__, file=sys.stderr)
