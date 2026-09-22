#!/usr/bin/env python3
"""page_unify_check.py: apply overlay/patch_page_unify.py to a copy of kv_cache_utils.py and run the patched
unify_kv_cache_spec_page_size on MiMo-shaped specs (TP2): NVFP4 target (180 B rows) + fp8 drafter, and fp8 target
+ fp8 drafter (must be unchanged). Prints the resulting block sizes, paddings and bytes/token/layer."""
import subprocess, sys, torch
# throwaway container: patch the installed module in place, then import it normally
subprocess.run([sys.executable, "/ov/patch_page_unify.py"], check=True)
import vllm.v1.core.kv_cache_utils as mod
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, KVQuantMode
def specs(row_bytes, target_quant):
    s = {}
    kw = dict(dtype=torch.uint8 if target_quant == KVQuantMode.NVFP4 else torch.float8_e4m3fn, kv_quant_mode=target_quant)
    for i in range(9):
        s[f"full.{i}"] = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=192, head_size_v=128,
                                           state_content_bytes=row_bytes, **kw)
    for i in range(39):
        s[f"swa.{i}"] = SlidingWindowSpec(block_size=16, num_kv_heads=4, head_size=192, head_size_v=128,
                                          sliding_window=128, state_content_bytes=row_bytes, **kw)
    for i in range(5):
        s[f"draft.{i}"] = SlidingWindowSpec(block_size=16, num_kv_heads=4, head_size=128, head_size_v=128,
                                            sliding_window=1024, dtype=torch.float8_e4m3fn,
                                            kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
    return s
for label, row, q in (("NVFP4 target 180 B rows", 180, KVQuantMode.NVFP4), ("fp8 target 320 B rows", 320, KVQuantMode.FP8_PER_TENSOR)):
    out = mod.unify_kv_cache_spec_page_size(specs(row, q))
    print(f"== {label}")
    for k in ("full.0", "swa.0", "draft.0"):
        s = out[k]; bpt = s.page_size_bytes / s.block_size / s.num_kv_heads
        print(f"  {k:8s} block {s.block_size:4d} page {s.page_size_bytes:6d} B padded={s.page_size_padded} -> {s.page_size_bytes / s.block_size:7.1f} B/token/layer")
