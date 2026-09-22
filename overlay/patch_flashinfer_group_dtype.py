#!/usr/bin/env python3
"""FlashInfer metadata builder: take the KV dtype from the group's own spec, not the global cache config.

With `--kv-cache-dtype nvfp4` on the target (served by the DiffKV backend) the DFlash drafter keeps its own
fp8 KV (speculative config kv_cache_dtype); its layers run on FlashInfer, whose builder read the GLOBAL
cache dtype ("nvfp4") and refused ("requires the SM100 trtllm-gen FlashInfer path"). A group whose spec is not
NVFP4 must not inherit the target's dtype. Idempotent; fails closed on drift."""
import os, sys
from pathlib import Path
MARK = "# [mimo26-fi-group-dtype]"
P = Path(os.environ.get("MIMO26_FLASHINFER_PY", "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py"))
OLD = """        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
"""
NEW = """        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            """ + MARK + """ a non-NVFP4 group (e.g. a drafter with its own fp8 KV) must not inherit nvfp4
            if self.cache_dtype.startswith("nvfp4") and not self.kv_cache_spec.kv_quant_mode.is_nvfp4:
                self.cache_dtype = "fp8"
"""
def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1)); print(f"patched {P.name} (FlashInfer builder uses its group's KV dtype)"); return 0
if __name__ == "__main__":
    sys.exit(main())
