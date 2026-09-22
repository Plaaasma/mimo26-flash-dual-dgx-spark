#!/usr/bin/env python3
"""CacheConfig.get_resolved_kv_cache_layout: fall back to an explicit VLLM_KV_CACHE_LAYOUT.

A speculative config with its own kv_cache_dtype makes the proposer copy the target's CacheConfig BEFORE the engine
core resolves the KV layout; the copy never receives the resolved name, and the drafter's FlashInfer forward then
raises "KV cache layout has not been resolved yet". With the layout pinned in the environment (the kit sets
VLLM_KV_CACHE_LAYOUT) the copy can answer from it. Idempotent; fails closed on drift."""
import os, sys
from pathlib import Path
MARK = "# [mimo26-kv-layout-fallback]"
P = Path(os.environ.get("MIMO26_CACHE_PY", "/usr/local/lib/python3.12/dist-packages/vllm/config/cache.py"))
OLD = """    def get_resolved_kv_cache_layout(self) -> KVCacheLayout:
        if self.kv_cache_layout is None:
            raise ValueError(
"""
NEW = """    def get_resolved_kv_cache_layout(self) -> KVCacheLayout:
        if self.kv_cache_layout is None:  """ + MARK + """
            import os as _os
            _req = _os.environ.get("VLLM_KV_CACHE_LAYOUT")
            if _req:
                return _layout_from_name(_req)
        if self.kv_cache_layout is None:
            raise ValueError(
"""
def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1)); print(f"patched {P.name} (KV layout falls back to VLLM_KV_CACHE_LAYOUT)"); return 0
if __name__ == "__main__":
    sys.exit(main())
