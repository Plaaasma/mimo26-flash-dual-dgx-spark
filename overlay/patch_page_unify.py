#!/usr/bin/env python3
"""KV page unification that scales the model's own pages instead of padding them to a drafter's page.

vLLM's hybrid allocator needs one page size for every KV group. Stock `unify_kv_cache_spec_page_size` takes the
LARGEST page and, for each smaller page, scales the block size when the page divides it, else pads the page.
With the NVFP4 target cache (180 B rows) next to the fp8 DFlash drafter, the drafter's page (16 x 1024 B = 16,384 B)
is the largest and neither target page (5,760 B full attention, 11,520 B sliding window) divides it, so all 48
target layers were padded to 16,384 B per 16 tokens = 1,024 B/token/layer, worse than fp8 (640): the pool fell from
1.80M (fp8) to 1.15M tokens (boot 31).

Here the common page is the smallest multiple of the page chain carried by most layers (pages that divide one another,
taken in order of layer count) that is >= the largest page: 23,040 B for MiMo NVFP4 (full attention 64-token blocks,
sliding window 32), and only the 5 drafter layers are padded. When that page equals the stock choice (e.g. fp8,
where the target's 20,480-B page is already the largest) the stock code runs unchanged.
Kill switch: MIMO26_PAGE_UNIFY=0. Idempotent; fails closed on anchor drift.
"""
import os
import sys
from pathlib import Path

MARK = "# [mimo26-page-unify]"
P = Path(os.environ.get("MIMO26_KV_CACHE_UTILS_PY",
                        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"))
OLD = """    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
"""
NEW = """    _m26_page = _mimo26_common_page(kv_cache_spec)  """ + MARK + """
    if _m26_page is not None:
        return _mimo26_unify_to(kv_cache_spec, _m26_page)
    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
"""
HELPER = '''


''' + MARK + ''' helpers
def _mimo26_common_page(kv_cache_spec):
    """Common page that the majority page chain scales into, or None when the stock rule gives the same page."""
    import collections
    import os as _os
    if _os.environ.get("MIMO26_PAGE_UNIFY", "1") != "1":
        return None
    counts = collections.Counter(
        s.page_size_bytes for s in kv_cache_spec.values() if not isinstance(s, MambaSpec)
    )
    if len(counts) <= 1:
        return None
    max_page = max(s.page_size_bytes for s in kv_cache_spec.values())
    chain = None
    for page, _n in sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0])):
        if chain is None or chain % page == 0:
            chain = chain or page
        elif page % chain == 0:
            chain = page
    common = -(-max_page // chain) * chain
    if common == max_page:
        return None
    logger.info(
        "[mimo26-page-unify] common KV page %d B instead of %d B (pages by layer count: %s)",
        common, max_page, dict(counts),
    )
    return common


def _mimo26_unify_to(kv_cache_spec, common):
    new = {}
    for name, spec in kv_cache_spec.items():
        page = spec.page_size_bytes
        if page == common:
            new_spec = spec
        elif isinstance(spec, MambaSpec):
            new_spec = replace(spec, page_size_padded=common)
        elif common % page == 0 and getattr(spec, "page_size_padded", None) is None:
            new_spec = replace(spec, block_size=spec.block_size * (common // page))
        elif isinstance(spec, AttentionSpec) and not isinstance(spec, MLAAttentionSpec):
            new_spec = replace(spec, page_size_padded=common)
        else:
            raise NotImplementedError(
                f"Layer {name}: page {page} B cannot be scaled or padded to {common} B"
            )
        assert new_spec.page_size_bytes == common, (name, new_spec.page_size_bytes, common)
        new[name] = new_spec
    return new
'''


def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1) + HELPER)
    print(f"patched {P.name} (KV page unification scales target pages, pads the drafter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
