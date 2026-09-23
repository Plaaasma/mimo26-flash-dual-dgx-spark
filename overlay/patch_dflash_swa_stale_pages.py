#!/usr/bin/env python3
"""DFlash drafter: never let XQA read freed (out-of-window) KV pages. Fixes the DFlash acceptance cliff.

Symptom: DFlash acceptance is 8/8 per step until the sequence passes the drafter's 1,024-token sliding window, then
exactly 1 token per step (no draft ever accepted) from ~1,070 tokens on (counting 1..3000, greedy); long agent turns
ran at ~1 token/step. Reported upstream on other images too (tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe#2).

Cause: on SM12x the non-causal drafter runs FlashInfer XQA spec-decode (8 draft rows, window 1024, sinks). Once the
sequence outgrows the window, the scheduler frees the drafter's oldest pages (the worker's block-table row keeps their
ids), and the shared pool hands them to the target's NVFP4 layers, whose packed bytes read as fp8 e4m3 include NaN
encodings (0x7F/0xFF). XQA skips out-of-window KV only in whole CTA tiles: the window-edge tile still loads the few
stale pages below the window, masks their scores, and multiplies P = 0 by V = NaN -> NaN. The drafter's attention
output was NaN on every step after the cliff (live check: hundreds of non-finite fp8 values in the below-window pages,
none in-window; XQA output NaN; the same call with those entries redirected was finite and matched an exact reference
to 1e-3). The trailing tile is load-guarded, the leading window edge is not.

Fix: before the draft attention metadata is built (and before a captured draft graph replays), point every block-table
entry that lies entirely below the first draft row's window at that row's first in-window page. Those positions are
masked anyway, so the result is unchanged except that XQA now loads finite, drafter-written fp8 there. Only the
per-step gathered copy (input_block_tables) of the drafter's own sliding-window groups is touched; it is re-gathered
from the persistent table every step. Gate: MIMO26_DFLASH_STALE_FIX (default 1). Idempotent; fails closed on drift.
"""
import os
import sys
from pathlib import Path

MARK = "# [mimo26-dflash-swa-stale]"
P = Path(os.environ.get("MIMO26_DFLASH_SPECULATOR_PY",
                        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"))
OLD = """        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_uniform_attn_metadata(
"""
NEW = """        if not dummy_run:  """ + MARK + """
            _mimo26_redirect_stale_swa_pages(self, num_reqs)

        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_uniform_attn_metadata(
"""
HELPER = '''


def _mimo26_redirect_stale_swa_pages(spec, num_reqs: int) -> None:  ''' + MARK + '''
    """Point block-table entries wholly below each request's draft window at its first in-window page.

    Freed sliding-window pages keep their ids in the worker's row and may since hold another layer's (e.g. NVFP4)
    bytes, which decode to NaN as fp8. XQA masks those positions but still loads the window-edge tile, and
    0 * NaN poisons the output. Masked positions only need finite data, so reuse an in-window page.
    """
    import os as _os

    if _os.environ.get("MIMO26_DFLASH_STALE_FIX", "1") != "1":
        return
    groups = getattr(spec, "_mimo26_swa_groups", None)
    if groups is None:
        groups = []
        for gid in spec.draft_kv_cache_group_ids:
            window = get_kv_cache_spec_sliding_window(
                spec.kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            )
            if window:
                groups.append((gid, int(window), int(spec.block_tables.kernel_block_sizes[gid])))
        spec._mimo26_swa_groups = groups
        if groups:
            logger.info("[mimo26-dflash-swa-stale] redirecting out-of-window draft pages for groups %s", groups)
    if not groups or num_reqs <= 0:
        return
    # First draft row's position = seq_len - num_query_per_req (seq_lens written by prepare_dflash_inputs).
    first_row = spec.input_buffers.seq_lens[:num_reqs].to(torch.int64) - spec.num_query_per_req
    for gid, window, block_size in groups:
        table = spec.block_tables.input_block_tables[gid]
        width = min(table.shape[1], -(-int(spec.draft_max_seq_len) // block_size))
        if width <= 0:
            continue
        # Blocks [0, n) end before first_row - (window - 1), the earliest position any draft row attends.
        n = torch.clamp((first_row - (window - 1)) // block_size, min=0, max=width - 1)
        rows = table[:num_reqs, :width]
        first_in_window = rows.gather(1, n[:, None])
        cols = torch.arange(width, device=table.device)[None, :]
        rows.copy_(torch.where(cols < n[:, None], first_in_window, rows))
'''


def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    for need in ("get_kv_cache_spec_sliding_window", "import torch\n", "logger = init_logger"):
        if need not in t:
            raise SystemExit(f"{P}: missing {need!r} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1) + HELPER)
    print(f"patched {P.name} (DFlash: out-of-window draft pages redirected; XQA no longer reads freed pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
