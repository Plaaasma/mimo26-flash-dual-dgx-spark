#!/usr/bin/env python3
"""MiMo fused-QKV pairing must survive across load_weights calls.

MiMoV2Model.load_weights pairs each layer's fp8 `qkv_proj.weight` with its `qkv_proj.weight_scale_inv` in a dict that
is created per call. The Omni wrapper loads through AutoWeightsLoader, which groups CONSECUTIVE weights by prefix
(itertools.groupby) and calls the language model's load_weights once per group. Loaders that do not yield in file
order (InstantTensor streams the small scale tensors first) split a scale from its weight into different groups:
each half returns "consumed" with nothing written, and all 48 QKV projections stay zero -> garbage output
(2026-09-22, boots 5-15). Keep the pending dict on the module instance so pairs complete across calls.
Idempotent; fails closed on drift."""
import os, sys
from pathlib import Path
MARK = "# [mimo26-qkv-pending]"
P = Path(os.environ.get("MIMO26_MIMO_V2_PY", "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py"))
OLD = "        pending_fp8_qkv_proj: dict[str, dict[str, torch.Tensor]] = {}\n"
NEW = ("        pending_fp8_qkv_proj: dict[str, dict[str, torch.Tensor]] = self.__dict__.setdefault(  " + MARK + "\n"
       "            \"_mimo26_pending_fp8_qkv_proj\", {}\n"
       "        )\n")
def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1)); print(f"patched {P.name} (fused-QKV pairing persists across load_weights calls)"); return 0
if __name__ == "__main__":
    sys.exit(main())
