#!/usr/bin/env python3
"""Parameter checksum probe (debug): when MIMO26_PARAM_SUMS=<tag>, after the model is loaded every rank writes
/root/.cache/vllm/param_sums_<tag>_rank<r>.json = {param name: [shape, dtype, byte-sum, float-sum-or-null, first 4 values]}.
Diffing two tags (e.g. instanttensor vs safetensors loader) pins exactly which parameters differ. Same anchors as
patch_nan_probe.py (both runners); idempotent; fails closed on drift."""
import os, sys
from pathlib import Path
MARK = "# [mimo26-param-sums]"
VLLM = Path(os.environ.get("MIMO26_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"))
TARGETS = [(VLLM / "v1/worker/gpu_model_runner.py", "logger.info_once("), (VLLM / "v1/worker/gpu/model_runner.py", "logger.info(")]
def anchors(call):
    old = "        " + call + "\n            \"Model loading took %s GiB memory and %.6f seconds\",\n"
    new = "        _mimo26_param_sums(getattr(self, \"model\", None))  " + MARK + "\n" + old
    return old, new
HELPER = '''

''' + MARK + ''' helper
def _mimo26_param_sums(model):
    import os as _os, json as _json, torch as _t
    tag = _os.environ.get("MIMO26_PARAM_SUMS", "")
    if not tag or model is None:
        return
    try:
        from vllm.distributed import get_tensor_model_parallel_rank as _r
        rank = _r()
    except Exception:
        rank = 0
    out = {}
    with _t.no_grad():
        for name, p in model.named_parameters():
            d = p.data
            try:
                flat = d.contiguous().flatten()
                bs = int(flat.view(_t.uint8).to(_t.int64).sum().item()) if d.numel() else 0
                fs = None; head = None
                if d.is_floating_point():
                    fs = float(flat.float().sum().item()); head = [float(x) for x in flat[:4].float().tolist()]
                else:
                    head = [int(x) for x in flat[:4].tolist()]
                out[name] = [list(d.shape), str(d.dtype), bs, fs, head]
            except Exception as e:
                out[name] = [list(d.shape), str(d.dtype), None, None, str(e)[:80]]
        for name, b in model.named_buffers():
            d = b.data
            if d.numel() == 0 or not d.is_floating_point():
                continue
            try:
                flat = d.contiguous().flatten()
                out["BUF:" + name] = [list(d.shape), str(d.dtype), int(flat.view(_t.uint8).to(_t.int64).sum().item()), float(flat.float().sum().item()), [float(x) for x in flat[:4].float().tolist()]]
            except Exception:
                pass
    path = f"/root/.cache/vllm/param_sums_{tag}_rank{rank}.json"
    with open(path, "w") as f:
        _json.dump(out, f)
    logger.warning("[param-sums] wrote %d entries to %s", len(out), path)
'''
def main() -> int:
    for P, call in TARGETS:
        if not P.exists():
            print(f"{P}: absent — skipping"); continue
        t = P.read_text()
        if MARK in t:
            print(f"{P.name}: {MARK} already present — skipping"); continue
        OLD, NEW = anchors(call)
        if t.count(OLD) != 1:
            raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
        P.write_text(t.replace(OLD, NEW, 1) + HELPER); print(f"patched {P} (param sums, MIMO26_PARAM_SUMS=<tag> to arm)")
    return 0
if __name__ == "__main__":
    sys.exit(main())
