#!/usr/bin/env python3
"""NaN probe (debug): when MIMO26_NAN_PROBE=1, after the model is loaded register forward hooks on every decoder
layer and its self_attn / mlp / norms, the embeddings and lm_head; the first module whose output holds NaN/Inf
(with finite input) is logged with its name, plus per-forward first-NaN summaries for the first few forwards.
Needs --enforce-eager (hooks inside a torch.compile'd backbone do not fire). Idempotent; fails closed on drift."""
import os, sys
from pathlib import Path
MARK = "# [mimo26-nan-probe]"
VLLM = Path(os.environ.get("MIMO26_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"))
TARGETS = [  # (file, anchor logger call) -- V1 runner and the V2 runner ("Using V2 Model Runner")
    (VLLM / "v1/worker/gpu_model_runner.py", "logger.info_once("),
    (VLLM / "v1/worker/gpu/model_runner.py", "logger.info("),
]
def anchors(call):
    old = "        " + call + "\n            \"Model loading took %s GiB memory and %.6f seconds\",\n"
    new = "        _mimo26_install_nan_probe(getattr(self, \"model\", None))  " + MARK + "\n" + old
    return old, new
HELPER = '''

''' + MARK + ''' helper
def _mimo26_install_nan_probe(model):
    import os as _os, torch as _t
    if _os.environ.get("MIMO26_NAN_PROBE", "0") != "1" or model is None:
        return
    state = {"forward": 0, "reported": set(), "first_in_forward": None}
    def _bad(o):
        ts = []
        def walk(x):
            if isinstance(x, _t.Tensor):
                ts.append(x)
            elif isinstance(x, (tuple, list)):
                for y in x: walk(y)
        walk(o)
        for t in ts:
            if t.is_floating_point() and t.numel() and not _t.isfinite(t).all():
                return True, tuple(t.shape), str(t.dtype)
        return False, None, None
    def hook(name):
        def _h(mod, inp, out):
            bad_out, shape, dt = _bad(out)
            if not bad_out:
                return
            bad_in, _, _ = _bad(inp)
            key = (name, bad_in)
            if key in state["reported"]:
                return
            state["reported"].add(key)
            logger.error("[nan-probe] forward %d: %s -> non-finite OUTPUT %s %s (input %s)", state["forward"], name,
                         shape, dt, "ALSO non-finite" if bad_in else "finite  <== ORIGIN")
        return _h
    def pre_root(mod, inp):
        state["forward"] += 1
    n = 0
    for name, mod in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if (last in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm", "embed_tokens", "lm_head", "norm",
                     "qkv_proj", "o_proj", "attn", "experts", "gate", "gate_proj", "up_proj", "down_proj", "shared_experts",
                     "rotary_emb", "q_norm", "k_norm", "gate_up_proj", "act_fn", "logits_processor")
                or (last.isdigit() and ".layers." in name + ".")):
            mod.register_forward_hook(hook(name)); n += 1
    model.register_forward_pre_hook(pre_root)
    logger.warning("[nan-probe] armed on %d modules", n)
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
        t = t.replace(OLD, NEW, 1) + HELPER
        P.write_text(t); print(f"patched {P} (NaN probe hooks, MIMO26_NAN_PROBE=1 to arm)")
    return 0
if __name__ == "__main__":
    sys.exit(main())
