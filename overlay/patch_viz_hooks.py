#!/usr/bin/env python3
"""Live activation telemetry hooks for the Spark dashboard (MIMO26_VIZ=1; installs nothing otherwise).

Runtime: overlay/mimo26_viz_runtime.py -> vllm/mimo26_viz_runtime.py. Hooks, all inside code torch.compile never
traces (the MoE and attention custom ops, the eager model runner / sampler / scheduler):
  fused_moe/runner/moe_runner.py   _apply_quant_method, after select_experts -> record_routing(layer_name, topk_ids)
  attention/backends/triton_attn_diffkv.py  end of forward                   -> record_attn(layer, output)
  models/mimo_v2.py                compute_logits                            -> record_hidden3d(hidden_states)
  v1/worker/gpu/model_runner.py    after model load -> init(device); each step after record_batch -> set_batch
  v1/worker/gpu/async_utils.py     get_output, sampled ids on the host      -> record_sampled (counts only)
  v1/core/sched/scheduler.py       end of schedule()                         -> rate-limited KV pool / request snapshot
Telemetry must never cost a boot: an anchor that does not match exactly once skips that hook with a warning
(MIMO26_VIZ_STRICT=1 turns that into a failure). Idempotent.
"""
import os
import shutil
import sys
from pathlib import Path

V = Path(os.environ.get("MIMO26_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"))
RT_SRC = Path(os.environ.get("MIMO26_VIZ_RT_SRC", str(Path(__file__).with_name("mimo26_viz_runtime.py"))))
MARK = "# [mimo26-viz]"
STRICT = os.environ.get("MIMO26_VIZ_STRICT", "0") == "1"
IMPORT = "from vllm import mimo26_viz_runtime as _viz  " + MARK
skipped = []


def patch(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if f"{MARK} {label}" in text:
        print(f"{path.name}: {label} already present — skipping"); return
    n = text.count(old)
    if n != 1:
        msg = f"{path}: expected exactly one anchor for {label}, found {n}"
        if STRICT:
            raise SystemExit(msg + " — refusing (MIMO26_VIZ_STRICT=1)")
        print("WARNING: " + msg + " — hook skipped, telemetry partial"); skipped.append(label); return
    path.write_text(text.replace(old, new, 1)); print(f"patched {path.name}: {label}")


SCHED_HELPER = '''

def _mimo26_viz_sched(sched, scheduler_output):  ''' + MARK + ''' viz-sched
    """Rate-limited KV-pool / request snapshot for the dashboard (UDP JSON). Counts only, no token ids."""
    import os, time, json, socket
    if os.environ.get("MIMO26_VIZ", "0") != "1":
        return
    now = time.monotonic()
    if now - getattr(sched, "_m26_viz_ts", 0.0) < 0.25:
        return
    sched._m26_viz_ts = now
    try:
        sock = getattr(sched, "_m26_viz_sock", None)
        if sock is None:
            sock = sched._m26_viz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = os.environ.get("MIMO26_VIZ_UDP", "127.0.0.1:9103").rsplit(":", 1)
        nb = getattr(sched.cache_config, "num_gpu_blocks", 0) or 0
        sched_tok = scheduler_output.num_scheduled_tokens if scheduler_output is not None else {}
        reqs = []
        for r in list(sched.running)[:64]:
            reqs.append({"id": r.request_id[-8:], "computed": int(r.num_computed_tokens),
                         "prompt": int(r.num_prompt_tokens), "total": int(r.num_tokens),
                         "age": round(time.time() - float(r.arrival_time), 1),
                         "sched": int(sched_tok.get(r.request_id, 0))})
        # Pool accounting in block ids (all KV groups share one pool of page ids): cached = free blocks that still
        # carry a prefix hash (evictable), free = blocks without one.
        pool = {}
        try:
            bp = sched.kv_cache_manager.block_pool
            n_free_total = int(bp.get_num_free_blocks())
            n_cached = int(len(bp.cached_block_hash_to_block))
            pool = {"blocks_total": int(getattr(bp, "num_gpu_blocks", nb) or nb), "blocks_cached": n_cached,
                    "blocks_free": max(0, n_free_total - n_cached), "blocks_evictable": n_free_total}
            kc = getattr(sched, "kv_cache_config", None)
            if kc is not None:
                pool["groups"] = len(kc.kv_cache_groups)
        except Exception:
            pass
        frame = {"kind": "sched", "ts": time.time(), "usage": float(sched.kv_cache_manager.usage),
                 "pool_tokens": int(nb) * int(getattr(sched, "block_size", 0) or 0),
                 "waiting": len(sched.waiting), "reqs": reqs, **pool}
        sock.sendto(json.dumps(frame).encode(), (host, int(port)))
    except Exception:
        pass


'''


def main() -> int:
    if os.environ.get("MIMO26_VIZ", "0") != "1":
        print("patch_viz_hooks: MIMO26_VIZ != 1 — telemetry hooks not installed")
        return 0
    if not RT_SRC.exists():
        print(f"WARNING: {RT_SRC} missing — telemetry hooks not installed")
        return 1 if STRICT else 0
    shutil.copyfile(RT_SRC, V / "mimo26_viz_runtime.py"); print("installed mimo26_viz_runtime.py")

    # 1) routed experts, modular MoE path (MarlinExperts), inside the moe_forward custom op
    old = ("            topk_weights, topk_ids = self.router.select_experts(\n"
           "                hidden_states=hidden_states,\n"
           "                router_logits=router_logits,\n"
           "                topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
           "                input_ids=input_ids,\n"
           "            )\n")
    patch(V / "model_executor/layers/fused_moe/runner/moe_runner.py", old,
          old + "            " + IMPORT + " viz-routing\n"
                "            _viz.record_routing(self.layer_name, topk_ids)\n", "viz-routing")
    # 2) per-head attention output norms, end of the DiffKV backend forward (inside the attention custom op)
    old = "            token_start = token_end\n        return output\n"
    patch(V / "v1/attention/backends/triton_attn_diffkv.py", old,
          "            token_start = token_end\n"
          "        " + IMPORT + " viz-attn\n"
          "        _viz.record_attn(layer, output)\n"
          "        return output\n", "viz-attn")
    # 3) final hidden state at the sampled positions (eager, outside the graphs)
    old = "        logits = self.logits_processor(self.lm_head, hidden_states)\n        return logits\n"
    patch(V / "model_executor/models/mimo_v2.py", old,
          "        " + IMPORT + " viz-h3d\n"
          "        _viz.record_hidden3d(hidden_states)\n" + old, "viz-h3d")
    # 4a) allocate right after model load: eager, before the first CUDA graph capture
    mr = V / "v1/worker/gpu/model_runner.py"
    old = ('            "Model loading took %s GiB memory and %.6f seconds",\n'
           "            format_gib(m.consumed_memory),\n"
           "            time_after_load - time_before_load,\n"
           "        )\n")
    patch(mr, old, old + "        " + IMPORT + " viz-init\n        _viz.init(self.device)\n", "viz-init")
    # 4b) per-request token spans + the host-side transfer, every step (eager, main thread)
    old = ("        self.step_timing.record_batch(\n"
           "            input_batch, batch_desc.cg_mode == CUDAGraphMode.FULL\n"
           "        )\n")
    patch(mr, old, old + "        " + IMPORT + " viz-batch\n        _viz.set_batch(input_batch)\n", "viz-batch")
    # 5) accepted tokens per request once the D2H copy has landed (host lists; counts only)
    old = "        self.model_runner_output.sampled_token_ids = sampled_token_ids\n"
    patch(V / "v1/worker/gpu/async_utils.py", old,
          old + "        " + IMPORT + " viz-sampled\n"
                "        _viz.record_sampled(self.model_runner_output.req_ids, sampled_token_ids)\n", "viz-sampled")
    # 6) scheduler snapshot (EngineCore process, head only)
    s = V / "v1/core/sched/scheduler.py"
    text = s.read_text()
    if f"{MARK} viz-sched" in text:
        print("scheduler.py: viz-sched already present — skipping")
    else:
        needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        old = "        return scheduler_output\n"
        if text.count(needle) == 1 and text.count(old) == 1:
            text = text.replace(needle, SCHED_HELPER + needle, 1)
            text = text.replace(old, "        _mimo26_viz_sched(self, scheduler_output)  " + MARK + "\n" + old, 1)
            s.write_text(text); print("patched scheduler.py: viz-sched")
        elif STRICT:
            raise SystemExit(f"{s}: viz-sched anchors not unique — refusing (MIMO26_VIZ_STRICT=1)")
        else:
            print(f"WARNING: {s}: viz-sched anchors not unique — hook skipped"); skipped.append("viz-sched")
    if skipped:
        print("patch_viz_hooks: skipped hooks: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
