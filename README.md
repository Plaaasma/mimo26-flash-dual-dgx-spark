# MiMo-V2.6-Flash-RL on 2x DGX Spark — custom vLLM kit

Serves `dealignai/MiMo-V2.6-Flash-RL-UNCENSORED` (309B/15B-active MoE, 9 global + 39 SWA-128 layers, fp8 attention/dense,
MXFP4 experts, DFlash drafter, vision + audio) on two GB10 Sparks with vLLM TP=2 over the CX7 RoCE link.
Modeled on the [GLM-5.3 EXL3 kit](https://github.com/Plaaasma/glm53-flash-dual-dgx-spark): same node roles, `.env`, `start.sh` commands, boot phases for
the dashboard, zram/reclaim memory rails, orphan-shm cleanup.

## What is custom
* **Image** `mimo26-sm121:local-*` = `vllm/vllm-openai:mimo-v26-aarch64-cu130` (arm64, vLLM 0.29.1rc1.dev449+geb8798058, torch 2.13/cu130,
  NCCL 2.30.7, InstantTensor 0.2.0). Not the cu129 variant: it compiles part of its MoE/`_C` kernels only as arch-specific
  `sm_120a`, which an sm_121 GB10 cannot execute ("no kernel image is available" in the first MoE layer); the cu130 build
  ships generic `sm_120` SASS. + `patches/` (from diffbot's sm_120 recipe; the originals match this image byte for byte):
  * `mimo_v2.py` — `cache_config` reaches the attention layers so `--kv-cache-dtype fp8` really applies (stock: silently bf16);
    global layers stay global; opt-in FP8 `o_proj` (`OPROJ_FP8=1`).
  * `triton_attn_diffkv.py` — fp8 KV with per-tensor descales, 64 split-KV segments on the global layers, mixed-batch
    partition (decode/verify rows keep split-KV next to prefill rows), dispatch of long-q rows to the custom prefill kernel.
  * `triton_unified_attention_diffkv.py` — split-KV for the spec-decode verify step (stock ran every 8-token DFlash
    verify on 18 CTAs = most of the decode step), `BLOCK_M 128 / 8 warps` prefill tiles, E4M3 K/V with fp16 dots.
  * `kernels/attention/prefill_attn.cu` — FA2-style prefill attention for the global layers at the TP2 shape
    (32/2 heads, 192/128), prebuilt for `sm_121a` at image build (`scripts/build-kernel.py`), no JIT in the server.
  * `custom_all_reduce.py` — `VLLM_CA_MAX_SIZE_MB` knob (unused on two nodes; NCCL does the all-reduce over RoCE).
* **Loader**: `--load-format instanttensor` with `overlay/patch_qkv_pending.py` (REQUIRED: upstream's MiMo fused-QKV loader pairs
  each layer's fp8 weight with its scale in a per-call dict, and the Omni auto-loader calls it once per run of consecutive
  same-prefix tensors; InstantTensor yields out of file order, so without the patch all 48 QKV projections stay zero and the
  model emits garbage), `overlay/patch_instanttensor_local.py` (each rank reads the whole checkpoint from
  its own NVMe with Direct I/O, no NCCL all-gather buffers) and `overlay/patch_load_release.py` (returns allocator slack after
  the load): 161 GB in ~40 s at 4.8 GB/s, versus 549 s for the mmap safetensors loader.
* **Launch** (`start.sh`, inner scripts in `logs/inner-*.sh`): `--moe-backend marlin` (+ `VLLM_MARLIN_INPUT_DTYPE=fp8`),
  `--linear-backend triton` (the CUTLASS c3x fp8 block-scaled GEMM vLLM picks by default throws `Error Internal` on sm_121),
  `--gpu-memory-utilization 0.80` next to the `--kv-cache-memory` pin (vLLM's startup gate compares device free memory
  against total x utilization even when the pool is pinned; default 0.92 can never pass on a Spark),
  `VLLM_USE_DEEP_GEMM=0` (DeepGEMM corrupts fp8 on SM12x), `--no-async-scheduling` (spec decode + async = garbage under
  concurrency), `cudagraph_mode FULL_DECODE_ONLY` (required by the patched attention builder), `--kv-cache-memory` pinned
  per rank (no probing on unified memory), `--generation-config auto` (the checkpoint's temperature 1.0 / top_p 0.95; no
  repetition penalty: vLLM applies it to every prompt token too, which in a coding agent suppresses the identifiers the
  model just read and has to copy verbatim into edits), audio tower skipped (`"audio":0`), images up to 800/request, 16 sequences with CUDA
  graphs up to 128 tokens (16 x 8 DFlash verify tokens), `--enable-prompt-tokens-details` (responses report
  `usage.prompt_tokens_details.cached_tokens`).
* **NVFP4 KV cache** (`--kv-cache-dtype nvfp4`, default): `patches/nvfp4_diffkv.py` writes fp4 nibbles + e4m3 scales per 16
  values (180 B/token/head vs 320 fp8); the DiffKV kernel decodes them in place for decode and spec-verify, and prefill
  on the global layers dequantizes the blocks a chunk touches once into bf16 (exact) and runs the bf16 path.
  `overlay/patch_page_unify.py` stops vLLM padding every target KV page up to the fp8 drafter's page (which made the
  NVFP4 pool smaller than fp8); `overlay/patch_flashinfer_group_dtype.py` and `patch_kv_layout_fallback.py` let the
  drafter keep its own fp8 cache.
* **Chat template**: MiMo's original (`overlay/patch_chat_template.py`, `MIMO26_THINK_PREFILL=none`). The
  dealignai UNCENSORED checkpoint differs from `XiaomiMiMo/MiMo-V2.6-Flash-RL` in one weight shard
  (`model_pp0_ep0_shard0`) and in `chat_template.jinja`, which prefills every thinking block with "The user has asked a
  specific research or creative-writing question...". In a coding agent that sentence follows every tool result and the
  model obeys it: after "continue my project" and three file reads its thinking went to Aristotle's Poetics or Shapley
  values in 4 of 4 samples; with the original template it planned the TODO items in 4 of 4 (`tests/agent_check.py`).
  Config, generation config, tokenizer, processor and DFlash files are byte-identical to the original.
* **Speculative decoding: MTP** (`SPEC_METHOD=mtp`, the checkpoint's own MTP head, 3 draft tokens). DFlash (the
  bundled 5-layer drafter, `SPEC_METHOD=dflash`) accepts 8 of 8 tokens per step on predictable text, but only for the
  first ~1,000 generated tokens of a response: counting to 3,000 (greedy) gave 7.7-8.0 per step up to 999 tokens and
  1.1-1.5 after, with or without the page-unification patch, with the draft run eagerly, and in mixed prefill/decode
  steps; the FlashInfer drafter kernel matches a reference exactly, so it is in vLLM's DFlash path (same report:
  tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe#2). DFlash also drafts from garbage context on prefix-cache hits
  (vllm#47930), which agent traffic hits almost always. MTP builds no draft context: 2.2-2.7 of 4 per step, flat to
  2,400 tokens, and the KV pool grows to 3,557,838 tokens without the drafter's cache.
* **Tool calls**: `overlay/patch_qwen3_toolparse.py` keeps string arguments that contain a literal `</parameter>`,
  `</function>` or `<parameter=` intact (the stock qwen3/mimo parser cut them at the first tag and leaked the rest into
  `content` when streaming, i.e. silently truncated file writes); normal calls parse byte-identically.
* **Scheduling**: `--long-prefill-token-threshold 2048` so a very long prompt takes at most half of each step's 4,096
  tokens; without it a 400K-token prefill (~10 min) queued every other user's new request behind it.
* **Thinking**: on by default (chat template); `chat_template_kwargs.enable_thinking` and `reasoning_effort` (none =
  off) both work, streamed or not (`tests/thinking_check.py`). Requests with a JSON `response_format` run on the
  no-thinking path (`overlay/patch_json_nothink.py`): with thinking on, ~1 in 5 strict-JSON requests otherwise came
  back with the reasoning running to max_tokens and an empty `content`.

## Commands
```
./start.sh build      # docker build (this node); BUILD_WORKER=1 builds on the worker too
./start.sh ship       # copy the image to the worker
./start.sh sync       # kit dir + model snapshot -> worker
./start.sh check      # preflight + resolved config + the docker commands, no launch
./start.sh restart    # stop + start; ./start.sh status | logs [worker] | stop
tests/smoke.sh        # chat / reasoning / tool call / vision through the API
tests/agent_check.py  # coding-agent conversation at temperature 1.0: does the thinking stay on the project?
```
GLM and MiMo cannot run at once (memory); `start.sh` refuses while a `glm53-exl3-*` container runs, and the GLM kit refuses
while :8888 is held. A socat forwarder on the other node (see the GLM kit's `host-setup/`) keeps the old API address working.

## Boot
Host-side watchdog per node (`scripts/memwatch.sh`, `MIMO26_WATCHDOG_MIB`) kills the local container before a Spark
livelocks; root crons reclaim cold pages from the containers; `scripts/boot_timeline.py` prints per-stage timings of
the last boot from the head log.

## Memory
Weights ~88 GB per rank + drafter; the KV pool is `KV_CACHE_MEMORY` per rank (fp8 ≈ 7.2 KB/token all-in at TP2:
10 GB ≈ 1.4M tokens, 12 GB ≈ 1.7M). Boot needs ≥ `MIMO26_MIN_BOOT_AVAIL_MIB` available on both nodes; the boot guard
reclaims cold pages to zram when a node drops under `MIMO26_BOOT_GUARD_MIB`. Never probe memory by trial on a Spark:
an overshoot livelocks the node (no OOM killer).

## Measured (2026-09-22, NVFP4 KV, 16 sequences, DFlash 7, tonyd's fixed prompt set v1, temperature 0)
| streams | aggregate tok/s | per-stream tok/s | mean TTFT |
|---|---|---|---|
| 1 | 40.7 | 46.5 | 0.35 s |
| 2 | 55.7 | 33.6 | 0.58 s |
| 4 | 96.4 | 29.0 | 0.57 s |
| 8 | 153.7 | 23.7 | 0.66 s |
| 12 | 198.8 | 20.0 | 0.78 s |
| 16 | 233.1 | 18.0 | 0.90 s |

Per stream at 1 stream by category: coding 69, structured 72, math 62, json 45, reasoning 34, prose 22, narrative 18
(DFlash accepts 5-6 of 7 drafts on code/structured text, ~1 on prose). The same 1-stream run on later boots the same day
gave 39.4-40.3 tok/s per stream (with and without the dashboard telemetry); boot-to-boot spread on this bench is 40-46.

Long context (unique filler document, passcode at 50% depth, NVFP4 KV): 169K tokens prefilled in 111 s (1,530 tok/s),
338K in 325 s (1,041 tok/s), 661K in 1,002 s (660 tok/s); the passcode was recalled at every length; time spent
outside the engine (tokenization, rendering) stayed under 2 s.

| | this kit (NVFP4) | this kit (fp8) | tonyd's GB10 recipe (fp8) |
|---|---|---|---|
| KV pool at a 12 GB/rank pin | 3,191,849 tokens | 1,804,514 | 1.87M at GMU 0.90 |
| 1 / 6 streams aggregate | 38.3 / 100.1 | 35.5 / 106.9 | 45.6 / 155.8 |
| cold prefill @2K / @32K | 2,012 / 1,982 tok/s | 2,283 / 2,171 | 1,947 / 1,425 |
| boot to healthy | 140-170 s | 124-155 s | ~11 min |

Where a decode step goes (torch profiler, 1 stream, ~87-116 ms per verify step of 8 tokens): Marlin MXFP4 MoE ~50%
(reads every distinct expert the 8 tokens route to; at memory bandwidth), NCCL all-reduce ~20% (waiting for the slower
rank, not the network: a 64 KB all-reduce is 50-60 us), o_proj bf16 ~10%, fp8 QKV ~8%, attention < 1%. At 6 streams
the 48 verify tokens touch ~80% of all experts, so a step reads ~60-70 GB per rank: the bandwidth floor. The worker
node here also runs a remote desktop (5-10% of its SMs), and TP lockstep runs both ranks at the slower one's pace.

## Dashboard
`dashboard/` is a zero-dependency live dashboard: `agent.py` on each node (`:9101`, GPU/memory/cpu/net plus any watched
host processes), `collector.py` on the dashboard node (`:9102`, polls the vLLM metrics and both agents every 2.5 s,
keeps 35 days of history in SQLite, receives the engine telemetry on UDP `:9103`) and `dash_server.py` (`:3000`, serves
the page and mirrors every number on it as JSON under `/api`). Run them as systemd units with `User=` your user;
configuration is by environment:

| variable | meaning (default) |
|---|---|
| `SPARK_HEAD_SSH` | `user@<head fabric IP>` when the vLLM head is the other node; its container is then reached with `docker -H ssh://…` (empty: this node) |
| `SPARK_HEAD_KITS` | serving kits that can own the head, `container:kit_dir` pairs; the running one drives the boot bar (`mimo26-head:~/mimo26/kit`) |
| `VLLM_METRICS_URL` | `http://localhost:8888/metrics` |
| `SPARK_AGENT_W` | the other node's agent, `http://<ip>:9101/stats` (`SPARK_AGENT_H` defaults to localhost) |
| `SPARK_NODE_LABELS` | role labels for the two node cards, this node first (`HEAD · API,WORKER`) |
| `SPARK_WATCH_PROCS` | agent: comma-separated command names to report, RSS + GPU memory (none) |
| `SPARK_PROC_CAP_MIB`, `SPARK_PROC_CAP_TOTAL_MIB` | flag the node card and `/api/procs` when the watched processes pass these (none) |
| `SPARK_DB_PATH`, `PORT`, `COLLECTOR` | history DB path (next to `collector.py`), page port (3000), collector URL |

**Engine telemetry** (`MIMO26_VIZ=1` and `MIMO26_VIZ_UDP=<dashboard node>:9103` in `.env`; with `MIMO26_VIZ=0` nothing
is patched). `overlay/patch_viz_hooks.py` installs `overlay/mimo26_viz_runtime.py` and hooks only code torch.compile never
traces: the MoE runner right after expert selection (routed expert ids), the end of the DiffKV attention forward
(per-head output norms), `compute_logits` (final hidden state at the sampled positions, fixed random 3-D projection),
the model runner's per-step hook (request spans, and the device-to-host transfer) and the scheduler (KV pool and request
snapshot). TP rank 0 records into preallocated device buffers, which is safe inside CUDA graph capture and replay; the
transfer is queued from the step hook at up to `MIMO26_VIZ_HZ` (10), so the publisher thread never calls CUDA. Frames
carry counts only: no token ids or text leave the engine, request ids are cut to 8 characters. A hook whose anchor moved
is skipped with a warning instead of failing the boot (`MIMO26_VIZ_STRICT=1` to fail). Cost, A/B on the fixed prompt set
at 1 stream: 39.9-40.3 tok/s per stream with telemetry, 39.4-40.1 without; the counting ceiling 71.3-71.8 vs 71.5-72.6.

What the page shows: the model as a 48-layer tower (each token's routed experts as strands across 47 plates of 256
experts; 32 head cells per layer, the 9 global-attention layers in gold with a context rail that grows with the longest
live context, the 39 sliding-window layers in teal with a fixed 128-token rail; a per-layer attention-output magnitude
spine; the DFlash drafter glowing with the tokens it got accepted), a layer x token x head volume, the final hidden
state's trajectory, per-layer magnitude over time, the KV pool per request; in 2-D the expert-routing and per-head
heatmaps. The rail has throughput (per-stream decode over the streams actually decoding), KV used of the engine's own
capacity, spec-decode acceptance by draft position, a boot stage bar with MiMo's measured stage durations, and a
memory-bandwidth estimate (experts touched per step x 6.7 MB per rank + ~5.9 GB of fixed weights + KV reads).

## Knobs worth A/B-ing (all in .env)
`SPEC_METHOD` dflash/mtp/none, `DFLASH_TOKENS` 3..7, `VLLM_DIFFKV_*`, `MARLIN_A8`, `OPROJ_FP8`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `THINKING`, `KV_CACHE_MEMORY`, `MIMO26_VIZ`.

## References
* tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark (the 150 tok/s aggregate at C6 recipe, stock kernels, TP2 fp8 KV)
* diffbot/MiMo-V2.6-Flash-RL-FP8KV-W4A8-2x-RTX-PRO-6000 (`recipe/`: the DiffKV patches and kernels used here)
* vLLM PRs #45200 (mxfp4 store), #57508 (qkv sharding), #57784 (bf16 router, Eagle3 on Omni), #58128 (fp8 KV DiffKV)
