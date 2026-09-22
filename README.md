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
  per rank (no probing on unified memory), `--generation-config auto` + `repetition_penalty 1.05` (agents that send no
  sampling params loop otherwise), audio tower skipped (`"audio":0`), images up to 800/request.

## Commands
```
./start.sh build      # docker build (this node); BUILD_WORKER=1 builds on the worker too
./start.sh ship       # copy the image to the worker
./start.sh sync       # kit dir + model snapshot -> worker
./start.sh check      # preflight + resolved config + the docker commands, no launch
./start.sh restart    # stop + start; ./start.sh status | logs [worker] | stop
tests/smoke.sh        # chat / reasoning / tool call / vision through the API
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

## Measured (2026-09-22, tonyd's fixed prompt set v1, fp8 KV, DFlash 7, temperature 0)
| | this kit | tonyd's GB10 recipe |
|---|---|---|
| C1 aggregate / per-stream | 35.5 / 40.7 tok/s | 45.6 / 53.3 |
| C6 aggregate / per-stream | 106.9 / 21.5 tok/s | 155.8 / 31.2 |
| cold prefill @2K / @32K | 2,283 / 2,171 tok/s | 1,947 / 1,425 |
| DFlash accepted per step (coding / structured / prose) | 5.5 / 6.2 / 1.1 | 5.2 / 6.9 / 1.2 |
| KV pool (10 GB pin, fp8) | 1,503,764 tokens | 1.87M at GMU 0.90 |
| boot to healthy | 124-155 s | ~11 min |

Where a decode step goes (torch profiler, 1 stream, ~87-116 ms per verify step of 8 tokens): Marlin MXFP4 MoE ~50%
(reads every distinct expert the 8 tokens route to; at memory bandwidth), NCCL all-reduce ~20% (waiting for the slower
rank, not the network: a 64 KB all-reduce is 50-60 us), o_proj bf16 ~10%, fp8 QKV ~8%, attention < 1%. At 6 streams
the 48 verify tokens touch ~80% of all experts, so a step reads ~60-70 GB per rank: the bandwidth floor. The worker
node here also runs a remote desktop (5-10% of its SMs), and TP lockstep runs both ranks at the slower one's pace.

## Knobs worth A/B-ing (all in .env)
`SPEC_METHOD` dflash/mtp/none, `DFLASH_TOKENS` 3..7, `VLLM_DIFFKV_*`, `MARLIN_A8`, `OPROJ_FP8`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `THINKING`, `KV_CACHE_MEMORY`.

## References
* tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark (the 150 tok/s aggregate at C6 recipe, stock kernels, TP2 fp8 KV)
* diffbot/MiMo-V2.6-Flash-RL-FP8KV-W4A8-2x-RTX-PRO-6000 (`recipe/`: the DiffKV patches and kernels used here)
* vLLM PRs #45200 (mxfp4 store), #57508 (qkv sharding), #57784 (bf16 router, Eagle3 on Omni), #58128 (fp8 KV DiffKV)
