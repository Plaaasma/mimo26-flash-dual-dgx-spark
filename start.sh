#!/usr/bin/env bash
# ============================================================================
# start.sh — MiMo-V2.6-Flash-RL on 2x DGX Spark (GB10 / SM121): vLLM TP=2 over CX7 RoCE
# ============================================================================
#   head   : this machine (HEAD_IP) — vLLM rank 0 + OpenAI API on :PORT
#   worker : WORKER_SSH (WORKER_IP) — vLLM rank 1, --headless
#   image  : IMAGE built from ./Dockerfile (base vllm/vllm-openai:mimo-v26-cu129 + patches/, see README)
#   weights: HF cache snapshot MODEL@MODEL_REVISION, same path on both nodes
#
# Usage:
#   ./start.sh                start (preflight, launch, wait for health)
#   ./start.sh restart        stop + start
#   ./start.sh stop           remove both containers
#   ./start.sh status         containers + API health
#   ./start.sh logs [worker]  follow head (or worker) container logs
#   ./start.sh build          docker build IMAGE here (BUILD_WORKER=1: also on the worker)
#   ./start.sh ship           docker save IMAGE | ssh worker docker load (when the worker lacks it)
#   ./start.sh sync           rsync this kit dir + the model snapshot head -> worker
#   ./start.sh check          preflight + print the resolved config and the docker commands (no launch)
#
# Settings live in .env (copied from .env.example on first run); any variable can be overridden
# on the command line (KV_CACHE_MEMORY=12000000000 ./start.sh restart). DRY_RUN=1 prints instead of running.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"
if [ ! -f .env ]; then
    [ -f .env.example ] || { echo "ERROR: missing .env.example" >&2; exit 1; }
    cp .env.example .env
    echo "[mimo26] wrote .env from .env.example — edit HEAD_IP / WORKER_IP / WORKER_SSH, then rerun"
    exit 1
fi
# command-line environment wins over .env
_cli="$(env | grep -E '^[A-Z][A-Z0-9_]*=' | grep -vE '^(PATH|HOME|USER|SHELL|PWD|OLDPWD|TERM|LANG|LC_[A-Z]+|SSH_[A-Z_]+|XDG_[A-Z_]+|DISPLAY|LOGNAME|MAIL|HOSTNAME|SHLVL|_|TMPDIR|DBUS_[A-Z_]+|DEBUGINFOD_URLS|LS_COLORS|COLORTERM|GDMSESSION|DESKTOP_SESSION|SESSION_MANAGER|WINDOWPATH|GNOME_[A-Z_]+|QT_[A-Z_]+|GTK_[A-Z_]+|IM_CONFIG_PHASE|MOTD_SHOWN|LESSOPEN|LESSCLOSE|SYSTEMD_EXEC_PID|INVOCATION_ID|JOURNAL_STREAM|MANAGERPID|CLAUDE[A-Z_]*|CLAUDECODE|NVM_[A-Z_]+|GIT_[A-Z_]+|HF_HOME|HOST)=' || true)"
set -a; # shellcheck disable=SC1091
source ./.env; set +a
if [ -n "$_cli" ]; then
    while IFS= read -r kv; do
        k="${kv%%=*}"; v="${kv#*=}"
        export "$k=$v"
    done <<< "$_cli"
fi

# ------------------------------- defaults ----------------------------------
HEAD_IP="${HEAD_IP:?set HEAD_IP in .env}"; WORKER_IP="${WORKER_IP:?set WORKER_IP in .env}"
WORKER_SSH="${WORKER_SSH:-$USER@$WORKER_IP}"
HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"; WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f1np1}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"; WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f1}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
MODEL="${MODEL:?set MODEL}"; MODEL_REVISION="${MODEL_REVISION:?set MODEL_REVISION}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-65}"
IMAGE="${IMAGE:-mimo26-sm121:local}"; BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:mimo-v26-cu129}"
PORT="${PORT:-8888}"; SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-v2.6-flash}"
TP="${TP:-2}"; NNODES="${NNODES:-2}"; MASTER_PORT="${MASTER_PORT:-29522}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-300000}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"; MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
SPEC_METHOD="${SPEC_METHOD:-dflash}"; DFLASH_TOKENS="${DFLASH_TOKENS:-7}"; MTP_TOKENS="${MTP_TOKENS:-3}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"; LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"; CAPTURE_SIZES="${CAPTURE_SIZES:-1 2 4 8 16 32 64}"
ASYNC_SCHED="${ASYNC_SCHED:-0}"; REP_PENALTY="${REP_PENALTY:-1.0}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"; SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
if [ -z "${LIMIT_MM:-}" ]; then LIMIT_MM='{"image":16,"video":1,"audio":0}'; fi
if [ -z "${MM_PROC_KWARGS:-}" ]; then MM_PROC_KWARGS='{"max_pixels":2073600}'; fi
READY_TIMEOUT="${READY_TIMEOUT:-3600}"; NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"; FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
MIMO26_MIN_BOOT_AVAIL_MIB="${MIMO26_MIN_BOOT_AVAIL_MIB:-104000}"

CONTAINER_HEAD="${CONTAINER_HEAD:-mimo26-head}"; CONTAINER_WORKER="${CONTAINER_WORKER:-mimo26-worker}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
WORKER_HOME="${WORKER_HOME:-/home/${WORKER_SSH%%@*}}"
WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-$WORKER_HOME/.cache/huggingface}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-mimo26}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-mimo26}"
WORKER_KIT_DIR="${WORKER_KIT_DIR:-$WORKER_HOME/mimo26/kit}"
LOGDIR="$SCRIPT_DIR/logs"; mkdir -p "$LOGDIR"
HEAD_SCRIPT="$LOGDIR/inner-head.sh"; WORKER_SCRIPT="$LOGDIR/inner-worker.sh"
SSH_MUX="-o ControlMaster=auto -o ControlPath=/tmp/mimo26-mux-%C -o ControlPersist=120"
BOOT_ID="$(date +%Y%m%d-%H%M%S)"
MODEL_SUBDIR="models--${MODEL//\//--}"
MODEL_HOST_DIR="$HF_CACHE_DIR/hub/$MODEL_SUBDIR/snapshots/$MODEL_REVISION"
MODEL_DIR="/root/.cache/huggingface/hub/$MODEL_SUBDIR/snapshots/$MODEL_REVISION"   # in-container
DRY_RUN="${DRY_RUN:-}"

log()  { printf '\033[1;36m[mimo26]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[mimo26]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[mimo26]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }
# boot-state.json: same schema as the GLM kit so the cluster dashboard's boot bar can follow this kit
boot_phase() { printf '{"phase":"%s","t":%s,"boot_id":"%s","note":"%s"}\n' "$1" "$(date +%s.%N)" "$BOOT_ID" "${2:-}" > "$LOGDIR/boot-state.json.tmp" 2>/dev/null && mv -f "$LOGDIR/boot-state.json.tmp" "$LOGDIR/boot-state.json" 2>/dev/null || true; }
worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 $SSH_MUX "$WORKER_SSH" "$@"; }
run() { if [ -n "$DRY_RUN" ]; then printf '  DRY: %q' "$@"; echo; else "$@"; fi; }

# ------------------------------ preflight ----------------------------------
count_shards() { ls "$1"/model_pp0_ep*_shard0.safetensors "$1"/model_mtp.safetensors 2>/dev/null | wc -l; }
check_model_here() {
    local d="$1" where="$2"
    [ -f "$d/config.json" ] || die "$where: $d/config.json missing (run the download first, then ./start.sh sync)"
    local n; n=$(count_shards "$d")
    [ "$n" -ge "$EXPECTED_SHARDS" ] || die "$where: $n of $EXPECTED_SHARDS weight files in $d"
    if ls "$(dirname "$(dirname "$d")")"/blobs/*.incomplete >/dev/null 2>&1; then die "$where: incomplete blobs under $(dirname "$(dirname "$d")")/blobs"; fi
    [ "$SPEC_METHOD" != dflash ] || [ -f "$d/dflash/config.json" ] || die "$where: $d/dflash/config.json missing"
}
preflight() {
    command -v docker >/dev/null || die "docker missing on the head"
    worker_ssh true 2>/dev/null || die "cannot ssh to $WORKER_SSH (need passwordless ssh, ControlMaster ok)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing here: ./start.sh build"
    worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1" || die "image $IMAGE missing on the worker: ./start.sh ship"
    local hd wd; hd=$(docker image inspect -f '{{.Id}}' "$IMAGE"); wd=$(worker_ssh "docker image inspect -f '{{.Id}}' '$IMAGE'")
    [ "$hd" = "$wd" ] || die "image $IMAGE differs between head ($hd) and worker ($wd): ./start.sh ship"
    check_model_here "$MODEL_HOST_DIR" head
    worker_ssh "bash -s" <<EOS || die "worker model check failed (./start.sh sync)"
set -e; d='$WORKER_CACHE_DIR/hub/$MODEL_SUBDIR/snapshots/$MODEL_REVISION'
[ -f "\$d/config.json" ] || { echo "worker: \$d/config.json missing"; exit 1; }
n=\$(ls "\$d"/model_pp0_ep*_shard0.safetensors "\$d"/model_mtp.safetensors 2>/dev/null | wc -l)
[ "\$n" -ge $EXPECTED_SHARDS ] || { echo "worker: \$n of $EXPECTED_SHARDS weight files"; exit 1; }
! ls "\$(dirname "\$(dirname "\$d")")"/blobs/*.incomplete >/dev/null 2>&1 || { echo "worker: incomplete blobs"; exit 1; }
EOS
    # one model server at a time: the GLM kit owns :PORT when its containers run
    local conflict=die; [ -n "$DRY_RUN" ] && conflict=warn   # `check` reports conflicts instead of aborting
    if docker ps --format '{{.Names}}' | grep -qE '^glm53-exl3-head$'; then
        $conflict "glm53-exl3-head is running (GLM-5.3 owns :$PORT). Stop it first: (cd ~/glm53/exl3-kit && ./start.sh stop) — only with the operator's go"
    fi
    if worker_ssh "docker ps --format '{{.Names}}'" | grep -qE '^glm53-exl3-(head|worker)$'; then
        $conflict "a glm53-exl3 container is running on the worker; stop the GLM stack first"
    fi
    if ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; then
        docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_HEAD$" || $conflict "port $PORT is held by another process on the head"
    fi
    local ha wa
    ha=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    wa=$(worker_ssh "awk '/MemAvailable/{printf \"%d\", \$2/1024}' /proc/meminfo")
    log "MemAvailable: head ${ha} MiB, worker ${wa} MiB (need >= ${MIMO26_MIN_BOOT_AVAIL_MIB} on both once the old containers are gone)"
    PRE_HEAD_AVAIL=$ha; PRE_WORKER_AVAIL=$wa
}
wait_for_headroom() {
    # after teardown: both nodes must have room for weights + KV + activations, otherwise vLLM dies mid-load
    # (and on Spark an overshoot livelocks the node). Waits up to 10 min for co-tenant processes to shrink.
    local i ha wa
    for i in $(seq 1 60); do
        ha=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
        wa=$(worker_ssh "awk '/MemAvailable/{printf \"%d\", \$2/1024}' /proc/meminfo" 2>/dev/null || echo 0)
        if [ "$ha" -ge "$MIMO26_MIN_BOOT_AVAIL_MIB" ] && [ "$wa" -ge "$MIMO26_MIN_BOOT_AVAIL_MIB" ]; then
            log "headroom ok: head ${ha} MiB, worker ${wa} MiB"; return 0
        fi
        [ $((i % 6)) -eq 1 ] && warn "waiting for headroom: head ${ha} MiB, worker ${wa} MiB (need ${MIMO26_MIN_BOOT_AVAIL_MIB}; other processes hold the rest)"
        sleep 10
    done
    die "not enough free memory to boot (head ${ha} MiB, worker ${wa} MiB; need ${MIMO26_MIN_BOOT_AVAIL_MIB} each)"
}

# --------------------------- image / weights -------------------------------
build() {
    log "building $IMAGE from $BASE_IMAGE (this node) ..."
    docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || run docker pull "$BASE_IMAGE"
    run docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" "$SCRIPT_DIR"
    if [ "${BUILD_WORKER:-0}" = 1 ]; then
        sync_kit
        log "building $IMAGE on the worker ..."
        worker_ssh "cd '$WORKER_KIT_DIR' && docker image inspect '$BASE_IMAGE' >/dev/null 2>&1 || docker pull '$BASE_IMAGE'; docker build --build-arg BASE_IMAGE='$BASE_IMAGE' -t '$IMAGE' ."
    fi
}
ship() {
    local hd wd
    hd=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null) || die "image $IMAGE missing here"
    wd=$(worker_ssh "docker image inspect -f '{{.Id}}' '$IMAGE' 2>/dev/null" || true)
    if [ "$hd" = "$wd" ]; then log "worker already has $IMAGE ($hd)"; return 0; fi
    log "shipping $IMAGE to the worker over ssh (docker save | docker load; ~26 GB) ..."
    [ -n "$DRY_RUN" ] && { echo "  DRY: docker save $IMAGE | ssh $WORKER_SSH docker load"; return 0; }
    docker save "$IMAGE" | worker_ssh "docker load" | tail -1
}
sync_kit() {
    log "syncing kit dir -> $WORKER_SSH:$WORKER_KIT_DIR"
    worker_ssh "mkdir -p '$WORKER_KIT_DIR'"
    run rsync -a --delete --exclude logs --exclude .env -e "ssh $SSH_MUX" "$SCRIPT_DIR/" "$WORKER_SSH:$WORKER_KIT_DIR/"
}
sync_weights() {
    local src="$HF_CACHE_DIR/hub/$MODEL_SUBDIR" dst="$WORKER_CACHE_DIR/hub/$MODEL_SUBDIR"
    [ -d "$src" ] || die "no local snapshot at $src"
    check_model_here "$MODEL_HOST_DIR" head
    log "syncing weights $src -> $WORKER_SSH:$dst (rsync, symlinks kept, low I/O priority) ..."
    worker_ssh "mkdir -p '$dst'"
    run ionice -c2 -n7 nice -n 19 rsync -aH --info=progress2 ${SYNC_BWLIMIT:+--bwlimit=$SYNC_BWLIMIT} \
        --exclude '*.incomplete' -e "ssh $SSH_MUX" "$src/" "$WORKER_SSH:$dst/"
}

# ---------------------------- inner scripts --------------------------------
write_inner_scripts() {
    local role
    for role in head worker; do
        local f="$HEAD_SCRIPT"; [ "$role" = worker ] && f="$WORKER_SCRIPT"
        cat > "$f" <<'EOS'
#!/bin/bash
set -euo pipefail
say() { echo "[mimo26-${ROLE}] $*"; }
[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
# shellcheck disable=SC2206
NAMES=(${SERVED_MODEL_NAME} ${SERVED_MODEL_ALIASES:-})
ARGS=(
    "${MODEL_DIR}"
    --served-model-name "${NAMES[@]}"
    --trust-remote-code
    --host 0.0.0.0 --port "${PORT}"
    --tensor-parallel-size "${TP}" --nnodes "${NNODES}" --node-rank "${RANK}"
    --master-addr "${HEAD_IP}" --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice
    --enable-prefix-caching
    --generation-config auto
    --override-generation-config "{\"repetition_penalty\": ${REP_PENALTY}, \"max_new_tokens\": ${DEFAULT_MAX_TOKENS:-131072}}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --moe-backend "${MOE_BACKEND}" --linear-backend "${LINEAR_BACKEND}"
    --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"pass_config\":{\"fuse_allreduce_rms\":false}}"
)
[ "${RANK}" != 0 ] && ARGS+=(--headless)
[ "${ENFORCE_EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
# shellcheck disable=SC2206
[ -n "${CAPTURE_SIZES:-}" ] && ARGS+=(--cudagraph-capture-sizes ${CAPTURE_SIZES})
# vLLM refuses to start unless device free memory >= total x gpu-memory-utilization, even with a pinned KV pool
# (v1/worker/utils.py request_memory; default 0.92 = 111.9 GiB, more than a Spark ever has free). Always pass both.
ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${LOAD_FORMAT:-}" ] && ARGS+=(--load-format "${LOAD_FORMAT}")
[ -n "${KV_CACHE_MEMORY:-}" ] && ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
[ "${ASYNC_SCHED:-0}" = 1 ] || ARGS+=(--no-async-scheduling)
case "${SPEC_METHOD:-dflash}" in
    dflash)
        spec="{\"method\":\"dflash\",\"model\":\"${MODEL_DIR}/dflash\",\"num_speculative_tokens\":${DFLASH_TOKENS:-7}"
        [ -n "${DFLASH_DRAFT_TP:-}" ] && spec="${spec},\"draft_tensor_parallel_size\":${DFLASH_DRAFT_TP}"
        # the drafter's standard attention backend has no NVFP4 path: give it its own (fp8) KV dtype
        if [ "${KV_CACHE_DTYPE}" = nvfp4 ] || [ -n "${DFLASH_KV_DTYPE:-}" ]; then spec="${spec},\"kv_cache_dtype\":\"${DFLASH_KV_DTYPE:-fp8}\""; fi
        # batch-size schedule [[lo,hi,k],...]: deep spec helps at low concurrency, at high concurrency every extra verify
        # token drags more distinct experts through memory (48 tokens x top-8 touch ~78% of the experts)
        [ -n "${DFLASH_DYNAMIC:-}" ] && spec="${spec},\"num_speculative_tokens_per_batch_size\":${DFLASH_DYNAMIC}"
        ARGS+=(--speculative-config "${spec}}") ;;
    mtp)  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-3}}") ;;
    none) ;;
    *) say "FATAL: SPEC_METHOD=${SPEC_METHOD} (dflash|mtp|none)"; exit 1 ;;
esac
[ -n "${THINKING:-}" ] && ARGS+=(--default-chat-template-kwargs "{\"enable_thinking\": ${THINKING}}")
if [ "${LANGUAGE_MODEL_ONLY:-0}" = 1 ]; then
    ARGS+=(--language-model-only)
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ -n "${MM_PROC_KWARGS:-}" ] && ARGS+=(--mm-processor-kwargs "${MM_PROC_KWARGS}")
    [ "${SKIP_MM_PROFILING:-1}" = 1 ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS}); ARGS+=("${EXTRA[@]}")
fi
# kit patchers (overlay/patch_*.py): applied to the vLLM tree at container start, like the GLM kit
shopt -s nullglob
for p in /opt/mimo26/overlay/patch_*.py; do say "patch: $(basename "$p")"; python3 "$p"; done
shopt -u nullglob
# chat template with a thinking prefill that does not misdescribe the request (overlay/patch_chat_template.py)
[ -f /tmp/mimo26_chat_template.jinja ] && ARGS+=(--chat-template /tmp/mimo26_chat_template.jinja)
say "launching: vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
EOS
        chmod +x "$f"
    done
}

# ------------------------------- launch ------------------------------------
SHM_CLEANUP_HOST="$SCRIPT_DIR/overlay/shm_cleanup.py"
cleanup_orphan_shm() {
    # vLLM's /dev/shm/psm_* segments outlive a killed container (--ipc=host); remove the ones nobody maps
    local where="$1" out run="docker run --rm --pid=host --ipc=host --cap-add SYS_PTRACE -v /dev/shm:/dev/shm"
    [ -f "$SHM_CLEANUP_HOST" ] || return 0
    if [ "$where" = worker ]; then
        scp -q $SSH_MUX -o BatchMode=yes "$SHM_CLEANUP_HOST" "${WORKER_SSH}:/tmp/mimo26_shm_cleanup.py" 2>/dev/null || return 0
        out=$(worker_ssh "$run -v /tmp/mimo26_shm_cleanup.py:/opt/shm_cleanup.py:ro --entrypoint python3 '$IMAGE' /opt/shm_cleanup.py" 2>/dev/null | grep '^removed' || true)
    else
        out=$($run -v "$SHM_CLEANUP_HOST:/opt/shm_cleanup.py:ro" --entrypoint python3 "$IMAGE" /opt/shm_cleanup.py 2>/dev/null | grep '^removed' || true)
    fi
    [ -n "$out" ] && log "shm cleanup ($where): $out"; return 0
}
retune_zram() {
    [ -n "${MIMO26_ZRAM_ALGO:-}" ] && [ -x /usr/local/sbin/glm53-zram ] || return 0
    local p1 p2
    ( sudo -n /usr/local/sbin/glm53-zram "$MIMO26_ZRAM_ALGO" "${MIMO26_ZRAM_GIB:-10}" 2>&1 | sed 's/^/    head:   /' ) & p1=$!
    ( worker_ssh "sudo -n /usr/local/sbin/glm53-zram '$MIMO26_ZRAM_ALGO' '${MIMO26_ZRAM_GIB:-10}'" 2>&1 | sed 's/^/    worker: /' ) & p2=$!
    wait "$p1" "$p2"   # never a bare wait: the memwatch daemon is a job of this shell too
}
container_env() {
    # -e list shared by both ranks (printed as one string; values are trusted .env content)
    local v out=""
    local -a common=(
        NCCL_IB_DISABLE=0 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_NVLS_ENABLE=0
        NCCL_CUMEM_ENABLE=0 NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=0 NCCL_IGNORE_CPU_AFFINITY=1
        "NCCL_DEBUG=$NCCL_DEBUG" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        ${NCCL_PROTO:+NCCL_PROTO=$NCCL_PROTO} ${NCCL_ALGO:+NCCL_ALGO=$NCCL_ALGO}
        ${VLLM_KV_CACHE_LAYOUT:+VLLM_KV_CACHE_LAYOUT=$VLLM_KV_CACHE_LAYOUT}
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/root/.cache/huggingface
        VLLM_CACHE_ROOT=/root/.cache/vllm TRITON_CACHE_DIR=/root/.cache/vllm/triton
        TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor
        "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT" "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        FLASHINFER_DISABLE_VERSION_CHECK=1
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        VLLM_USE_DEEP_GEMM=0
        VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    )
    [ "${MARLIN_A8:-1}" = 1 ] && common+=(VLLM_MARLIN_INPUT_DTYPE=fp8)
    [ "${OPROJ_FP8:-0}" = 1 ] && common+=(VLLM_MIMO_OPROJ_FP8=1)
    for v in VLLM_DIFFKV_FULL_ATTN_SEGMENTS VLLM_DIFFKV_PREFILL_BLOCK_M VLLM_DIFFKV_PREFILL_NUM_WARPS VLLM_DIFFKV_SPEC_3D_MAX_Q \
             VLLM_DIFFKV_SPEC_3D_BLOCK_M VLLM_DIFFKV_CUSTOM_PREFILL_MIN_Q VLLM_CA_MAX_SIZE_MB VLLM_ATTENTION_BACKEND \
             MIMO26_IT_LOCAL_READS MIMO26_NAN_PROBE MIMO26_PARAM_SUMS MIMO26_PAGE_UNIFY MIMO26_JSON_NOTHINK \
             MIMO26_NVFP4_DQ_MIN_Q MIMO26_NVFP4_DQ_MAX_MB MIMO26_VIZ MIMO26_VIZ_UDP MIMO26_VIZ_HZ MIMO26_VIZ_MAX_T MIMO26_VIZ_STRICT \
             MIMO26_THINK_PREFILL; do
        [ -n "${!v:-}" ] && common+=("$v=${!v}")
    done
    for v in SERVED_MODEL_NAME SERVED_MODEL_ALIASES PORT TP NNODES HEAD_IP MASTER_PORT MODEL_DIR \
             KV_CACHE_DTYPE KV_CACHE_MEMORY GPU_MEM_UTIL MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             SPEC_METHOD DFLASH_TOKENS DFLASH_DRAFT_TP DFLASH_DYNAMIC DFLASH_KV_DTYPE MTP_TOKENS MOE_BACKEND LINEAR_BACKEND CUDAGRAPH_MODE CAPTURE_SIZES \
             ASYNC_SCHED THINKING REP_PENALTY DEFAULT_MAX_TOKENS LANGUAGE_MODEL_ONLY LIMIT_MM MM_PROC_KWARGS SKIP_MM_PROFILING EXTRA_ARGS VLLM_API_KEY LOAD_FORMAT ENFORCE_EAGER; do
        common+=("$v=${!v:-}")
    done
    for v in "${common[@]}"; do out+=" -e $(printf '%q' "$v")"; done
    echo "$out"
}
launch_cluster() {
    if [ -z "$DRY_RUN" ]; then
        docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
        worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true
        local p1 p2; cleanup_orphan_shm head & p1=$!; cleanup_orphan_shm worker & p2=$!; wait "$p1" "$p2"
        retune_zram
        wait_for_headroom
    fi
    mkdir -p "$CACHE_ROOT"
    if [ -z "$DRY_RUN" ]; then
        worker_ssh "mkdir -p '$WORKER_VLLM_CACHE'"
        scp -q $SSH_MUX -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"
    fi
    local envs; envs="$(container_env)"
    # the kit's overlay/ (patch_*.py) is bind-mounted over the copy baked into the image, so patcher edits need no rebuild
    local common_run="--gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 -v $SCRIPT_DIR/overlay:/opt/mimo26/overlay:ro ${EXTRA_MOUNTS:-}"   # EXTRA_MOUNTS: extra -v args (same host paths on both nodes), e.g. stock files over the patched ones for bisection

    log "starting worker on ${WORKER_SSH} (rank 1; NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    local wcmd="docker run -d --name '$CONTAINER_WORKER' $common_run \
        -v '$WORKER_CACHE_DIR:/root/.cache/huggingface' -v '$WORKER_VLLM_CACHE:/root/.cache/vllm' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' $envs \
        -e ROLE=worker -e RANK=1 -e NCCL_SOCKET_IFNAME='$WORKER_CX7_IF' -e GLOO_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e NCCL_IB_HCA='$WORKER_CX7_IB' -e NCCL_IB_GID_INDEX='$NCCL_IB_GID_INDEX' -e VLLM_HOST_IP='$WORKER_IP' \
        --entrypoint bash '$IMAGE' /start.sh"
    if [ -n "$DRY_RUN" ]; then echo "  DRY (worker via ssh): $wcmd"; else worker_ssh "$wcmd" >/dev/null; fi

    log "starting head (rank 0; API :${PORT}; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    local hcmd="docker run -d --name '$CONTAINER_HEAD' $common_run \
        -v '$HF_CACHE_DIR:/root/.cache/huggingface' -v '$CACHE_ROOT:/root/.cache/vllm' \
        -v '$HEAD_SCRIPT:/start.sh:ro' $envs \
        -e ROLE=head -e RANK=0 -e NCCL_SOCKET_IFNAME='$HEAD_CX7_IF' -e GLOO_SOCKET_IFNAME='$HEAD_CX7_IF' \
        -e NCCL_IB_HCA='$HEAD_CX7_IB' -e NCCL_IB_GID_INDEX='$NCCL_IB_GID_INDEX' -e VLLM_HOST_IP='$HEAD_IP' \
        --entrypoint bash '$IMAGE' /start.sh"
    if [ -n "$DRY_RUN" ]; then echo "  DRY (head): $hcmd"; else eval "$hcmd" >/dev/null; fi
    log "containers up — head=${CONTAINER_HEAD}, worker=${CONTAINER_WORKER}"
    arm_memwatch
}
# Host-side low-memory watchdog per node (scripts/memwatch.sh): kills the local container before the node livelocks.
arm_memwatch() {
    local floor="${MIMO26_WATCHDOG_MIB:-1000}"
    [ "$floor" != 0 ] || return 0
    [ -n "$DRY_RUN" ] && { echo "  DRY: memwatch floor ${floor} MiB on both nodes"; return 0; }
    pkill -f "[m]emwatch.sh $CONTAINER_HEAD " 2>/dev/null || true
    setsid nohup "$SCRIPT_DIR/scripts/memwatch.sh" "$CONTAINER_HEAD" "$floor" "$LOGDIR/memwatch-head.log" >/dev/null 2>&1 < /dev/null &
    scp -q $SSH_MUX -o BatchMode=yes "$SCRIPT_DIR/scripts/memwatch.sh" "${WORKER_SSH}:/tmp/mimo26-memwatch.sh"
    worker_ssh "pkill -f '[m]imo26-memwatch.sh $CONTAINER_WORKER ' 2>/dev/null; setsid nohup bash /tmp/mimo26-memwatch.sh '$CONTAINER_WORKER' '$floor' /tmp/mimo26-memwatch-worker.log >/dev/null 2>&1 < /dev/null &" || warn "could not arm the worker memwatch"
    log "memwatch armed: kill under ${floor} MiB MemAvailable (logs: $LOGDIR/memwatch-head.log, worker /tmp/mimo26-memwatch-worker.log)"
}

# ---------------------------- boot memory rails ----------------------------
reclaim_both() {
    local n="$1" why="$2"
    [ "$n" != 0 ] && [ -x /usr/local/sbin/glm53-reclaim ] || return 0
    log "$why: pushing ${n} GiB of cold pages to zram on both nodes ..."
    local p1 p2
    ( sudo -n /usr/local/sbin/glm53-reclaim "$CONTAINER_HEAD" "$n" 2>&1 | sed 's/^/    head:   /' || warn "head reclaim failed" ) & p1=$!
    ( worker_ssh "sudo -n /usr/local/sbin/glm53-reclaim '$CONTAINER_WORKER' '$n'" 2>&1 | sed 's/^/    worker: /' || warn "worker reclaim failed" ) & p2=$!
    wait "$p1" "$p2"   # a bare wait also waits on the memwatch daemon (a job of this shell) and never returns
}
postload_reclaim_watch() {
    local i
    for i in $(seq 1 1200); do
        docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true || return 0
        if docker logs "$CONTAINER_HEAD" 2>&1 | grep -aq "Loading weights took\|Model loading took"; then
            boot_phase containers "weights loaded; KV allocation and graph capture"
            reclaim_both "${MIMO26_POSTLOAD_RECLAIM:-0}" "post-load reclaim"; break
        fi
        sleep 2
    done
    for i in $(seq 1 1200); do
        docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true || return 0
        if docker logs "$CONTAINER_HEAD" 2>&1 | grep -aq "Graph capturing finished\|Capturing CUDA graphs.*finished\|init engine.*took"; then
            reclaim_both "${MIMO26_PREAPI_RECLAIM:-0}" "pre-API reclaim"; return 0
        fi
        sleep 1
    done
}
boot_headroom_guard() {
    local thr="${MIMO26_BOOT_GUARD_MIB:-4500}" gib="${MIMO26_BOOT_GUARD_GIB:-3}" last_h=0 last_w=0 now h w
    [ "$thr" != 0 ] && [ -x /usr/local/sbin/glm53-reclaim ] || return 0
    while true; do
        now=$(date +%s)
        h=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
        w=$(worker_ssh "awk '/MemAvailable/{printf \"%d\", \$2/1024}' /proc/meminfo" 2>/dev/null || echo 999999)
        if [ "${h:-999999}" -lt "$thr" ] && [ $((now - last_h)) -gt 6 ]; then
            log "boot guard: head MemAvailable ${h} MiB < ${thr} — reclaiming ${gib} GiB"
            sudo -n /usr/local/sbin/glm53-reclaim "$CONTAINER_HEAD" "$gib" 2>&1 | sed 's/^/    head:   /' || true; last_h=$now
        fi
        if [ "${w:-999999}" -lt "$thr" ] && [ $((now - last_w)) -gt 6 ]; then
            log "boot guard: worker MemAvailable ${w} MiB < ${thr} — reclaiming ${gib} GiB"
            worker_ssh "sudo -n /usr/local/sbin/glm53-reclaim '$CONTAINER_WORKER' '$gib'" 2>&1 | sed 's/^/    worker: /' || true; last_w=$now
        fi
        sleep 2
    done
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health" logpid="" elapsed=0 healthy=0 exited=0 dead_side="" worker_fail=0
    log "waiting for ${url} (timeout ${READY_TIMEOUT}s); streaming head logs — Ctrl-C detaches, the server keeps running"
    _stop_logtail() { [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true; wait "$logpid" 2>/dev/null || true; logpid=""; }
    trap '_stop_logtail; warn "interrupted — containers keep running (./start.sh logs | stop)"; exit 130' INT
    docker logs -f --tail 200 "$CONTAINER_HEAD" 2>&1 & logpid=$!   # from the start: the patcher lines land in the boot log
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then exited=1; dead_side=head; break; fi
        if worker_ssh "docker inspect -f '{{.State.Running}}' '$CONTAINER_WORKER' 2>/dev/null" | grep -q true; then worker_fail=0
        else worker_fail=$((worker_fail + 1)); [ "$worker_fail" -ge 3 ] && { exited=1; dead_side=worker; break; }; fi
        sleep 10; elapsed=$((elapsed + 10))
    done
    _stop_logtail
    trap 'warn "interrupted — containers keep running"; exit 130' INT
    if [ "$healthy" = 1 ]; then log "health check passed after ${elapsed}s"
    elif [ "$exited" = 1 ]; then warn "${dead_side} container exited after ${elapsed}s"
    else warn "timed out after ${elapsed}s"; fi
    [ "$healthy" = 1 ]
}
collect_failure_logs() {
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    worker_ssh "docker logs '$CONTAINER_WORKER' 2>&1" >"$LOGDIR/worker.log" 2>&1 || true
}
on_ready() {
    local spec="DFlash k=${DFLASH_TOKENS}"; [ "$SPEC_METHOD" = mtp ] && spec="MTP k=${MTP_TOKENS}"; [ "$SPEC_METHOD" = none ] && spec=off
    local kv="${KV_CACHE_MEMORY:+pinned $KV_CACHE_MEMORY B/rank}"; kv="${kv:-gpu-util $GPU_MEM_UTIL}"
    local pool; pool=$(docker logs "$CONTAINER_HEAD" 2>&1 | grep -a "GPU KV cache size" | tail -1 | sed 's/.*GPU KV cache size: //' || true)
    log "======================================================================"
    log "MiMo-V2.6-Flash is UP (TP=${TP}, nnodes=${NNODES}, image ${IMAGE})"
    log "  endpoint   : http://${HEAD_IP}:${PORT}/v1   model id: ${SERVED_MODEL_NAME}${SERVED_MODEL_ALIASES:+ (+ $SERVED_MODEL_ALIASES)}"
    log "  weights    : ${MODEL}@${MODEL_REVISION:0:8}  kv=${KV_CACHE_DTYPE} (${kv}) pool=${pool:-?}"
    log "  features   : spec=${spec}, moe=${MOE_BACKEND}${MARLIN_A8:+ a8}, graphs=${CUDAGRAPH_MODE}, mm=${LIMIT_MM}, thinking=${THINKING:-template default}"
    log "  manage     : ./start.sh status | logs | logs worker | stop"
    log "======================================================================"
}

# ------------------------------- commands ----------------------------------
start() {
    preflight
    write_inner_scripts
    log "config: image=${IMAGE} spec=${SPEC_METHOD} kv=${KV_CACHE_DTYPE}/${KV_CACHE_MEMORY:-util $GPU_MEM_UTIL} max-len=${MAX_MODEL_LEN} seqs=${MAX_NUM_SEQS} batched=${MAX_NUM_BATCHED_TOKENS} moe=${MOE_BACKEND} graphs=${CUDAGRAPH_MODE} port=${PORT}"
    if [ -n "$DRY_RUN" ]; then launch_cluster; log "dry run: nothing started"; return 0; fi
    boot_phase launching "starting both containers"
    launch_cluster
    boot_phase containers "containers started; engine init and weight load"
    postload_reclaim_watch &
    boot_headroom_guard & local guard_pid=$!
    if wait_for_health; then
        kill "$guard_pid" 2>/dev/null || true
        boot_phase healthy "API healthy; post-ready reclaim"
        reclaim_both "${MIMO26_POSTREADY_RECLAIM:-0}" "post-ready reclaim"
        on_ready
        boot_phase ready serving
        return 0
    fi
    kill "$guard_pid" 2>/dev/null || true
    boot_phase failed "server did not become healthy"
    collect_failure_logs
    echo "---- last 60 lines of head log ----"; tail -n 60 "$LOGDIR/head.log" || true
    echo "---- last 40 lines of worker log ----"; tail -n 40 "$LOGDIR/worker.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}
stop() {
    boot_phase teardown "stopping containers"
    log "stopping head container ..."; docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (none running)"
    log "stopping worker container on ${WORKER_SSH} ..."; worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || log "  (none running)"
    pkill -f "[m]emwatch.sh $CONTAINER_HEAD " 2>/dev/null || true
    worker_ssh "pkill -f '[m]imo26-memwatch.sh $CONTAINER_WORKER ' 2>/dev/null" || true
    log "stopped."
}
status() {
    log "head (${CONTAINER_HEAD} on $(hostname)):"; docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then log "  API: healthy — http://127.0.0.1:${PORT}/v1"; else log "  API: not responding"; fi
    log "worker (${CONTAINER_WORKER} on ${WORKER_SSH}):"; worker_ssh "docker ps -a --filter name=${CONTAINER_WORKER} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null || log "  (worker unreachable)"
}
logs() {
    case "${1:-head}" in
        worker) trap '' INT; worker_ssh "docker logs -f --tail 100 '$CONTAINER_WORKER'" || true ;;
        *)      trap '' INT; docker logs -f --tail 100 "$CONTAINER_HEAD" || true ;;
    esac
}
usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
main() {
    case "${1:-start}" in
        start)   start ;;
        restart) stop; start ;;
        stop)    stop ;;
        status)  status ;;
        logs)    shift || true; logs "$@" ;;
        build)   build ;;
        ship)    ship ;;
        sync)    sync_kit; sync_weights ;;
        check)   DRY_RUN=1; [ "${SKIP_PREFLIGHT:-0}" = 1 ] || preflight; write_inner_scripts; launch_cluster; echo; echo "---- head inner script argv ----"; sed -n '/^ARGS=(/,/^)/p' "$HEAD_SCRIPT" | head -40 ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}
main "$@"
