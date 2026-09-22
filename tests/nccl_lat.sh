#!/usr/bin/env bash
# nccl_lat.sh [NCCL_PROTO] [NCCL_ALGO]: run tests/nccl_lat.py on both nodes with the kit's fabric settings (+ optional protocol/algo override).
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; set -a; . "$R/.env"; set +a
PROTO="${1:-}"; ALGO="${2:-}"; IMAGE="${IMAGE:?}"; PORT_=29778
ex=""; [ -n "$PROTO" ] && ex="$ex -e NCCL_PROTO=$PROTO"; [ -n "$ALGO" ] && ex="$ex -e NCCL_ALGO=$ALGO"
common="--rm --gpus all --network host --ipc=host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 -v $R/tests:/tests:ro -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e MASTER_ADDR=$HEAD_IP -e MASTER_PORT=$PORT_ $ex --entrypoint python3"
ssh -o BatchMode=yes "$WORKER_SSH" "docker run $common -e RANK=1 -e NCCL_SOCKET_IFNAME=$WORKER_CX7_IF -e GLOO_SOCKET_IFNAME=$WORKER_CX7_IF -e NCCL_IB_HCA=$WORKER_CX7_IB $IMAGE /tests/nccl_lat.py" 2>&1 | grep -v "^INFO\|^W0\|^WARNING" &
docker run $common -e RANK=0 -e NCCL_SOCKET_IFNAME=$HEAD_CX7_IF -e GLOO_SOCKET_IFNAME=$HEAD_CX7_IF -e NCCL_IB_HCA=$HEAD_CX7_IB $IMAGE /tests/nccl_lat.py 2>&1 | grep -v "^INFO\|^W0\|^WARNING"
wait
