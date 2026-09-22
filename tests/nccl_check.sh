#!/usr/bin/env bash
# nccl_check.sh: run tests/nccl_check.py on both nodes with the kit's NCCL settings (from .env). ~1 GB per node:
# safe next to a running server. Prints per-rank OK/BAD lines and "RESULT: ALL OK".
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; set -a; . "$R/.env"; set +a
IMAGE="${IMAGE:?}"; PORT_=29777
common="--rm --gpus all --network host --ipc=host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 -v $R/tests:/tests:ro -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e MASTER_ADDR=$HEAD_IP -e MASTER_PORT=$PORT_ --entrypoint python3"
ssh -o BatchMode=yes "$WORKER_SSH" "docker run $common -e RANK=1 -e NCCL_SOCKET_IFNAME=$WORKER_CX7_IF -e GLOO_SOCKET_IFNAME=$WORKER_CX7_IF -e NCCL_IB_HCA=$WORKER_CX7_IB $IMAGE /tests/nccl_check.py" 2>&1 | grep -v "^INFO\|^W0\|^WARNING" &
docker run $common -e RANK=0 -e NCCL_SOCKET_IFNAME=$HEAD_CX7_IF -e GLOO_SOCKET_IFNAME=$HEAD_CX7_IF -e NCCL_IB_HCA=$HEAD_CX7_IB $IMAGE /tests/nccl_check.py 2>&1 | grep -v "^INFO\|^W0\|^WARNING"
wait
