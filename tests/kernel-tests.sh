#!/usr/bin/env bash
# kernel-tests.sh: run the patched DiffKV attention tests (diffbot's, vs fp32 references) and the custom prefill
# kernel test inside a throwaway GPU container of the kit image. GPU must be free (no serving container running):
# the tests allocate a few hundred MB and take about a minute. Usage: tests/kernel-tests.sh [image]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-$(grep -E '^IMAGE=' "$R/.env" | cut -d= -f2)}"
if docker ps --format '{{.Names}}' | grep -qE 'glm53-exl3|mimo26-'; then echo "a serving container is running on this node — refusing"; exit 2; fi
docker run --rm --gpus all --entrypoint bash -e TORCH_CUDA_ARCH_LIST=12.1a -e EXT_BUILD=/opt/mimo26/attn-build \
    -v "$R/patches:/tests:ro" -v "$R/kernels/attention:/kernels:ro" "$IMAGE" -c '
set -e; cd /tests
for t in test_diffkv_fp8.py test_diffkv_prefill.py test_diffkv_spec3d.py; do echo "== $t"; python3 $t 2>&1 | grep -v "^INFO\|^W0" | tail -12; done
echo "== prefill_attn.cu (custom kernel, prebuilt sm_121a)"; python3 /kernels/test_prefill_attn.py 2>&1 | grep -v "^INFO\|^W0" | tail -12'
