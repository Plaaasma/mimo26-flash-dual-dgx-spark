#!/usr/bin/env bash
# gemm_check.sh [image]: run tests/gemm_check.py in a throwaway GPU container (small: ~1 GB, fine next to a serving container).
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; IMAGE="${1:-$(grep -E '^IMAGE=' "$R/.env" | cut -d= -f2)}"
docker run --rm --gpus all --entrypoint python3 -v "$R/tests:/tests:ro" "$IMAGE" /tests/gemm_check.py 2>&1 | grep -v "^INFO\|^W0\|^WARNING"
