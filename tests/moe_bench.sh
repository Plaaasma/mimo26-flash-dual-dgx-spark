#!/usr/bin/env bash
# moe_bench.sh [image] [extra args...]: diffbot's single-layer MoE kernel bench (MiMo TP2 shapes) in a throwaway GPU
# container. Defaults: layers=2 (3.2 GB of random experts, fits next to a running server), Ms=8,16,48,64,512,
# backends=marlin,marlin_a8. Pass a different image to compare kernel builds (e.g. tonyd's native sm_121 image).
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; IMAGE="${1:-$(grep -E '^IMAGE=' "$R/.env" | cut -d= -f2)}"; shift 2>/dev/null
docker run --rm --gpus all --entrypoint python3 -v "$R/tests:/tests:ro" "$IMAGE" /tests/moe_bench.py layers=2 Ms=8,16,48,64,512 backends=marlin,marlin_a8 "$@" 2>&1 | grep -v "^INFO\|^W0\|^WARNING"
