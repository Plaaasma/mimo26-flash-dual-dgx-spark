#!/usr/bin/env bash
# boot-bg.sh [start|restart]: run start.sh from the kit directory regardless of the caller's cwd (for ssh/nohup use).
cd "$(dirname "$(readlink -f "$0")")/.." && exec ./start.sh "${1:-start}"
