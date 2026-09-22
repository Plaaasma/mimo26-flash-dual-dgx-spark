#!/usr/bin/env bash
# memwatch.sh <container> <min_avail_mib> [log]: host-side low-memory watchdog for one serving container.
# Spark unified memory has no OOM killer: an overshoot livelocks the node until a power cycle. This loop samples
# MemAvailable every second and `docker kill`s the container when two consecutive samples are under the floor
# (after one reclaim attempt through the root helper). Exits by itself when the container is gone.
set -u
c="${1:?container}"; floor="${2:?min MiB}"; log="${3:-/var/tmp/mimo26-memwatch.log}"
low=0; t_reclaim=0
say() { printf '%s memwatch %s: %s\n' "$(date +%FT%T)" "$c" "$*" >> "$log"; }
say "armed floor=${floor} MiB pid=$$"
while :; do
    state=$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null) || { say "container gone, exiting"; exit 0; }
    [ "$state" = true ] || { say "container not running, exiting"; exit 0; }
    a=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    if [ "$a" -lt "$floor" ]; then
        low=$((low + 1))
        now=$(date +%s)
        if [ $((now - t_reclaim)) -gt 5 ] && [ -x /usr/local/sbin/glm53-reclaim ]; then
            sudo -n /usr/local/sbin/glm53-reclaim "$c" 2 >> "$log" 2>&1 || true; t_reclaim=$now
            a=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
        fi
        if [ "$a" -lt "$floor" ] && [ "$low" -ge 2 ]; then
            say "MemAvailable ${a} MiB < ${floor} for ${low} samples — killing container (swap used $(free -m | awk '/Swap/{print $3}') MiB)"
            docker kill "$c" >> "$log" 2>&1 || true
            say "killed"; exit 3
        fi
    else
        low=0
    fi
    sleep 1
done
