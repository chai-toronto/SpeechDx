#!/usr/bin/env bash
# Watch a Slurm job and emit one line per state change.
#
# Usage:
#   scripts/slurm_watch.sh <jobid> [poll_interval_s]
#
# Emits a line on:
#   - first observation
#   - any change in State, Reason, or NodeList
#   - terminal state (queried from sacct once the job leaves the queue)
#
# Designed to be tail-friendly / Monitor-friendly: each line is a discrete
# event you can react to.
set -euo pipefail

JOB="${1:?usage: $0 <jobid> [poll_interval_s]}"
INTERVAL="${2:-15}"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

prev=""
while true; do
    # %T=state, %r=reason, %M=elapsed, %N=node, %L=time-left
    line=$(squeue -h -j "$JOB" -o "%T|%r|%M|%N|%L" 2>/dev/null || true)
    if [[ -z "$line" ]]; then
        # Job left the queue — fetch terminal state once.
        final=$(sacct -j "$JOB" -X -P -n -o "State,ExitCode,Elapsed,MaxRSS" 2>/dev/null \
                | head -1)
        if [[ -n "$final" ]]; then
            echo "[$(ts)] job $JOB TERMINAL: $final"
        else
            echo "[$(ts)] job $JOB gone (no sacct record yet)"
        fi
        exit 0
    fi
    IFS='|' read -r st reason elapsed node left <<<"$line"
    # Only state/reason/node count as a "change" — elapsed/left tick every poll.
    key="$st|$reason|$node"
    if [[ "$key" != "$prev" ]]; then
        echo "[$(ts)] job $JOB state=$st reason=$reason elapsed=$elapsed left=$left node=${node:--}"
        prev="$key"
    fi
    sleep "$INTERVAL"
done
