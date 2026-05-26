#!/bin/bash
# run_with_data.sh -- run a command with AHB's data/ mounted from data.sqfs.
#
# data/ was converted from ~610k loose audio files to a single SquashFS image
# (data.sqfs) to cut /scratch inode usage. The manifests in metadata/ reference
# audio by ABSOLUTE path under .../Audio-Health-Benchmark/data/, so the image
# must appear at exactly that path.
#
# This cluster has no fusermount3 (the fuse3 package is not installed), so
# `squashfuse` cannot mount in the normal way. Inside an unprivileged user
# namespace the caller is root and libfuse performs the mount directly with no
# setuid helper -- so we enter `unshare -rm`, mount data.sqfs onto data/, and
# exec the command there. The mount exists only for that process tree and is
# torn down automatically when it exits.
#
# Usage:   ./run_with_data.sh <command> [args...]
# Example: ./run_with_data.sh python -u scripts/test_qwen3omni_all.py --all ...
set -euo pipefail

REPO=/scratch/kieu/Audio-Health-Benchmark
SQFS="$REPO/data.sqfs"
DEST="$REPO/data"

[ "$#" -ge 1 ] || { echo "run_with_data.sh: no command given" >&2; exit 1; }
[ -f "$SQFS" ] || { echo "run_with_data.sh: missing image $SQFS" >&2; exit 1; }
[ -d "$DEST" ] || { echo "run_with_data.sh: missing mountpoint $DEST" >&2; exit 1; }
SQFUSE=$(command -v squashfuse) || { echo "run_with_data.sh: squashfuse not on PATH" >&2; exit 1; }

exec unshare -rm bash -c '
  set -euo pipefail
  repo="$1"; sqfs="$2"; dest="$3"; sqfuse="$4"; shift 4
  # Cache-warm: stage the image onto node-local disk when running under Slurm.
  if [ -n "${SLURM_TMPDIR:-}" ] && cp "$sqfs" "$SLURM_TMPDIR/data.sqfs" 2>/dev/null; then
    sqfs="$SLURM_TMPDIR/data.sqfs"
    echo "[run_with_data] image staged to node-local scratch"
  fi
  "$sqfuse" "$sqfs" "$dest"
  echo "[run_with_data] mounted $dest -- $(ls "$dest" | wc -l) entries"
  cd "$repo"
  exec "$@"
' _ "$REPO" "$SQFS" "$DEST" "$SQFUSE" "$@"
