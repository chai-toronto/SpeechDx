#!/bin/bash
# run_with_data_local.sh -- like run_with_data.sh but MIG-safe.
#
# run_with_data.sh enters `unshare -rm` so libfuse can mount data.sqfs as
# the user. That trick works fine on a full H100 but the device-cgroup
# permissions for a MIG compute instance don't propagate through the user
# namespace — CUDA fails immediately with cudaErrorDevicesUnavailable
# inside the namespace. See memory/mig_unshare_incompat.md.
#
# This script avoids the namespace entirely:
#   1. Extract data.sqfs to $SLURM_TMPDIR/data (node-local NVMe, ~10 min
#      for the current 56 GB image with ~600k files).
#   2. Swap the empty mountpoint repo/data for a symlink → $SLURM_TMPDIR/data.
#   3. Run the command. The absolute path /scratch/.../data/foo.wav
#      resolves through the symlink — no namespace, MIG-safe.
#   4. On shell exit, restore repo/data as an empty directory.
#
# Concurrency: repo/data is shared filesystem state. Two jobs running this
# script simultaneously would clobber each other's symlink. We refuse to
# start if repo/data isn't a plain empty directory at entry; the trap on
# exit restores it. If a previous run crashed before the trap could fire
# (SIGKILL), repair manually: `rm -f data && mkdir -p data`.
#
# Usage:   ./run_with_data_local.sh <command> [args...]
# Example: ./run_with_data_local.sh python -u -m sdx single warm -e wavlm
set -euo pipefail

REPO=/scratch/kieu/Audio-Health-Benchmark
SQFS="$REPO/data.sqfs"
DEST="$REPO/data"

[ "$#" -ge 1 ] || { echo "run_with_data_local.sh: no command given" >&2; exit 1; }
[ -f "$SQFS" ] || { echo "run_with_data_local.sh: missing image $SQFS" >&2; exit 1; }
[ -n "${SLURM_TMPDIR:-}" ] || {
    echo "run_with_data_local.sh: \$SLURM_TMPDIR not set — needs node-local scratch." >&2
    exit 1
}
command -v unsquashfs >/dev/null || {
    echo "run_with_data_local.sh: unsquashfs not on PATH (module load StdEnv/2023?)." >&2
    exit 1
}

# Refuse if the mountpoint is already a symlink with a still-resolvable
# target (another instance might be active). If the symlink is *broken*
# (target unreachable — typical aftermath of a previous job that was
# SIGKILL'd before its trap could fire), auto-repair it: a stale broken
# symlink can't be backing any live process. Concurrency safety is
# unchanged — a live job's target is always present on its compute node.
if [ -L "$DEST" ]; then
    if [ -e "$DEST" ]; then
        echo "run_with_data_local.sh: $DEST is a live symlink (target exists)" \
             "— another instance may be active. Refusing." >&2
        exit 1
    fi
    echo "run_with_data_local.sh: $DEST is a stale broken symlink ($(readlink "$DEST"))" \
         "— auto-repairing." >&2
    rm -f "$DEST"
    mkdir -p "$DEST"
fi
[ -d "$DEST" ] || { echo "run_with_data_local.sh: missing mountpoint $DEST" >&2; exit 1; }
[ -z "$(ls -A "$DEST" 2>/dev/null)" ] || {
    echo "run_with_data_local.sh: $DEST is not empty (looks already populated)." >&2
    exit 1
}

LOCAL_DATA="$SLURM_TMPDIR/data"

# Idempotent within one allocation — a retry inside the same job reuses
# the already-extracted tree (Slurm clears $SLURM_TMPDIR between jobs, so
# this only short-circuits within a single allocation's lifetime).
if [ -d "$LOCAL_DATA" ] && [ -n "$(ls -A "$LOCAL_DATA" 2>/dev/null)" ]; then
    echo "[run_with_data_local] $LOCAL_DATA already populated, reusing"
else
    echo "[run_with_data_local] extracting $(stat -c %s "$SQFS" | numfmt --to=iec) → $LOCAL_DATA ..."
    t0=$(date +%s)
    NPROC="${SLURM_CPUS_PER_TASK:-$(nproc)}"
    # -q quiets per-file output; -no-progress drops the progress bar (we
    # buffer through Slurm's stdout); -processors parallelizes the
    # decompress across cores allocated to the job.
    unsquashfs -q -no-progress -processors "$NPROC" -d "$LOCAL_DATA" "$SQFS"
    echo "[run_with_data_local] extract took $(( $(date +%s) - t0 ))s" \
         "($(find "$LOCAL_DATA" -mindepth 1 -maxdepth 1 -type d | wc -l) top-level entries)"
fi

# Restore the empty mountpoint on any exit (clean, error, signal).
restore() {
    if [ -L "$DEST" ]; then
        rm -f "$DEST"
        mkdir -p "$DEST"
        echo "[run_with_data_local] restored $DEST as empty dir"
    fi
}
trap restore EXIT INT TERM

rmdir "$DEST"
ln -sfn "$LOCAL_DATA" "$DEST"
echo "[run_with_data_local] $DEST -> $LOCAL_DATA"

cd "$REPO"
"$@"
