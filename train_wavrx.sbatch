#!/bin/bash
#SBATCH --job-name=train_wavrx
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_wavrx_%j.out
#SBATCH --error=logs/train_wavrx_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Train the WavRx probe on a multi_L24 WavLM-large cache. Reader-only
# (no encoder load), so the smallest MIG slice (1g.10gb) is plenty —
# probe forward + Adam on a per-batch (24, T, 1024) tensor fits easily.
#
# Required env: WAVRX_DATASET=<dataset name>
# Optional   : WAVRX_TASKS='T13 T14 T15 T16'  (default: all tasks for that dataset)
#              SBATCH --time  per dataset (see README at bottom).
#
# Example:
#   sbatch --export=ALL,WAVRX_DATASET=mdvr,WAVRX_TASKS='T13 T14 T15 T16' \
#          --time=2:00:00 train_wavrx.sbatch
#
# Pipeline:
#   1. rsync the per-dataset multi_L24 cache from /scratch to $SLURM_TMPDIR
#      (compute-local NVMe). fp16 on disk keeps this manageable.
#   2. Run `sdx single train` per task with --overrides routing the cache
#      path through SLURM_TMPDIR. num_workers=8 + persistent_workers=True
#      lives in sdx/configs/training.yaml.

set -euo pipefail

WAVRX_DATASET="${WAVRX_DATASET:?must set WAVRX_DATASET=<dataset>}"

module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc arrow python/3.12 rust

source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p logs

# Ray + dataloader BLAS hygiene — same as warm.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ray's object store defaults to 30% of *node* memory (psutil-detected,
# not cgroup-aware on this kernel) — on the 768 GB Fir nodes that grabs
# ~200 GB before any training. We don't need the object store much (only
# scalar configs round-trip between driver and trial), so cap hard via
# the constants Ray exposes as RAY_DEFAULT_OBJECT_STORE_*.
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=$((2 * 1024 * 1024 * 1024))  # 2 GiB
# Ray's memory monitor also derives its threshold from the full node
# memory; false-positive worker kills hit us before the cgroup OOM does.
export RAY_memory_monitor_refresh_ms=0
# hpopt.yaml uses gpu: 0 (so CPU jobs can schedule). Ray 2.54+ would
# otherwise hide CUDA from the actor under num_gpus=0 — see hpopt.yaml
# for the smoke-run reference. Keep CUDA_VISIBLE_DEVICES intact so the
# trial actor lands on the slice we requested.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

nvidia-smi || true

# Same MIG pre-check used by warm — bail before the heavy rsync if this
# node's MIG handoff is broken.
set +e
python -c "import torch; torch.zeros(1).to('cuda'); torch.cuda.synchronize(); print('[sbatch] CUDA pre-check OK')" 2>&1
CUDA_RC=$?
set -e
if [[ "$CUDA_RC" -ne 0 ]]; then
    echo "[sbatch] CUDA pre-check FAILED on $(hostname) — MIG handoff likely broken. Exiting non-zero so Slurm can backfill us elsewhere."
    exit 1
fi

# ---- Cache rsync ----------------------------------------------------------
# Cache dir name is multi_L<num_layers> (wavlm-base+ → multi_L12,
# wavlm-large → multi_L24); resolve by reading the encoder yaml so the
# script doesn't go stale when we flip wavlm.yaml.
NUM_LAYERS=$(awk '/^num_layers:/ {print $2; exit}' sdx/configs/encoders/wavlm.yaml)
CACHE_MODE="multi_L${NUM_LAYERS}"
SRC_CACHE="/scratch/kieu/Audio-Health-Benchmark/embeddings_avg_finalv2/${WAVRX_DATASET}/wavlm"
DST_CACHE="$SLURM_TMPDIR/embeddings_avg_finalv2/${WAVRX_DATASET}/wavlm"

if [[ ! -d "$SRC_CACHE/train/$CACHE_MODE" || ! -d "$SRC_CACHE/val/$CACHE_MODE" ]]; then
    echo "[sbatch] ERROR: missing $CACHE_MODE cache at $SRC_CACHE — warm hasn't completed for ${WAVRX_DATASET}."
    exit 2
fi

mkdir -p "$DST_CACHE/train" "$DST_CACHE/val"
echo "[sbatch] rsync $SRC_CACHE/{train,val}/$CACHE_MODE  →  $DST_CACHE ..."
t0=$(date +%s)
# Two top-level splits (train, val) each containing $CACHE_MODE/cache.hdf5.
# -a preserves mtime so partial resumes via rsync don't re-copy. We only
# copy the active cache_mode subtree; any sibling caches (e.g. legacy
# single/) on /scratch are skipped.
rsync -a --info=stats2 "$SRC_CACHE/train/$CACHE_MODE" "$DST_CACHE/train/"
rsync -a --info=stats2 "$SRC_CACHE/val/$CACHE_MODE"   "$DST_CACHE/val/"
echo "[sbatch] rsync took $(( $(date +%s) - t0 ))s"
du -sh "$DST_CACHE" || true

# ---- Train ----------------------------------------------------------------
# slurm_tmpdir override reroutes train_cache_dir / val_cache_dir from
# the persistent /scratch path to the freshly-rsynced SLURM_TMPDIR copy.
SLURM_TMPDIR_REPO_REL="$SLURM_TMPDIR/embeddings_avg_finalv2/"
OVERRIDES="slurm_tmpdir: ${SLURM_TMPDIR_REPO_REL}"

if [[ -z "${WAVRX_TASKS:-}" ]]; then
    # Default: every task whose stem maps to WAVRX_DATASET. Cheap python
    # call into the registry — avoids hard-coding the mapping in this script.
    WAVRX_TASKS=$(python -c "
from sdx.orchestrator import discover_tasks
for t in discover_tasks(datasets=['${WAVRX_DATASET}'], tasks=None):
    print(t)
")
fi

echo "[sbatch] WAVRX_TASKS=$(echo $WAVRX_TASKS | tr '\n' ' ')"

FAILED=()
for TASK in $WAVRX_TASKS; do
    echo "[sbatch] ==== train $TASK × wavlm × wavrx ===="
    set +e
    python -u -m sdx single train -t "$TASK" -e wavlm \
        --tag run1 --workers 1 --overrides "$OVERRIDES" --leaf
    RC=$?
    set -e
    if [[ "$RC" -ne 0 ]]; then
        echo "[sbatch] $TASK FAILED rc=$RC"
        FAILED+=("$TASK")
    fi
done

if (( ${#FAILED[@]} > 0 )); then
    echo "[sbatch] failed tasks: ${FAILED[*]}"
    exit 3
fi

echo "[sbatch] all WAVRX_TASKS completed for dataset=${WAVRX_DATASET}."
exit 0
