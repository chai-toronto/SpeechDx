#!/bin/bash
#SBATCH --job-name=wavrx_c19_gpu
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/wavrx_c19_gpu_%A_%a.out
#SBATCH --error=logs/wavrx_c19_gpu_%A_%a.err
#SBATCH --exclude=fc10713
#SBATCH --signal=B:USR1@180

# c19sounds wavrx HP search on GPU — pulled off the CPU fanout because
# the 1.3 TB cache makes each epoch ~15 h on CPU (2335 iters × 23 s/it).
# On a 3g.40gb MIG the same iter is ~0.5 s → ~30 min/epoch.
#
# One MIG slice per task (array 0-4 → T19 T20 T21 T22 T23). Rolling loop:
# the SIGUSR1 trap requeues the SAME index on partial completion at wall
# hit, sdx/train.py auto-resumes the saved Tune state from /scratch.
#
# Reader-only, so no run_with_data wrapper. batch_size=4 keeps the paper
# STFT (~2.47 GB fwd per batch-unit on c19sounds' (B*12,500,768) input)
# at ~10 GB fwd + ~10 GB bwd, safe on the 40 GB slice. number_of_epochs
# stays at training.yaml's 50 (the protocol; early-stops at patience 5).

set -euo pipefail

module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc arrow python/3.12 rust

source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=$((2 * 1024 * 1024 * 1024))
export RAY_memory_monitor_refresh_ms=0
# hpopt.yaml is gpu: 0 (CPU-friendly default); keep CUDA visible to the
# Ray actor on a GPU host so the single trial lands on the MIG slice.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Cap Ray CPU count → fewer prestart workers (one --workers 1 leaf here,
# but keeps behavior consistent with the CPU jobs). See sdx/train.py.
export SDX_RAY_NUM_CPUS=8
# Chunk + gradient-checkpoint the modulation-block STFT (bit-identical
# output). Bounds the ~40 GB STFT intermediate to ~3.3 GB/chunk, which
# lets batch_size go back to the protocol 16 on a 40 GB MIG. See model/wavrx.py.
export WAVRX_MOD_CHUNK=16

WAVRX_CYCLE="${WAVRX_CYCLE:-1}"
WAVRX_CYCLE_MAX="${WAVRX_CYCLE_MAX:-8}"
TAG="${WAVRX_TAG:-c19_gpu}"

C19_TASKS=(T19 T20 T21 T22 T23)
TASK="${C19_TASKS[$SLURM_ARRAY_TASK_ID]}"
RESULTS="exps/single_task/${TASK}/wavlm-wavrx-${TAG}/test_results.txt"

echo "[sbatch] cycle ${WAVRX_CYCLE}/${WAVRX_CYCLE_MAX}  array_idx=$SLURM_ARRAY_TASK_ID  task=$TASK  node=$(hostname)  job=${SLURM_JOB_ID}"
nvidia-smi || true

# ---- Pre-TIMEOUT requeue trap ----------------------------------------------
REQUEUED_FROM_TRAP=0
on_timeout_signal() {
    if (( REQUEUED_FROM_TRAP )); then return; fi
    REQUEUED_FROM_TRAP=1
    if [[ -f "$RESULTS" ]]; then
        echo "[sbatch] SIGUSR1: $TASK complete — no requeue."
        return
    fi
    if (( WAVRX_CYCLE >= WAVRX_CYCLE_MAX )); then
        echo "[sbatch] SIGUSR1: hit WAVRX_CYCLE_MAX=${WAVRX_CYCLE_MAX} — not requeueing."
        return
    fi
    NEXT_CYCLE=$(( WAVRX_CYCLE + 1 ))
    echo "[sbatch] SIGUSR1 (pre-TIMEOUT): submitting cycle ${NEXT_CYCLE}/${WAVRX_CYCLE_MAX} for $TASK."
    sbatch --array="$SLURM_ARRAY_TASK_ID" \
        --export=ALL,WAVRX_CYCLE="$NEXT_CYCLE",WAVRX_CYCLE_MAX="$WAVRX_CYCLE_MAX",WAVRX_TAG="$TAG" \
        "$0" || echo "[sbatch] SIGUSR1: requeue failed — chain may end here."
}
trap on_timeout_signal SIGUSR1

# MIG handoff pre-check.
set +e
python -c "import torch; torch.zeros(1).to('cuda'); torch.cuda.synchronize(); print('[sbatch] CUDA pre-check OK')" 2>&1
CUDA_RC=$?
set -e
if [[ "$CUDA_RC" -ne 0 ]]; then
    echo "[sbatch] CUDA pre-check FAILED on $(hostname) — exit non-zero for backfill."
    exit 1
fi

# Scalar overrides only (fork-stamp can't take a partial hpopt_params
# dict — it drops sibling keys). number_of_epochs NOT overridden:
# training.yaml's 50 is the protocol. batch_size = protocol default 16
# now that the chunked modulation block (WAVRX_MOD_CHUNK) keeps the STFT
# intermediate small enough to fit the 40 GB MIG at B=16.
OVERRIDES=$(cat <<'YAML'
slurm_tmpdir: /scratch/kieu/Audio-Health-Benchmark/embeddings_avg_finalv2/
batch_size: 16
YAML
)

echo "[sbatch] ==== train $TASK × wavlm × wavrx on 3g.40gb MIG (batch=4, ep=50, cycle ${WAVRX_CYCLE}) ===="
set +e
python -u -m sdx single train -t "$TASK" -e wavlm \
    --tag "$TAG" --workers 1 \
    --overrides "$OVERRIDES" --leaf &
TRAIN_PID=$!
while :; do
    wait "$TRAIN_PID"
    RC=$?
    (( RC > 128 )) || break
done
set -e

echo "[sbatch] python exited rc=$RC  cycle=${WAVRX_CYCLE}  task=$TASK"

if [[ -f "$RESULTS" ]]; then
    echo "[sbatch] $TASK complete — chain done."
    exit 0
fi
if (( REQUEUED_FROM_TRAP )); then
    echo "[sbatch] SIGUSR1 trap already requeued — exiting."
    exit "$RC"
fi
if (( WAVRX_CYCLE >= WAVRX_CYCLE_MAX )); then
    echo "[sbatch] hit WAVRX_CYCLE_MAX=${WAVRX_CYCLE_MAX} without finishing $TASK."
    exit 1
fi
NEXT_CYCLE=$(( WAVRX_CYCLE + 1 ))
echo "[sbatch] partial (rc=$RC) — submitting cycle ${NEXT_CYCLE}/${WAVRX_CYCLE_MAX} for $TASK."
sbatch --array="$SLURM_ARRAY_TASK_ID" \
    --export=ALL,WAVRX_CYCLE="$NEXT_CYCLE",WAVRX_CYCLE_MAX="$WAVRX_CYCLE_MAX",WAVRX_TAG="$TAG" \
    "$0"
exit 0
