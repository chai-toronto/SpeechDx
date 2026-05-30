#!/bin/bash
#SBATCH --job-name=q3o_loop
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/q3o_loop_%j.out
#SBATCH --error=logs/q3o_loop_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=USR1@180     # USR1 to ALL job processes (incl. python) 3 min before time limit

set -euo pipefail

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc python/3.12 opencv/4.12.0 cuda/12.6
source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export CUDA_HOME="${EBROOTCUDA:-$CUDA_HOME}"
export VLLM_USE_DEEP_GEMM=0
# (Single job, no parallel hits → HF rate-limit not a concern; keep online so
# transformers' mistral-check can complete.)

# Cap BLAS / OMP threading — libblis aborts when threads requested != created.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$HF_HUB_CACHE" exps/qwen3omni_all logs

# Walltime budget for the python script: auto-derive from THIS job's slurm
# allocation. Always recompute — inheriting WALLTIME_S across requeues was a
# bug: a long-budget value (e.g. 5h) would leak into a 1h requeue cycle and
# the script would never decide to checkpoint before slurm killed it.
unset WALLTIME_S
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    WALLTIME_S=$(( SLURM_JOB_END_TIME - $(date +%s) - 240 ))
else
    WALLTIME_S=3540
fi
export WALLTIME_S
echo "[sbatch] WALLTIME_S=${WALLTIME_S}s (TimeLimit-based)"

nvidia-smi || true

# ------------------------------------------------------------------
# Run the all-tasks sweep with resume; on exit-2 (incomplete), requeue.
# ------------------------------------------------------------------
set +e
# data/ is a SquashFS image (data.sqfs); run_with_data.sh mounts it inside a
# user namespace so the absolute audio paths in the manifests resolve.
./run_with_data.sh python -u scripts/test_qwen3omni_all.py \
    --all \
    --out-dir exps/qwen3omni_all \
    --max-duration 600 \
    --max-num-seqs 16 \
    --gpu-mem-util 0.92 \
    --max-new-tokens 4096 \
    --chunk-size 64 \
    --walltime-s "$WALLTIME_S" \
    --safety-s 240
RC=$?
set -e

echo "[sbatch] python exit code: $RC"
if [[ "$RC" -eq 0 ]]; then
    echo "[sbatch] all tasks complete."
    exit 0
else
    # Treat exit 2 (incomplete) AND any crash as a requeue-able outcome,
    # provided some tasks still need work. The python script's
    # is_task_complete check will short-circuit if everything is done.
    echo "[sbatch] non-zero exit ($RC) — resubmitting self."
    if sbatch "$0"; then
        # Requeue submitted successfully — this cycle did its job. Report
        # success to Slurm so it doesn't email a FAIL notification.
        exit 0
    fi
    # Requeue itself failed — propagate the original error.
    exit "$RC"
fi
