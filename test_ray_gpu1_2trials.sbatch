#!/bin/bash
#SBATCH --job-name=ray_gpu1_2trials
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ray_gpu1_2trials_%j.out
#SBATCH --error=logs/ray_gpu1_2trials_%j.err
#SBATCH --exclude=fc10713

# Ray-resource sanity test: does resources_per_trial.gpu=1 starve a second
# concurrent trial on a single MIG slice? We run `sdx single train -t T7
# -e wavlm` with max_concurrent_trials=2, num_samples=2, gpu=1. If Ray
# treats the MIG slice as 1 logical GPU (the expected behaviour), the
# second trial sits in PENDING until the first releases the slice; if it
# somehow runs both concurrently we'd see two `Ray Tune override:` blocks
# interleaved in the log within seconds of each other.
#
# Reader-only: T7 (dementiabank) has a complete multi_L12 wavlm cache on
# /scratch (53 GB train + 18 GB val) so we read straight from there
# instead of rsync'ing to SLURM_TMPDIR — the test only cares about Ray
# scheduling behavior, not IO.

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

# Cap Ray's object store and disable its memory monitor — same rationale
# as train_wavrx.sbatch (default 30% of node memory is psutil-detected,
# not cgroup-aware on Fir's 768 GB nodes).
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=$((2 * 1024 * 1024 * 1024))
export RAY_memory_monitor_refresh_ms=0

nvidia-smi || true

# MIG handoff pre-check — fail fast on a broken node so we can resubmit.
set +e
python -c "import torch; torch.zeros(1).to('cuda'); torch.cuda.synchronize(); print('[sbatch] CUDA pre-check OK')" 2>&1
CUDA_RC=$?
set -e
if [[ "$CUDA_RC" -ne 0 ]]; then
    echo "[sbatch] CUDA pre-check FAILED on $(hostname) — MIG handoff broken. Exit non-zero so Slurm can backfill."
    exit 1
fi

# Overrides:
#   slurm_tmpdir          → read cache straight from /scratch (no rsync).
#   max_concurrent_trials → ConcurrencyLimiter ceiling. With Ray's resource
#                           scheduler also gating, this is the upper bound,
#                           not a guarantee.
#   ray_tune_config       → 2 samples (one HP draw per trial), gpu=1.
#   number_of_epochs      → keep each trial short; we only need them
#                           overlapping (or not) for a few minutes.
OVERRIDES=$(cat <<'YAML'
slurm_tmpdir: /scratch/kieu/Audio-Health-Benchmark/embeddings_avg_finalv2/
max_concurrent_trials: 2
number_of_epochs: 3
ray_tune_config:
  num_samples: 2
  resources_per_trial:
    cpu: 2
    gpu: 1
YAML
)

echo "[sbatch] Overrides:"
printf '%s\n' "$OVERRIDES"
echo "[sbatch] ==== train T7 × wavlm × wavrx (Ray gpu=1, 2 trials) ===="

# --tag picks an out-of-the-way exp folder so this test doesn't collide
# with any real wavrx run.
python -u -m sdx single train -t T7 -e wavlm \
    --tag ray_gpu1_2trials_test --workers 1 \
    --overrides "$OVERRIDES" --leaf

echo "[sbatch] done."
