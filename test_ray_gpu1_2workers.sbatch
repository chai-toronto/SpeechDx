#!/bin/bash
#SBATCH --job-name=ray_gpu1_2workers
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ray_gpu1_2workers_%j.out
#SBATCH --error=logs/ray_gpu1_2workers_%j.err
#SBATCH --exclude=fc10713

# Ray-resource sanity test: with `sdx single train --workers 2` the
# orchestrator fans out into TWO leaf subprocesses, each of which calls
# ray.init() and tune.run() with resources_per_trial.gpu=1. Two separate
# Ray clusters, both claiming the same MIG slice — does one block on
# the other (starvation) or do they happily share the GPU at the CUDA
# level? That's the answer we need before deciding whether to drop to
# gpu=0.5 (one cluster, two fractional trials) instead.
#
# Pick T7 + T8: both non-CV, both on dementiabank, so the two
# subprocesses read the *same* cache file (53 GB train + 18 GB val) —
# h5py opens with locking=False so concurrent reads are safe. Reader-
# only: no encoder load, the 1g.10gb slice is plenty for a probe +
# Adam on a per-batch (16, T, 768) tensor.

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
# as train_wavrx.sbatch. The two leaf Ray clusters each pick this up.
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

# Overrides: scalar top-level keys only — they stamp cleanly through the
# fork machinery. A nested-dict override of a multi-line block target
# like hpopt_params would drop sibling keys: stamp_overrides rewrites
# the matched lines as a single flow-style scalar, so partial dict
# overrides lose every key that wasn't explicitly listed
# (continue_exp, optim_metric, ... → "reference not valid" at trial
# load time). The fork would need a full-merge stamp to handle that,
# which is a separate change.
#
#   slurm_tmpdir → read cache straight from /scratch (no rsync).
#   number_of_epochs → keep each trial short.
#
# Default num_samples=5 from hpopt.yaml stands. Each leaf runs 5 trials
# sequentially (max_concurrent_trials=1 inside each Ray cluster); the
# FIRST trial from each is the only overlap signal we need.
OVERRIDES=$(cat <<'YAML'
slurm_tmpdir: /scratch/kieu/Audio-Health-Benchmark/embeddings_avg_finalv2/
number_of_epochs: 3
YAML
)

echo "[sbatch] Overrides:"
printf '%s\n' "$OVERRIDES"
echo "[sbatch] ==== train T7 + T8 × wavlm × wavrx with --workers 2 ===="

# --tag picks an out-of-the-way exp folder so this test doesn't collide
# with any real wavrx run. --workers 2 → ThreadPoolExecutor(max_workers=2)
# in sdx/run.py dispatches the two (task, wavlm) pairs to two leaf
# subprocesses concurrently.
python -u -m sdx single train -t T7 -t T8 -e wavlm \
    --tag ray_gpu1_2workers_test --workers 2 \
    --overrides "$OVERRIDES"

echo "[sbatch] done."
