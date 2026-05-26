#!/bin/bash
#SBATCH --job-name=warm_wavlm
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --output=logs/warm_wavlm_%j.out
#SBATCH --error=logs/warm_wavlm_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Warm the wavlm cache for every dataset, with cache_pool=none (full T x D
# per uid in single/, not single_avg/). Rolling 3 h per cycle on a full
# H100 (NOT a MIG slice — see "MIG + unshare incompatibility" below).
# On non-zero exit (timeout, OOM, partial, per-pair failures), the job
# resubmits itself; on exit 0 (cache fully warm) it stops. `run_warm` is
# idempotent and skips already-warm (uid, version) entries via the
# per-(dataset, encoder) writer lock, so resubmits accumulate progress.
#
# MIG + unshare incompatibility (2026-05-22, job 41067511):
#   `run_with_data.sh` enters `unshare -rm` to mount data.sqfs as the user.
#   On a MIG slice (`--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1`) the MIG
#   compute instance is visible to nvidia-smi (which runs outside unshare)
#   but CUDA inside the namespace fails with cudaErrorDevicesUnavailable
#   at the first `t.to('cuda')` — the device-cgroup permissions for the
#   MIG instance don't propagate through the user namespace. The other
#   `run_with_data.sh` users (qwen3omni*) ask for a full H100, hiding
#   this. Until we replace the unshare mount with something MIG-friendly,
#   keep this on a full H100.
#
# Knobs you may need to touch:
#   - sdx/configs/encoders/wavlm.yaml :: warm_batch_size  (currently 128
#     with attn_impl=flex + compute_dtype=bf16; if anything OOMs, the
#     OOM-safe wrapper halves the batch automatically — but lower this
#     yaml value to make the recovery cheaper).
#   - sdx/configs/main.yaml :: cache_pool  (currently `none`; switch to
#     `mean` for the cheap D-vector cache under single_avg/).

set -euo pipefail

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc arrow python/3.12 ffmpeg rust

source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE" logs

# Single-process warm — keep BLAS threads from oversubscribing the 8 cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Bound the resubmit chain so a deterministic failure (e.g. persistent OOM)
# can't loop forever. Increment is set just before each requeue.
WARM_CYCLE="${WARM_CYCLE:-1}"
WARM_CYCLE_MAX="${WARM_CYCLE_MAX:-12}"
echo "[sbatch] cycle ${WARM_CYCLE}/${WARM_CYCLE_MAX}  job=${SLURM_JOB_ID:-?}"

nvidia-smi || true

# ------------------------------------------------------------------
# Run the warm sweep.
#
# data/ is a SquashFS image (data.sqfs); run_with_data.sh mounts it inside
# a user namespace so the absolute audio paths in the manifests resolve.
# `sdx single warm -e wavlm` (no -d / -t filters) sweeps every paper task,
# but cache state is keyed by (dataset, encoder) — so duplicate tasks per
# dataset cost only a near-instant uncached_ids() scan after the first one.
# ------------------------------------------------------------------
set +e
./run_with_data.sh python -u -m sdx single warm -e wavlm
RC=$?
set -e

echo "[sbatch] sdx warm exit code: $RC"

if [[ "$RC" -eq 0 ]]; then
    echo "[sbatch] all wavlm caches warm — done."
    exit 0
fi

if (( WARM_CYCLE >= WARM_CYCLE_MAX )); then
    echo "[sbatch] hit WARM_CYCLE_MAX=${WARM_CYCLE_MAX} without finishing." \
         "Refusing to requeue. Investigate the per-pair logs under" \
         "logs/warm/<timestamp>/ and rerun manually once fixed."
    exit "$RC"
fi

NEXT_CYCLE=$(( WARM_CYCLE + 1 ))
echo "[sbatch] non-zero exit ($RC) — resubmitting as cycle ${NEXT_CYCLE}/${WARM_CYCLE_MAX}."
if sbatch --export=ALL,WARM_CYCLE="$NEXT_CYCLE",WARM_CYCLE_MAX="$WARM_CYCLE_MAX" "$0"; then
    # Requeue submitted successfully — don't fail this cycle's job (avoids a
    # spurious FAIL email per resubmit).
    exit 0
fi
echo "[sbatch] requeue itself failed — propagating exit code."
exit "$RC"
