#!/bin/bash
#SBATCH --job-name=warm_wavlm_mig
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/warm_wavlm_mig_%j.out
#SBATCH --error=logs/warm_wavlm_mig_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL
# Known-bad MIG handoff nodes — see memory/fir_mig_per_node_broken.md.
# Saves the runtime pre-check from firing repeatedly on the same node.
# Append more as we discover them. If this list gets long, file a ticket.
#SBATCH --exclude=fc10713
# Slurm sends SIGUSR1 to the batch shell 180 s before the wall-time TIMEOUT
# fires (then SIGTERM, then SIGKILL). We trap SIGUSR1 to resubmit the rolling
# chain — without this, TIMEOUT kills the script before the resubmit branch
# at the bottom of the file runs, and the chain dies after a single 3-h
# cycle. With it, a clean TIMEOUT is just another "non-zero exit" event from
# the chain's perspective. The `B:` prefix targets the batch script itself
# (not the running step), which is what we need to fire our trap.
#SBATCH --signal=B:USR1@180

# MIG-targeted warm — 3g.40gb slice. Avoids run_with_data.sh's `unshare -rm`
# (which breaks MIG CUDA init, see memory/mig_unshare_incompat.md) by
# extracting data.sqfs to $SLURM_TMPDIR and symlinking data/ at it via
# run_with_data_local.sh. Per-cycle overhead: ~10 min to unsquashfs the
# 56 GB image — negligible against a 3 h cycle and a much shorter MIG
# queue wait. Rolling resubmit on non-zero exit; cycle counter caps at 12.
#
# Knobs to touch:
#   - sdx/configs/encoders/wavlm.yaml :: warm_batch_size  (currently 128)
#   - sdx/configs/encoders/wavlm.yaml :: attn_impl        (currently flex)
#   - sdx/configs/main.yaml :: cache_pool                 (currently none)

set -euo pipefail

module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc arrow python/3.12 ffmpeg rust

source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE" logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# expandable_segments lets PyTorch's caching allocator grow segments on
# demand instead of holding large fixed-size blocks. Sweeping hundreds of
# variable-size batches (post-OOM-shrink) fragments the cache hard
# otherwise — OOMs in the high-30s GiB territory on a 40 GB MIG even when
# the *live* working set is only a few GiB. The CUDA OOM error message
# itself recommends this on fragmentation-bounded failures.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WARM_CYCLE="${WARM_CYCLE:-1}"
WARM_CYCLE_MAX="${WARM_CYCLE_MAX:-12}"
echo "[sbatch] cycle ${WARM_CYCLE}/${WARM_CYCLE_MAX}  job=${SLURM_JOB_ID:-?}  (MIG 3g.40gb)"

# Pre-TIMEOUT resubmit. Slurm fires SIGUSR1 180 s before the wall-time hits
# (see #SBATCH --signal above). We submit the next cycle here and let the
# current job die naturally — the warmer's chunk-level resume picks up
# wherever this cycle left off. Guarded so the trap only ever requeues once
# per job, even if Slurm decides to redeliver USR1.
REQUEUED_FROM_TRAP=0
on_timeout_signal() {
    if (( REQUEUED_FROM_TRAP )); then return; fi
    REQUEUED_FROM_TRAP=1
    if (( WARM_CYCLE >= WARM_CYCLE_MAX )); then
        echo "[sbatch] SIGUSR1: hit WARM_CYCLE_MAX=${WARM_CYCLE_MAX} — not requeueing on timeout."
        return
    fi
    NEXT_CYCLE=$(( WARM_CYCLE + 1 ))
    echo "[sbatch] SIGUSR1 (pre-TIMEOUT): submitting cycle ${NEXT_CYCLE}/${WARM_CYCLE_MAX} now so the chain survives the wall-time kill."
    sbatch --export=ALL,WARM_CYCLE="$NEXT_CYCLE",WARM_CYCLE_MAX="$WARM_CYCLE_MAX" "$0" || \
        echo "[sbatch] SIGUSR1: requeue submission failed — chain may end here."
}
trap on_timeout_signal SIGUSR1

nvidia-smi || true

# Fast CUDA pre-check (~5 s) before the 10-min unsquashfs. Some Fir GPU
# nodes have a broken MIG handoff to PyTorch: CUDA_VISIBLE_DEVICES is set
# correctly, nvidia-smi sees the MIG instance, but `.to('cuda')` raises
# cudaErrorDevicesUnavailable. Bail fast if we landed on a bad node and let
# the resubmit branch below land us elsewhere — don't `exit 1` here or
# we'd skip the resubmit and Slurm would just mark the cycle FAILED.
set +e
python -c "import torch; torch.zeros(1).to('cuda'); torch.cuda.synchronize(); print('[sbatch] CUDA pre-check OK')" 2>&1
CUDA_RC=$?
set -e

if [[ "$CUDA_RC" -ne 0 ]]; then
    echo "[sbatch] CUDA pre-check FAILED on $(hostname) — MIG handoff likely broken on this node. Skipping warm; loop will resubmit."
    RC=1
else
    # Run warmer in the background and `wait` so the SIGUSR1 trap above can
    # fire mid-warm. A foreground child blocks bash's signal dispatch — the
    # signal would only be processed after the child exits, which is too
    # late if Slurm sent SIGKILL behind it.
    set +e
    ./run_with_data_local.sh python -u -m sdx single warm -e wavlm &
    WARM_PID=$!
    # `wait` returns 128+SIGNUM when interrupted by a trapped signal; keep
    # waiting until the child actually finishes so $? reflects the real
    # warmer exit code.
    while :; do
        wait "$WARM_PID"
        RC=$?
        (( RC > 128 )) || break
    done
    set -e
fi

echo "[sbatch] sdx warm exit code: $RC"

if [[ "$RC" -eq 0 ]]; then
    echo "[sbatch] all wavlm caches warm — done."
    exit 0
fi

if (( REQUEUED_FROM_TRAP )); then
    echo "[sbatch] non-zero exit ($RC) but SIGUSR1 trap already requeued the next cycle — exiting cleanly."
    exit "$RC"
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
    exit 0
fi
echo "[sbatch] requeue itself failed — propagating exit code."
exit "$RC"
