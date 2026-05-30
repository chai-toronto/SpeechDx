#!/bin/bash
#SBATCH --job-name=warm_wavrx
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/warm_wavrx_%j.out
#SBATCH --error=logs/warm_wavrx_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=fc10713
#SBATCH --signal=B:USR1@180

# Multi-layer WavLM-large warm for WavRx. Same MIG-safe + rolling-resubmit
# pattern as warm_wavlm_loop_mig.sbatch — see that file's header for the
# rationale. The only differences here:
#   - output_hidden_states=true on wavlm.yaml means the warmer writes a
#     ``multi_L24/`` cache (24 datasets per uid/v key); writer/reader
#     handle the tuple shape in sdx/dataio/cache.py.
#   - Per-layer storage is fp16 to fit the 24× footprint (sdx/warm.py
#     _encode_chunk_batch + sdx/dataio/cache.py _cache).
#
# Required env: pass ``-d <dataset>`` (and optionally ``-t T<N>``) via
# WARM_FILTER. Examples:
#   sbatch --export=ALL,WARM_FILTER='-d mdvr' warm_wavrx.sbatch
#   sbatch --export=ALL,WARM_FILTER='-d c19sounds' warm_wavrx.sbatch

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WARM_CYCLE="${WARM_CYCLE:-1}"
WARM_CYCLE_MAX="${WARM_CYCLE_MAX:-12}"
WARM_FILTER="${WARM_FILTER:-}"
echo "[sbatch] cycle ${WARM_CYCLE}/${WARM_CYCLE_MAX}  job=${SLURM_JOB_ID:-?}  (MIG 3g.40gb)  filter='${WARM_FILTER}'"

REQUEUED_FROM_TRAP=0
on_timeout_signal() {
    if (( REQUEUED_FROM_TRAP )); then return; fi
    REQUEUED_FROM_TRAP=1
    if (( WARM_CYCLE >= WARM_CYCLE_MAX )); then
        echo "[sbatch] SIGUSR1: hit WARM_CYCLE_MAX=${WARM_CYCLE_MAX} — not requeueing on timeout."
        return
    fi
    NEXT_CYCLE=$(( WARM_CYCLE + 1 ))
    echo "[sbatch] SIGUSR1 (pre-TIMEOUT): submitting cycle ${NEXT_CYCLE}/${WARM_CYCLE_MAX}."
    sbatch --export=ALL,WARM_CYCLE="$NEXT_CYCLE",WARM_CYCLE_MAX="$WARM_CYCLE_MAX",WARM_FILTER="$WARM_FILTER" "$0" || \
        echo "[sbatch] SIGUSR1: requeue submission failed — chain may end here."
}
trap on_timeout_signal SIGUSR1

nvidia-smi || true

# MIG handoff pre-check — bail fast if this node's MIG is broken so the
# rolling loop can land us elsewhere.
set +e
python -c "import torch; torch.zeros(1).to('cuda'); torch.cuda.synchronize(); print('[sbatch] CUDA pre-check OK')" 2>&1
CUDA_RC=$?
set -e

if [[ "$CUDA_RC" -ne 0 ]]; then
    echo "[sbatch] CUDA pre-check FAILED on $(hostname) — MIG handoff likely broken on this node. Skipping warm; loop will resubmit."
    RC=1
else
    set +e
    # data/ is a static symlink to data.orig/ — the original unpacked dataset
    # tree on /scratch. No per-job mount (no run_with_data.sh unshare; no
    # run_with_data_local.sh extract) → no contention on the shared symlink
    # path, so parallel warm jobs Just Work. The inode-quota concern that
    # motivated data.sqfs no longer applies (Fir /scratch is uncapped on
    # inodes per `lfs quota`).
    # shellcheck disable=SC2086  # word-split WARM_FILTER on purpose
    python -u -m sdx single warm -e wavlm $WARM_FILTER &
    WARM_PID=$!
    while :; do
        wait "$WARM_PID"
        RC=$?
        (( RC > 128 )) || break
    done
    set -e
fi

echo "[sbatch] sdx warm exit code: $RC"

if [[ "$RC" -eq 0 ]]; then
    echo "[sbatch] wavrx cache warm — done."
    exit 0
fi

if (( REQUEUED_FROM_TRAP )); then
    echo "[sbatch] non-zero exit ($RC) but SIGUSR1 trap already requeued the next cycle — exiting cleanly."
    exit "$RC"
fi

if (( WARM_CYCLE >= WARM_CYCLE_MAX )); then
    echo "[sbatch] hit WARM_CYCLE_MAX=${WARM_CYCLE_MAX} without finishing." \
         "Investigate logs/warm/<timestamp>/ and rerun manually once fixed."
    exit "$RC"
fi

NEXT_CYCLE=$(( WARM_CYCLE + 1 ))
echo "[sbatch] non-zero exit ($RC) — resubmitting as cycle ${NEXT_CYCLE}/${WARM_CYCLE_MAX}."
if sbatch --export=ALL,WARM_CYCLE="$NEXT_CYCLE",WARM_CYCLE_MAX="$WARM_CYCLE_MAX",WARM_FILTER="$WARM_FILTER" "$0"; then
    exit 0
fi
echo "[sbatch] requeue itself failed — propagating exit code."
exit "$RC"
