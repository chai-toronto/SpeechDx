#!/bin/bash
#SBATCH --job-name=wavrx_cpu_long_loop
#SBATCH --account=def-mariakak_cpu
#SBATCH --cpus-per-task=192
#SBATCH --mem=740G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/wavrx_cpu_long_loop_%A_%a.out
#SBATCH --error=logs/wavrx_cpu_long_loop_%A_%a.err
#SBATCH --signal=B:USR1@180

# Rolling CPU fanout for the non-c19sounds single tasks at the real
# ep=50 protocol. Two array indices, one whole node each:
#   idx 0 → 9 non-CV tasks   (tag cpu_long)
#   idx 1 → 13 CV tasks      (tag cpu_long_cv; auto-routes to train_cv.py)
# c19sounds (T19-T23) runs separately on GPU (wavrx_c19sounds_gpu.sbatch).
#
# Rolling: at ep=50 some tasks don't plateau-stop early and run ~50
# epochs × 5 trials = many hours, exceeding one 24 h wall. The SIGUSR1
# trap requeues the SAME array index on partial completion; sdx auto-
# resumes each task's Tune state (and SpeechBrain's 15-min checkpoints),
# so a cycle continues mid-trial rather than restarting it. Tags are
# preserved so resume finds the on-disk state from the earlier
# single-shot jobs (41840711 / 41858872).

set -euo pipefail

module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc arrow python/3.12 rust
source /scratch/kieu/speech-health-ai/spa/bin/activate
cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p logs

# OMP=32 + num_workers=0 matches the Phase-3 known-good config. The
# num_workers=8 experiment oversubscribed the node (6 leaf Ray clusters
# × ~192 prestart workers + 48 dataloader procs → GCS/actor RPC timeouts
# that killed trials). STFT is compute-bound, not IO-bound, so dropping
# DataLoader workers loses nothing.
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export BLIS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export TORCH_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=""
# Cap each leaf's Ray cluster CPU count → caps prestart workers. Without
# this, 3 leaves × ~192 prestart workers cold-start at once and overload
# the GCS ("node timed out during startup"), killing trials. Each leaf
# runs 1 concurrent trial × 2 cpu, so 8 is ample; the 32 OMP compute
# threads are unaffected (not Ray workers). See sdx/train.py.
export SDX_RAY_NUM_CPUS=8
# Chunk + gradient-checkpoint the modulation-block STFT (bit-identical
# output, verified). Bounds the ~40 GB STFT intermediate to ~3.3 GB/chunk,
# which is what lets us run --workers 6 again without OOM. See model/wavrx.py.
export WAVRX_MOD_CHUNK=16
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=$((2 * 1024 * 1024 * 1024))
export RAY_memory_monitor_refresh_ms=0

WAVRX_CYCLE="${WAVRX_CYCLE:-1}"
WAVRX_CYCLE_MAX="${WAVRX_CYCLE_MAX:-12}"

# 3 nodes: non-CV on one, the 13 CV tasks split across two (CV is the
# bottleneck — 5 folds × HP search each). idx1/idx2 share the
# cpu_long_cv tag with DISJOINT task sets, so their exp dirs never
# collide and each node's completion/requeue logic checks only its own
# subset.
TASK_BATCHES=(
    "T1 T2 T7 T8 T9 T24 T25 T26 T27"
    "T3 T4 T5 T6 T10 T11"
    "T12 T13 T14 T15 T16 T17 T18"
)
TAGS=(cpu_long cpu_long_cv cpu_long_cv)
TASKS="${TASK_BATCHES[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "[sbatch] cycle ${WAVRX_CYCLE}/${WAVRX_CYCLE_MAX}  idx=$SLURM_ARRAY_TASK_ID  tag=$TAG  node=$(hostname)  job=${SLURM_JOB_ID}"
echo "[sbatch] tasks: $TASKS"
python -c "import torch; print(f'[sbatch] torch={torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}')"

# Completion: CV tasks write test_results.yaml, others test_results.txt.
task_complete() {
    local t=$1 base="exps/single_task/$1/wavlm-wavrx-${TAG}"
    [[ -f "${base}/test_results.txt" || -f "${base}/test_results.yaml" ]]
}
all_complete() { local t; for t in $TASKS; do task_complete "$t" || return 1; done; return 0; }

REQUEUED_FROM_TRAP=0
on_timeout_signal() {
    (( REQUEUED_FROM_TRAP )) && return
    REQUEUED_FROM_TRAP=1
    if all_complete; then echo "[sbatch] SIGUSR1: all $TAG tasks complete — no requeue."; return; fi
    if (( WAVRX_CYCLE >= WAVRX_CYCLE_MAX )); then
        echo "[sbatch] SIGUSR1: hit WAVRX_CYCLE_MAX — not requeueing."; return
    fi
    local n=$(( WAVRX_CYCLE + 1 ))
    echo "[sbatch] SIGUSR1 (pre-TIMEOUT): submitting cycle ${n}/${WAVRX_CYCLE_MAX} idx=$SLURM_ARRAY_TASK_ID."
    sbatch --array="$SLURM_ARRAY_TASK_ID" \
        --export=ALL,WAVRX_CYCLE="$n",WAVRX_CYCLE_MAX="$WAVRX_CYCLE_MAX" "$0" || \
        echo "[sbatch] SIGUSR1: requeue failed."
}
trap on_timeout_signal SIGUSR1

OVERRIDES=$(cat <<'YAML'
slurm_tmpdir: /scratch/kieu/Audio-Health-Benchmark/embeddings_avg_finalv2/
precision: fp32
eval_precision: fp32
num_workers: 0
persistent_workers: False
YAML
)

T_ARGS=""
for t in $TASKS; do T_ARGS="$T_ARGS -t $t"; done

echo "[sbatch] ==== train [$TASKS] tag=$TAG --workers 6 (ep=50, cycle ${WAVRX_CYCLE}) ===="
set +e
# shellcheck disable=SC2086
python -u -m sdx single train $T_ARGS -e wavlm --tag "$TAG" --workers 6 --overrides "$OVERRIDES" &
PID=$!
while :; do wait "$PID"; RC=$?; (( RC > 128 )) || break; done
set -e
echo "[sbatch] python exited rc=$RC cycle=${WAVRX_CYCLE} tag=$TAG"

if all_complete; then echo "[sbatch] all $TAG tasks complete — chain done."; exit 0; fi
if (( REQUEUED_FROM_TRAP )); then echo "[sbatch] trap already requeued — exiting."; exit "$RC"; fi
if (( WAVRX_CYCLE >= WAVRX_CYCLE_MAX )); then echo "[sbatch] hit WAVRX_CYCLE_MAX without finishing $TAG."; exit 1; fi
n=$(( WAVRX_CYCLE + 1 ))
echo "[sbatch] partial — submitting cycle ${n}/${WAVRX_CYCLE_MAX} idx=$SLURM_ARRAY_TASK_ID."
sbatch --array="$SLURM_ARRAY_TASK_ID" \
    --export=ALL,WAVRX_CYCLE="$n",WAVRX_CYCLE_MAX="$WAVRX_CYCLE_MAX" "$0"
exit 0
