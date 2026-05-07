#!/bin/bash -l
# Minimal Slurm driver for `run_all_cross.py` — same *shape* as `run_all_slurm.sh`,
# with only the extra bits cross-eval needs (`--no-writer`, optional array → encoder).
#
# Nibi vs the old Trillium-style `run_all_slurm.sh`:
#   - Here we do NOT set HF_*_OFFLINE (compute nodes need Hub for first-time weights
#     unless you pre-download everything to $SCRATCH/hf).
#   - Nibi usually requires an explicit GPU type, not plain `--gpus-per-node=1`.
#     Edit the #SBATCH line below (see comment).
#   - Ray + Tune + many parallel `training.train` processes are fragile on small
#     cgroup RAM; keep JOBS low (default 1). The long `run_all_cross_slurm.sh`
#     documents TMPDIR/RAY_TMPDIR/MIG — those fixes live in Python + the other
#     script if you hit OOM or Ray PENDING hangs.
#
#SBATCH --job-name=run_cross
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@120
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=1
## Nibi: use ONE of these (uncomment / adjust); plain `1` often fails on nibi.
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
## #SBATCH --gpus-per-node=h100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@mail.utoronto.ca

set -euo pipefail

export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

mkdir -p ~/.config/matplotlib
chmod 755 ~/.config 2>/dev/null || true

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
export FONTCONFIG_PATH="${FONTCONFIG_PATH:-/tmp/fontconfig-$USER}"
mkdir -p "$MPLCONFIGDIR" "$FONTCONFIG_PATH"

# Disk-backed Ray root (avoid tmpfs /tmp filling cgroup RAM)
export RAY_TMPDIR="${RAY_TMPDIR:-$SCRATCH/ray_tmp}"
mkdir -p "$RAY_TMPDIR"
export RAY_object_store_memory="${RAY_object_store_memory:-2000000000}"

module load StdEnv/2023 gcc python/3.11 arrow ffmpeg rust
source /scratch/aina10/SpeechDx/.venv/bin/activate

mkdir -p exps/slurm_logs
GPU_CSV="exps/slurm_logs/${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used \
           --format=csv,noheader -l 5 > "$GPU_CSV" &
LOGGER_PID=$!
trap 'kill $LOGGER_PID 2>/dev/null || true' EXIT TERM INT

# Optional: SLURM array → one encoder per task (same idea as the long script).
ENCODERS=(ast audiomae clap emotion2vec hubert mms opera_gt qwen3voice w2v2 wavjepa wavlm whisper)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${ENCODER:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID < ${#ENCODERS[@]} )); then
        ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
    else
        echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((${#ENCODERS[@]} - 1)))" >&2
        exit 2
    fi
fi

ENC_ARGS=()
if [[ -n "${ENCODER:-}" ]]; then
    for E in ${ENCODER//,/ }; do ENC_ARGS+=(--encoder="$E"); done
fi
DS_ARGS=()
if [[ -n "${DATASET:-}" ]]; then
    for D in ${DATASET//,/ }; do DS_ARGS+=(--dataset="$D"); done
fi
TASK_ARGS=()
if [[ -n "${TASK:-}" ]]; then
    for T in ${TASK//,/ }; do TASK_ARGS+=(--task="$T"); done
fi
TEST_ONLY_ARG=()
[[ -n "${TEST_ONLY:-}" ]] && TEST_ONLY_ARG+=(--test-only)
CACHE_ONLY_ARG=()
[[ -n "${CACHE_ONLY:-}" ]] && CACHE_ONLY_ARG+=(--cache-only)
NO_WRITER_ARG=()
if [[ "${NO_WRITER:-1}" == "1" && -z "${CACHE_ONLY:-}" ]]; then
    NO_WRITER_ARG+=(--no-writer)
fi

JOBS="${JOBS:-1}"

echo "=== cross run: ${ENC_ARGS[*]} ${DS_ARGS[*]} ${TASK_ARGS[*]} ${NO_WRITER_ARG[*]} ${TEST_ONLY_ARG[*]} ${CACHE_ONLY_ARG[*]}  (JOBS=$JOBS) ==="

python run_all_cross.py run \
    --device=cuda \
    -j "$JOBS" \
    "${ENC_ARGS[@]}" \
    "${DS_ARGS[@]}" \
    "${TASK_ARGS[@]}" \
    "${NO_WRITER_ARG[@]}" \
    "${TEST_ONLY_ARG[@]}" \
    "${CACHE_ONLY_ARG[@]}"

echo "=== finished. status: ✓"
