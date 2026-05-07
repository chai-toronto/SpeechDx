#!/bin/bash -l
#SBATCH --job-name=run_cross
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@120
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# In --no-writer mode run_all_cross.py inlines a StubEncoder (model.stub_encoder),
# so the heavy encoder weights are NEVER loaded. Still: each training subprocess
# runs its own Ray cluster (ray.init). JOBS>1 stacks N Ray heads + N plasmas in
# one cgroup → frequent OOM-kills (Ray "SYSTEM_ERROR / connection code 2").
# Default JOBS=1 serializes cross-tasks; override with sbatch --export=ALL,JOBS=2
# only if you have proven headroom (e.g. full H100 + 128G).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
# We use the SMALLEST H100 MIG slice (10 GB). For --no-writer probe training
# you only need ~2 GB VRAM; MIG slices start MUCH faster than full h100s.
# Override on the sbatch command line via --gpus-per-node=<type>:1.
# Available H100 MIG slices (def-mariakak_gpu has access to these):
#   nvidia_h100_80gb_hbm3_1g.10gb   10 GB  (smallest, fastest queue) <-- default
#   nvidia_h100_80gb_hbm3_2g.20gb   20 GB
#   nvidia_h100_80gb_hbm3_3g.40gb   40 GB
#   h100                            80 GB  (full card, most contended)
# The "o*" node GPUs (a5000, t4, l40s) may not be accessible to your account.
#
# ----- Submission patterns -----
# 1) ALL encoders × ALL cross_tasks in ONE job, --no-writer, -j 1 (default):
#       sbatch run_all_cross_slurm.sh
#
# 2) Array, one slurm task per encoder (recommended for fastest wall-clock).
#    12 encoders → indices 0..11 ; %4 caps to 4 concurrent array tasks.
#       sbatch --array=0-11%4 --time=2:00:00 run_all_cross_slurm.sh
#
# 3) Filter via env vars (works with or without --array):
#       sbatch --export=ALL,ENCODER=emotion2vec,wavlm                 run_all_cross_slurm.sh
#       sbatch --export=ALL,TASK=torgo_uaspeech_dysC,uaspeech_torgo_dysC run_all_cross_slurm.sh
#       sbatch --export=ALL,DATASET=torgo_uaspeech                       run_all_cross_slurm.sh
#       sbatch --export=ALL,TEST_ONLY=1                                  run_all_cross_slurm.sh
#       sbatch --export=ALL,JOBS=4                                       run_all_cross_slurm.sh
#       sbatch --export=ALL,NO_WRITER=0                                  run_all_cross_slurm.sh   # disable
#
# Multiple values in ENCODER/DATASET/TASK are comma-separated.

set -euo pipefail

# ----- Where the venv lives (built earlier on the login node) -----
VENV="/scratch/aina10/SpeechDx/.venv"

# ----- HF cache: keep on $SCRATCH so weights survive across jobs -----
# nibi compute nodes have outbound HTTPS, so we let HF download on first use.
# Each encoder gets cached once under $HF_HOME and reused by every later job.
# (Set HF_HUB_OFFLINE=1 here if you ever move this to a no-internet cluster.)
export HF_HOME="$SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# ----- matplotlib / fontconfig: per-job tmp on node-local NVMe -----
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib-$USER"
export FONTCONFIG_PATH="$SLURM_TMPDIR/fontconfig-$USER"
mkdir -p "$MPLCONFIGDIR" "$FONTCONFIG_PATH"

# ----- Ray tmp: use $SLURM_TMPDIR (disk-backed) to avoid OOM-from-tmpfs -----
# We previously routed Ray to /tmp to dodge AF_UNIX's 107-byte socket-path
# limit, but /tmp on nibi is tmpfs (RAM); 4 concurrent Ray instances filled
# RAM and OOM-killed the trial workers. Now we use $SLURM_TMPDIR DIRECTLY
# (no extra suffix) so the session-dir path stays under 107 bytes:
#   /local/scratch/<u>.<job>.<arr>          (32 chars)
#   + /ray/session_<ts>_<pid>/sockets/plasma_store  (~67 chars)
#   = ~99 chars  ✓
export TMPDIR="$SLURM_TMPDIR/t"
export RAY_TMPDIR="$SLURM_TMPDIR"
mkdir -p "$TMPDIR"
# Cap Ray's plasma object store (bytes). Probe training barely uses plasma;
# keep low so Ray + SpeechBrain + DataLoader stay under cgroup RAM.
export RAY_object_store_memory=${RAY_object_store_memory:-2000000000}

# ----- Module + venv -----
module --force purge
module load StdEnv/2023 gcc python/3.11 arrow ffmpeg rust
source "$VENV/bin/activate"

echo "=== node     : $(hostname)"
echo "=== job id   : ${SLURM_JOB_ID:-?}  array_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "=== python   : $(which python)  ($(python -V))"
echo "=== cpus     : ${SLURM_CPUS_PER_TASK:-?}"
echo "=== gpus     : $(nvidia-smi -L | wc -l)"

# ----- Lightweight GPU monitor (writes a CSV alongside the slurm logs) -----
mkdir -p exps/slurm_logs
GPU_CSV="exps/slurm_logs/${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used \
           --format=csv,noheader -l 5 > "$GPU_CSV" &
LOGGER_PID=$!
trap 'kill $LOGGER_PID 2>/dev/null || true' EXIT TERM INT

# ============================================================================
# Encoder list — kept in sync with what run_all_cross._default_encoders()
# discovers in training/config/encoders/*.yaml (alphabetical, no others/).
# Note the qwen3_voice→qwen3voice rename done by ENCODER_NAME_OVERRIDES.
# ============================================================================
ENCODERS=(ast audiomae clap emotion2vec hubert mms opera_gt qwen3voice w2v2 wavjepa wavlm whisper)

# If submitted as an array, pick the encoder for this array index (unless
# the user explicitly supplied ENCODER via --export already).
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${ENCODER:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID < ${#ENCODERS[@]} )); then
        ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
    else
        echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range (have ${#ENCODERS[@]} encoders)" >&2
        exit 2
    fi
fi

# ----- Build CLI flags from env vars (mirrors run_all_slurm.sh style) -----
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
# Default: --no-writer ON (we have averaged embeddings already cached).
# Disable explicitly via NO_WRITER=0 if you ever need the writer pass.
if [[ "${NO_WRITER:-1}" == "1" && -z "${CACHE_ONLY:-}" ]]; then
    NO_WRITER_ARG+=(--no-writer)
fi

JOBS="${JOBS:-1}"

echo "=== launching: run_all_cross.py run --device=cuda -j $JOBS \\"
echo "                ${ENC_ARGS[*]} ${DS_ARGS[*]} ${TASK_ARGS[*]} \\"
echo "                ${NO_WRITER_ARG[*]} ${TEST_ONLY_ARG[*]} ${CACHE_ONLY_ARG[*]}"

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
