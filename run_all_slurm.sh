#!/bin/bash -l
#SBATCH --job-name=run_enc
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=5:00:00
#SBATCH --signal=TERM@120
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca

export HF_HOME=$SCRATCH/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p ~/.config/matplotlib
chmod 755 ~/.config

export MPLCONFIGDIR=/tmp/matplotlib-$USER
export FONTCONFIG_PATH=/tmp/fontconfig-$USER
mkdir -p $MPLCONFIGDIR $FONTCONFIG_PATH
export RAY_TMPDIR=/scratch/kieu/ray_tmp
mkdir -p $RAY_TMPDIR

module load StdEnv/2023 gcc arrow python/3.12 ffmpeg rust
source /scratch/kieu/speech-health-ai/spa/bin/activate

nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used \
           --format=csv,noheader -l 1 > "$SLURM_JOB_ID.gpu.csv" &
LOGGER_PID=$!

ENC_ARGS=()
if [[ -n "${ENCODER:-}" ]]; then
    for ENC in ${ENCODER//,/ }; do
        ENC_ARGS+=(--encoder="$ENC")
    done
fi
DS_ARGS=()
if [[ -n "${DATASET:-}" ]]; then
    for DS in ${DATASET//,/ }; do
        DS_ARGS+=(--dataset="$DS")
    done
fi
TASK_ARGS=()
if [[ -n "${TASK:-}" ]]; then
    for T in ${TASK//,/ }; do
        TASK_ARGS+=(--task="$T")
    done
fi
TEST_ONLY_ARG=()
if [[ -n "${TEST_ONLY:-}" ]]; then
    TEST_ONLY_ARG+=(--test-only)
fi
echo "=== Packed run: ${ENC_ARGS[*]} ${DS_ARGS[*]} ${TASK_ARGS[*]} ${TEST_ONLY_ARG[*]} ==="
python -m ahb single run --device=cuda -j "${JOBS:-3}" "${ENC_ARGS[@]}" "${DS_ARGS[@]}" "${TASK_ARGS[@]}" "${TEST_ONLY_ARG[@]}"

kill $LOGGER_PID
