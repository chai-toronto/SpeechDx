#!/bin/bash
#SBATCH --job-name=q3o_all
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=15:00
#SBATCH --output=logs/q3o_%x_%j.out
#SBATCH --error=logs/q3o_%x_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

: "${TASK:?TASK env var required (e.g. TASK=T3 sbatch ...)}"

module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc python/3.12 opencv/4.12.0 cuda/12.6
source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export CUDA_HOME="${EBROOTCUDA:-$CUDA_HOME}"
export VLLM_USE_DEEP_GEMM=0
# Avoid HF API rate-limits when many jobs start simultaneously — model already cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HUB_CACHE" exps/qwen3omni_all logs

nvidia-smi || true

python scripts/test_qwen3omni_all.py \
  --task "$TASK" \
  --out-dir exps/qwen3omni_all \
  --max-duration 600 \
  --max-num-seqs 16 \
  --gpu-mem-util 0.92 \
  --max-new-tokens 4096
