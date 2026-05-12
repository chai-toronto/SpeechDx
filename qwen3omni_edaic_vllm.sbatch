#!/bin/bash
#SBATCH --job-name=qwen3omni_edaic_vllm
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=45:00
#SBATCH --output=logs/qwen3omni_edaic_vllm_%j.out
#SBATCH --error=logs/qwen3omni_edaic_vllm_%j.err
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc python/3.12 opencv/4.12.0 cuda/12.6
source /scratch/kieu/speech-health-ai/spa/bin/activate

# flashinfer JIT needs nvcc + CUDA_HOME
export CUDA_HOME="${EBROOTCUDA:-$CUDA_HOME}"

# disable DeepGEMM FP8 warmup (package not installed; model is bf16 anyway)
export VLLM_USE_DEEP_GEMM=0

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE" exps/qwen3omni_edaic_vllm logs

nvidia-smi || true

python scripts/test_qwen3omni_edaic_vllm.py \
  --csv data/edaic/processed/edaic.csv \
  --audio-root data/edaic/processed/audio \
  --out "exps/qwen3omni_edaic_vllm/predictions_${SLURM_JOB_ID}.csv" \
  --max-duration 600 \
  --max-num-seqs 16 \
  --gpu-mem-util 0.92 \
  --max-new-tokens 4096
