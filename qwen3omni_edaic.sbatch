#!/bin/bash
#SBATCH --job-name=qwen3omni_edaic
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/qwen3omni_edaic_%j.out
#SBATCH --error=logs/qwen3omni_edaic_%j.err

set -euo pipefail

module load StdEnv/2023 gcc arrow python/3.12 ffmpeg rust
source /scratch/kieu/speech-health-ai/spa/bin/activate

cd /scratch/kieu/Audio-Health-Benchmark

export HF_HOME=/scratch/kieu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HUB_CACHE" exps/qwen3omni_edaic logs

nvidia-smi || true

python scripts/test_qwen3omni_edaic.py \
  --csv data/edaic/processed/edaic.csv \
  --audio-root data/edaic/processed/audio \
  --out "exps/qwen3omni_edaic/predictions_${SLURM_JOB_ID}.csv" \
  --max-duration 600 \
  --use-transformers
