#!/bin/bash
# Submit run_all_slurm.sh on Fir with the right MIG slice / full H100 per encoder.
# Sizes come from measured edaic peak GPU memory (see memory/encoder_gpu_memory.md).
#
# Usage:
#   ./submit_fir.sh               # submit one job per encoder, all datasets/tasks
#   ./submit_fir.sh whisper       # submit one encoder
#   ./submit_fir.sh wavlm clap    # submit a subset
#   DATASET=edaic ./submit_fir.sh # restrict datasets (forwarded to run_all.py)
#   TASK=edaic_phq ./submit_fir.sh
#   DRY=1 ./submit_fir.sh         # print sbatch commands without submitting
set -euo pipefail

# encoder -> "GPU_FLAG|CPUS|MEM|JOBS"
declare -A PRESET=(
  [whisper]="--gpus-per-node=h100:1|12|256G|3"
  [hubert]="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1|8|128G|2"
  [mms]="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1|8|128G|2"
  [w2v2]="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1|8|128G|2"
  [wavjepa]="--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1|8|128G|2"
  [qwen3voice]="--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1|6|96G|2"
  [opera_gt]="--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1|4|48G|2"
  [wavlm]="--gpus-per-node=h100:1|12|256G|2"
  [clap]="--gpus-per-node=h100:1|12|256G|1"
  [emotion2vec]="--gpus-per-node=h100:1|12|256G|2"
  [ast]="--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1|6|192G|2"
  [audiomae]="--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1|4|48G|2"
)

if [[ $# -gt 0 ]]; then
  ENCODERS=("$@")
else
  ENCODERS=("${!PRESET[@]}")
fi

EXPORTS="ALL"
[[ -n "${DATASET:-}" ]] && EXPORTS="$EXPORTS,DATASET=$DATASET"
[[ -n "${TASK:-}"    ]] && EXPORTS="$EXPORTS,TASK=$TASK"

for ENC in "${ENCODERS[@]}"; do
  if [[ -z "${PRESET[$ENC]:-}" ]]; then
    echo "⚠ no preset for '$ENC' — skipping" >&2
    continue
  fi
  IFS='|' read -r GPU_FLAG CPUS MEM PRESET_JOBS <<< "${PRESET[$ENC]}"
  JOBS="${JOBS_OVERRIDE:-$PRESET_JOBS}"
  JOB_NAME="run_${ENC}"
  [[ -n "${DATASET:-}" ]] && JOB_NAME="${JOB_NAME}_${DATASET}"
  [[ -n "${TASK:-}"    ]] && JOB_NAME="${JOB_NAME}_${TASK}"
  CMD=(sbatch
       --job-name="$JOB_NAME"
       "$GPU_FLAG"
       --cpus-per-task="$CPUS"
       --mem="$MEM"
       ${TIME:+--time="$TIME"}
       --export="$EXPORTS,ENCODER=$ENC,JOBS=$JOBS"
       run_all_slurm.sh)
  echo "+ ${CMD[*]}"
  [[ -n "${DRY:-}" ]] || "${CMD[@]}"
done
