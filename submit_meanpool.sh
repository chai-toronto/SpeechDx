#!/bin/bash -l
#SBATCH --job-name=meanpool_tmp2
#SBATCH --account=def-mariakak_gpu
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --mail-type=NONE

set -euo pipefail
mkdir -p exps/slurm_logs

module load StdEnv/2023 gcc arrow python/3.12 ffmpeg rust
source /scratch/kieu/speech-health-ai/spa/bin/activate

# 24 cpus = 3 CCDs aligned; meanpool streams one key at a time so RAM stays low.
python script/meanpool_cache.py --src tmp --dst tmp2 --workers "${SLURM_CPUS_PER_TASK}"
