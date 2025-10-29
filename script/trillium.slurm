#!/bin/bash -l
#SBATCH --job-name=speech-health-ai
#SBATCH --chdir=$SCRATCH/kieu/speech-health-ai  # Set the working directory
#SBATCH --output=exps/slurm_logs/%x_%j.out     # Output file (%x=job-name, %j=job-id)
#SBATCH --error=exps/slurm_logs/%x_%j.err      # Error file
#SBATCH --time=7:00:00             # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=96          # Number of CPU cores per task
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gpus-per-node=4         # Number of GPUs per node
#SBATCH --mail-type=END,FAIL        # Optional: email on job end/failure
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca  # Your email address

conda init
bash
conda activate speech-health-ai

nvidia-smi -l 10 -q -d UTILIZATION,MEMORY > exps/slurm_logs/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

python -m training.train training/config/main.yaml

kill $GPU_MONITOR_PID