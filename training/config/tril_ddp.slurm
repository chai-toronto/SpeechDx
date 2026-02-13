#!/bin/bash -l
#SBATCH --job-name=speech-health-ai
#SBATCH --chdir=./  # Set the working directory
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt     # Output file (%x=job-name, %j=job-id)
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt      # Error file
#SBATCH --time=4:30:00             # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=24          # Number of CPU cores per task
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --gpus-per-node=4         # Number of GPUs per node
#SBATCH --mail-type=BEGIN,END,FAIL        # Optional: email on job end/failure
#SBATCH --mail-user=dat.kieu@mail.utoronto.ca  # Your email address

export HF_HOME=$SCRATCH/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 2. Fix your matplotlib config directory permissions
mkdir -p ~/.config/matplotlib
chmod 755 ~/.config

# 3. Set environment variables to avoid permission issues
export MPLCONFIGDIR=$SLURM_TMPDIR/tmp/matplotlib-$USER
export FONTCONFIG_PATH=$SLURM_TMPDIR/tmp/fontconfig-$USER
mkdir -p $MPLCONFIGDIR $FONTCONFIG_PATH

module load StdEnv/2023 gcc arrow python/3.12 ffmpeg rust
source ~/shai/bin/activate

nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used \
           --format=csv,noheader -l 1 > "exps/slurm_logs/$SLURM_JOB_ID.gpu.csv" &
LOGGER_PID=$!

cd /scratch/kieu/speech-health-ai

torchrun --standalone --nproc-per-node=$SLURM_GPUS_PER_NODE -m training.train training/config/main.yaml

kill $LOGGER_PID