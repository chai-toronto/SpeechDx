#!/bin/bash
#SBATCH --job-name=cuda_probe_mig
#SBATCH --account=def-mariakak_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=logs/cuda_probe_mig_%j.out
#SBATCH --error=logs/cuda_probe_mig_%j.err

# Tiny probe — establish what CUDA sees on a 3g.40gb MIG slice on Fir. No
# heavy model load, no namespace, no data mount. If this fails the same
# way the warm jobs did, MIG itself can't host PyTorch on this cluster
# (rather than something specific to our warm setup).
set -euo pipefail
module purge --force >/dev/null 2>&1
module load StdEnv/2023 gcc python/3.12
source /scratch/kieu/speech-health-ai/spa/bin/activate

echo "=== Slurm + CUDA env vars ==="
env | grep -i -E '^(cuda|gpu|slurm_(gpus|gres|step_gpus|job_gpus|job_id|tasks_per_node)|sl(|cu)cs)' | sort
echo
echo "=== nvidia-smi from inside the job ==="
nvidia-smi
echo
echo "=== nvidia-smi -L (lists visible devices + MIG UUIDs) ==="
nvidia-smi -L
echo
echo "=== python: torch CUDA probe ==="
python <<'PY'
import os, sys, torch
print(f"CUDA_VISIBLE_DEVICES    : {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
print(f"CUDA_DEVICE_ORDER       : {os.environ.get('CUDA_DEVICE_ORDER')!r}")
print(f"SLURM_STEP_GPUS         : {os.environ.get('SLURM_STEP_GPUS')!r}")
print(f"SLURM_JOB_GPUS          : {os.environ.get('SLURM_JOB_GPUS')!r}")
print(f"torch.__version__       : {torch.__version__}")
print(f"torch.version.cuda      : {torch.version.cuda}")
print(f"torch.cuda.is_available : {torch.cuda.is_available()}")
try:
    n = torch.cuda.device_count()
    print(f"torch.cuda.device_count : {n}")
    if n > 0:
        print(f"current_device          : {torch.cuda.current_device()}")
        print(f"device 0 name           : {torch.cuda.get_device_name(0)}")
        print(f"device 0 props          : {torch.cuda.get_device_properties(0)}")
        x = torch.zeros(4, 4)
        print(f"CPU tensor              : ok ({x.shape}, {x.dtype})")
        try:
            y = x.to('cuda')
            print(f".to('cuda')             : SUCCESS, device={y.device}")
            z = (y @ y).sum()
            torch.cuda.synchronize()
            print(f"trivial matmul          : ok, sum={float(z)}")
        except Exception as e:
            print(f".to('cuda')             : FAILED {type(e).__name__}: {e}")
            sys.exit(2)
except Exception as e:
    print(f"device_count threw      : {type(e).__name__}: {e}")
    sys.exit(3)
PY
echo
echo "=== probe done ==="
