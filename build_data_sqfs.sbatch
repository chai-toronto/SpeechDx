#!/bin/bash
#SBATCH --job-name=mksqfs_data
#SBATCH --account=def-mariakak_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/kieu/Audio-Health-Benchmark/logs/mksqfs_%j.out

# Pack Audio-Health-Benchmark/data/ (~610k loose audio files, 1 inode each)
# into a single SquashFS image. Mounting it read-only with squashfuse gives
# the identical directory tree at 1 inode instead of ~610k. See HANDOFF / the
# inode-quota cleanup discussion.
set -euo pipefail
cd /scratch/kieu/Audio-Health-Benchmark

SRC=data
IMG=data.sqfs
TMP=data.sqfs.building

echo "[$(date)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "[$(date)] source entry count (files+dirs+links):"
find "$SRC" | wc -l

rm -f "$TMP"
echo "[$(date)] mksquashfs starting (zstd-9)..."
# -mem 8G: mksquashfs reads NODE memory, not the cgroup cap -- pin it well
# under --mem=24G so it is not OOM-killed.
mksquashfs "$SRC" "$TMP" \
    -comp zstd -Xcompression-level 9 \
    -noappend -no-progress -mem 8G \
    -processors "${SLURM_CPUS_PER_TASK:-16}"

mv "$TMP" "$IMG"
echo "[$(date)] BUILD OK -- image: $(du -h "$IMG" | cut -f1)"
echo "[$(date)] superblock:"
unsquashfs -s "$IMG"
echo "[$(date)] DONE"
