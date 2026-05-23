#!/bin/bash
#SBATCH --job-name=v3_degrade_n16
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/v3_degrade_n16_%A_%a.out
#SBATCH --error=logs/v3_degrade_n16_%A_%a.err

# Conservative downsampling nside=64 → nside=16 for all v3 monthly NetCDFs.
# Array of 8 tasks, each handles 1/8 of files (round-robin sharding by index).
# Output: /mnt/beegfs/otavio.feitosa/era5_healpix_v3_n16/{YYYY}/{YYYY}_{MM}.nc
# Each output file is ~22 MB (vs ~340 MB at nside=64) — total ~20 GB.

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/07_degrade_n64_to_n16.py \
    --src-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --dst-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3_n16 \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $(date)"
