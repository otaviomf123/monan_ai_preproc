#!/bin/bash
#SBATCH --job-name=v3_stats
#SBATCH --partition=fat
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v3_stats_%j.out
#SBATCH --error=logs/v3_stats_%j.err

# Compute v3 dataset stats + masks (Phases A/B/C).
# 32 worker processes, each owns ~8 of 235 variables and streams all 908+ files.
# Memory: ~9 MB per-pixel accumulator state per worker × 32 = ~300 MB.
# I/O: each file opened ~32 times (once per worker chunk). Could optimize by
# routing files to single workers but pickling overhead vs simplicity favors
# this layout for ~30 min jobs.

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/02_compute_stats.py \
    --data-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --max-z 5.0 \
    --workers 32

echo "Finished: $(date)"
