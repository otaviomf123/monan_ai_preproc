#!/bin/bash
#SBATCH --job-name=v3_norm_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3_norm_test_%j.out
#SBATCH --error=logs/v3_norm_test_%j.err

# Normalize test split into /scratch/local on g01.
# test: 2023-01-01 .. 2024-12-31 (~2924 timesteps × 22 MB ≈ 64 GB)
#
# 2025 is NOT included here — CDS is currently rejecting requests for
# 2025-Sep..Dec hourly data. Re-run later (with extended end-date) once
# those are available.

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

OUT=/scratch/local/monan_ia_v3/normalized

python scripts/03_normalize.py \
    --data-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --stats    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_stats/healpix_statistics_v3.json \
    --masks    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_stats/healpix_masks_v3.nc \
    --out-dir  "$OUT" \
    --split    test \
    --start-date 2023-01-01 --end-date 2024-12-31 \
    --max-z-clip 10 --workers 8

echo ""
ls -lah "$OUT"
df -h /scratch/local
echo "Finished: $(date)"
