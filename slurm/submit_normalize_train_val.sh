#!/bin/bash
#SBATCH --job-name=v3_norm_tv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v3_norm_tv_%j.out
#SBATCH --error=logs/v3_norm_tv_%j.err

# Normalize train + val splits into /scratch/local (NVMe on g01).
# train: 1979-01-01 .. 2021-11-30 (~62700 timesteps × 22 MB = ~1.4 TB)
# val:   2022-01-01 .. 2022-11-30 (~1336 timesteps × 22 MB = ~29 GB)
#
# Test split will be submitted separately after train+val are verified.

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

OUT=/scratch/local/monan_ia_v3/normalized
mkdir -p "$OUT"

echo "=== train ==="
python scripts/03_normalize.py \
    --data-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --stats    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_stats/healpix_statistics_v3.json \
    --masks    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_stats/healpix_masks_v3.nc \
    --out-dir  "$OUT" \
    --split    train \
    --start-date 1979-01-01 --end-date 2021-11-30 \
    --max-z-clip 10 --workers 8

echo ""
echo "=== val ==="
python scripts/03_normalize.py \
    --data-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --stats    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_stats/healpix_statistics_v3.json \
    --masks    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_stats/healpix_masks_v3.nc \
    --out-dir  "$OUT" \
    --split    val \
    --start-date 2022-01-01 --end-date 2022-11-30 \
    --max-z-clip 10 --workers 8

echo ""
echo "=== final file listing ==="
ls -lah "$OUT"
df -h /scratch/local
echo "Finished: $(date)"
