#!/bin/bash
#SBATCH --job-name=preproc_v3
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-60%16
#SBATCH --output=logs/preproc_v3_%a_%A.out
#SBATCH --error=logs/preproc_v3_%a_%A.err

# Tier-1 v3 preproc extension: 2019-2023 (5 years × 12 = 60 array tasks).
# All 4 sources verified complete on 2026-05-11.
#
# Index mapping: array_id N → year = 2019 + (N-1)//12, month = ((N-1)%12)+1

source ~/.bashrc
conda activate flash
set -eo pipefail

NIDX=$SLURM_ARRAY_TASK_ID
YEAR=$((2019 + (NIDX - 1) / 12))
MONTH=$(((NIDX - 1) % 12 + 1))

echo "=== MONAN-IA preproc -- $YEAR-$(printf '%02d' $MONTH) (array $NIDX) on $SLURM_NODELIST $(date) ==="
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
python scripts/convert_month.py \
    --year "$YEAR" --month "$MONTH" \
    --raw-root /mnt/beegfs/otavio.feitosa/era5 \
    --out-root /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --nside-high 256 --nside-low 64 \
    --overwrite
echo "Finished: $(date)"
