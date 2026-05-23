#!/bin/bash
#SBATCH --job-name=preproc_v3
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-324%16
#SBATCH --output=logs/preproc_v3_%a_%A.out
#SBATCH --error=logs/preproc_v3_%a_%A.err

# Tier-1 v3 preproc on fat partition: years with COMPLETE auxiliary data
# (main + additional_levels + hydrometeors + ozone_extended).
# Range: 1950-1976 (27 years × 12 months = 324 array tasks).
# Every output file gets the full 236-var schema.
#
# Index mapping: array_id N → year = 1950 + (N-1)//12, month = ((N-1)%12)+1
#   N=1   → 1950-01
#   N=12  → 1950-12
#   N=324 → 1976-12
#
# %16 concurrent — moderate I/O load on /mnt/beegfs.
# --overwrite: 1951 was used as a smoke test; rewrite for consistency.

source ~/.bashrc
conda activate flash
set -eo pipefail

NIDX=$SLURM_ARRAY_TASK_ID
YEAR=$((1950 + (NIDX - 1) / 12))
MONTH=$(((NIDX - 1) % 12 + 1))

echo "=========================================="
echo "MONAN-IA preproc -- $YEAR-$(printf '%02d' $MONTH) (array task $NIDX)"
echo "Job: $SLURM_JOB_ID array $NIDX | Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/convert_month.py \
    --year "$YEAR" --month "$MONTH" \
    --raw-root /mnt/beegfs/otavio.feitosa/era5 \
    --out-root /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --nside-high 256 --nside-low 64 \
    --overwrite

echo "Finished: $(date)"
