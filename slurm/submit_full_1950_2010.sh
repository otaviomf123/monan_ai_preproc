#!/bin/bash
#SBATCH --job-name=preproc_v3
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-732%16
#SBATCH --output=logs/preproc_v3_%a_%A.out
#SBATCH --error=logs/preproc_v3_%a_%A.err

# Full v3 preproc run on fat partition.
# Range: 1950-2010 (61 years × 12 months = 732 array tasks).
# Tier 1 (1950-1976, 324 months): full 236-var output.
# Tier 2 (1977-2010, 408 months): ~222 vars (missing 14 o3_extended levels).
# Both will need re-run for affected o3 vars when ozone_extended download finishes.
#
# Index mapping: array_id N → year = 1950 + (N-1)//12, month = ((N-1)%12)+1
#   N=1   → 1950-01
#   N=12  → 1950-12
#   N=13  → 1951-01
#   N=732 → 2010-12
#
# %16 concurrent — moderate I/O load on /mnt/beegfs.
# --overwrite: reprocess regardless of existing files (so 1951 files get
# the new metadata + chunking encoding).

# Don't use `set -e` before conda activate (~/.bashrc throws benign non-zero).
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
