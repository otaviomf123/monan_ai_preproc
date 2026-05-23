#!/bin/bash
#SBATCH --job-name=preproc_v3
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=1-612%16
#SBATCH --output=logs/preproc_v3_%a_%A.out
#SBATCH --error=logs/preproc_v3_%a_%A.err

# Tier-1 v3 preproc on fat partition (RING ordering, 235 vars).
# Range: 1950-2000 (51 years × 12 months = 612 array tasks).
#
# All 4 sources should be complete for these years:
#   main + additional_levels + hydrometeors + ozone_extended.
# 1998 has 1 missing additional_levels file (1 month) — that task will fail
# the strict 235-var check; rerun with --overwrite when source arrives.
#
# Index mapping: array_id N → year = 1950 + (N-1)//12, month = ((N-1)%12)+1
#   N=1   → 1950-01
#   N=612 → 2000-12
#
# Cache: /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_cache/ (pre-populated
# via scripts/precompute_cache.py — workers just reload).

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
