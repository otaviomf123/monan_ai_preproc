#!/bin/bash
#SBATCH --job-name=preproc_1951
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-12%4
#SBATCH --output=logs/preproc_1951_m%a_%A.out
#SBATCH --error=logs/preproc_1951_m%a_%A.err

# Full-year smoke test as array job: 12 months × 1951, max 4 concurrent tasks.
# (No `set -u`: conda's deactivate hooks reference unset variables.)

MONTH=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "MONAN-IA preproc -- 1951-${MONTH} (array task)"
echo "Job: $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Don't use `set -e` before conda activate: ~/.bashrc throws a benign non-zero.
source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/convert_month.py \
    --year 1951 --month "$MONTH" \
    --raw-root /mnt/beegfs/otavio.feitosa/era5 \
    --out-root /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --nside-high 256 --nside-low 64

echo "Finished: $(date)"
