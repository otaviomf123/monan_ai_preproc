#!/bin/bash
#SBATCH --job-name=v3_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v3_check_%j.out
#SBATCH --error=logs/v3_check_%j.err

# End-to-end consistency checks for v3 preprocessed dataset.
# Reads from /scratch/local on g01 (gpu partition required).

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/06_consistency_checks.py \
    --norm-dir /scratch/local/monan_ia_v3/normalized \
    --clim-dir /scratch/local/monan_ia_v3/climatology \
    --max-z 10.0 \
    --n-finite-samples 200 \
    --n-stat-samples 400 \
    --n-perf-reads 64

echo "Finished: $(date)"
