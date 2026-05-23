#!/bin/bash
#SBATCH --job-name=v3_clim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v3_clim_%j.out
#SBATCH --error=logs/v3_clim_%j.err

# Compute DOY × hour climatology (mean + std) from the train memmap.
# Output: two NetCDFs in /scratch/local/monan_ia_v3/climatology/
# (one mean, one std), each variable shape (366, 4, 49152) float32.

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

MEM=/scratch/local/monan_ia_v3/normalized

python scripts/05_compute_climatology.py \
    --memmap "$MEM/train.npy" \
    --times  "$MEM/train_times.npy" \
    --index  "$MEM/train_index.json" \
    --out-dir /scratch/local/monan_ia_v3/climatology \
    --batch 128

echo ""
ls -lh /scratch/local/monan_ia_v3/climatology/
echo "Finished: $(date)"
