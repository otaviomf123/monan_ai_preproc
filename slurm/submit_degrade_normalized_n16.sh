#!/bin/bash
#SBATCH --job-name=v3_norm_n16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3_norm_n16_%j.out
#SBATCH --error=logs/v3_norm_n16_%j.err

# Degrade the v3 NORMALIZED memmap dataset from nside=64 to nside=16
# using conservative spatial averaging (nanmean of 16 children per pixel).
# Same normalization (same stats.json + transforms) is preserved across
# resolutions — the coarse data is the spatial average of the fine z-scores.
#
# Reads + writes on /scratch/local on g01 (gpu partition required).

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/08_degrade_normalized_n64_to_n16.py \
    --src-dir  /scratch/local/monan_ia_v3/normalized \
    --dst-dir  /scratch/local/monan_ia_v3_n16/normalized \
    --clim-src /scratch/local/monan_ia_v3/climatology \
    --clim-dst /scratch/local/monan_ia_v3_n16/climatology \
    --batch 64

echo ""
echo "=== output layout ==="
ls -lh /scratch/local/monan_ia_v3_n16/normalized/  2>/dev/null
ls -lh /scratch/local/monan_ia_v3_n16/climatology/ 2>/dev/null
df -h /scratch/local | tail -2
echo "Finished: $(date)"
