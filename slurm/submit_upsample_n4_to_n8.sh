#!/bin/bash
#SBATCH --job-name=v3_up_n4_n8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v3_up_n4_n8_%j.out
#SBATCH --error=logs/v3_up_n4_n8_%j.err

# Hierarchical upsample n4 → n8 (parent broadcast) + residual = n8 - upsampled.
# Outputs:
#   /scratch/local/monan_ia_v3_n8_from_n4/  (coarse prior at n8)
#   /scratch/local/monan_ia_v3_n8_residual/ (high-freq target for cascade)

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/09_upsample_n4_to_n8_residual.py \
    --n4-dir  /scratch/local/monan_ia_v3_n4/normalized \
    --n8-dir  /scratch/local/monan_ia_v3_n8/normalized \
    --up-dir  /scratch/local/monan_ia_v3_n8_from_n4/normalized \
    --res-dir /scratch/local/monan_ia_v3_n8_residual/normalized \
    --n4-clim /scratch/local/monan_ia_v3_n4/climatology \
    --n8-clim /scratch/local/monan_ia_v3_n8/climatology \
    --up-clim-dir  /scratch/local/monan_ia_v3_n8_from_n4/climatology \
    --res-clim-dir /scratch/local/monan_ia_v3_n8_residual/climatology \
    --batch 128

echo ""
echo "=== output layouts ==="
for d in /scratch/local/monan_ia_v3_n8_from_n4 /scratch/local/monan_ia_v3_n8_residual; do
    echo "--- $d ---"
    ls -lh "$d/normalized/"   2>/dev/null | grep -v ^total
    echo "  .climatology:"
    ls -lh "$d/climatology/"  2>/dev/null | grep -v ^total
done
df -h /scratch/local | tail -2
echo "Finished: $(date)"
