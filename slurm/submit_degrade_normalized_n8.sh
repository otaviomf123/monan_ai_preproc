#!/bin/bash
#SBATCH --job-name=v3_norm_n8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/v3_norm_n8_%j.out
#SBATCH --error=logs/v3_norm_n8_%j.err

# Degrade the v3 NORMALIZED memmap dataset from nside=16 to nside=8 using
# the same conservative spatial averaging used for n64→n16. We chain from
# the already-degraded n16 (rather than re-degrading from n64) because the
# nested hierarchy means averaging is exact either way and the n16 source
# is much smaller (84 GB vs 1.4 TB), making the I/O ~16× cheaper.

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/08_degrade_normalized_n64_to_n16.py \
    --src-dir  /scratch/local/monan_ia_v3_n16/normalized \
    --dst-dir  /scratch/local/monan_ia_v3_n8/normalized \
    --clim-src /scratch/local/monan_ia_v3_n16/climatology \
    --clim-dst /scratch/local/monan_ia_v3_n8/climatology \
    --nside-high 16 --nside-low 8 \
    --batch 64

echo ""
echo "=== output layout ==="
ls -lh /scratch/local/monan_ia_v3_n8/normalized/  2>/dev/null
ls -lh /scratch/local/monan_ia_v3_n8/climatology/ 2>/dev/null
df -h /scratch/local | tail -2
echo "Finished: $(date)"
