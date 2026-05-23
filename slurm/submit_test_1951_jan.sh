#!/bin/bash
#SBATCH --job-name=preproc_1951_01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preproc_1951_01_%j.out
#SBATCH --error=logs/preproc_1951_01_%j.err

# Smoke test: convert one month (Jan 1951) of ERA5 to HEALPix nside=64.
# /mnt/beegfs is only reachable from the gpu partition (g01); GPU itself unused.

echo "=========================================="
echo "MONAN-IA preproc -- 1951-01 smoke test"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Don't use `set -e` here: ~/.bashrc has a stray non-zero exit (`module: not found`)
# that would abort the script on entry. Conda activate is also tolerant by design.
source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python scripts/convert_month.py \
    --year 1951 --month 1 \
    --raw-root /mnt/beegfs/otavio.feitosa/era5 \
    --out-root /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \
    --nside-high 256 --nside-low 64

echo ""
echo "=========================================="
echo "Validating output..."
python scripts/validate_output.py \
    /mnt/beegfs/otavio.feitosa/era5_healpix_v3/1951/1951_01.nc

echo "Finished: $(date)"
