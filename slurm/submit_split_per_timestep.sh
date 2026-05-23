#!/bin/bash
#SBATCH --job-name=v3_split_ts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v3_split_ts_%j.out
#SBATCH --error=logs/v3_split_ts_%j.err

# Gera um arquivo .npy por timestep (6h) para val + test, em pasta separada.
# Útil para plots, inferência manual e debug. NÃO inclui train (muito grande).

source ~/.bashrc
conda activate flash
set -eo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

MEMMAP=/scratch/local/monan_ia_v3/normalized
OUT=/scratch/local/monan_ia_v3/normalized_per_step

python scripts/04_split_per_timestep.py \
    --memmap-dir "$MEMMAP" \
    --out-dir    "$OUT" \
    --splits val test

echo ""
echo "=== summary ==="
du -sh "$OUT"/* 2>/dev/null
ls "$OUT/val" | head -3
echo "..."
ls "$OUT/val" | tail -3
echo ""
ls "$OUT/test" | head -3
echo "..."
ls "$OUT/test" | tail -3
echo "Finished: $(date)"
