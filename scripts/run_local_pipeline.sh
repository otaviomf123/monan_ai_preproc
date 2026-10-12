#!/usr/bin/env bash
# Rebuild the MONAN-AI v3 dataset (HEALPix nside=64, 235 vars) on a machine
# WITHOUT SLURM, reusing the frozen normalization stats in reference/stats/.
#
# NEVER runs 02_compute_stats.py: the stats/masks must stay identical to the
# ones the trained models (n8/n16/n32/n64, cascade, DA) were normalized with.
#
# Usage:
#   export RAW_ROOT=/path/era5            # raw ERA5 1° NetCDF (layout in docs/BUILD_V3_DATASET.md)
#   export OUT_ROOT=/path/era5_healpix_v3 # monthly HEALPix NetCDF (step 1 output, ~2.3 GB/month)
#   export NORM_DIR=/path/monan_ia_v3/normalized
#   export CLIM_DIR=/path/monan_ia_v3/climatology
#   export JOBS=8                         # parallel months in step 1 (~4 GB RAM each)
#   bash scripts/run_local_pipeline.sh <step>
#
# Steps (run in order, each one is restartable):
#   check      env + frozen stats md5
#   cache      precompute regrid/degrade index (once)
#   smoke      convert 2022-01 and 1979-01 and compare with the reference fingerprint
#   convert    convert every month 1978-12 .. 2024-12 (skips files that already exist)
#   verify     structural check of all monthly files
#   normalize  train/val/test memmaps with the frozen stats (float16, RING)
#   clim       DOY x hour climatology from train.npy
#   final      verify normalized splits vs reference + 06_consistency_checks
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${RAW_ROOT:?set RAW_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"
NORM_DIR="${NORM_DIR:-}"
CLIM_DIR="${CLIM_DIR:-}"
JOBS="${JOBS:-4}"
NORM_WORKERS="${NORM_WORKERS:-8}"
STATS="$REPO/reference/stats/healpix_statistics_v3.json"
MASKS="$REPO/reference/stats/healpix_masks_v3.nc"
LOGDIR="${LOGDIR:-$REPO/logs}"
mkdir -p "$LOGDIR"

# Split dates of the cluster build — DO NOT CHANGE (see reference/index/*.json).
TRAIN_START=1979-01-01; TRAIN_END=2021-11-30
VAL_START=2022-01-01;   VAL_END=2022-11-30
TEST_START=2023-01-01;  TEST_END=2024-12-31
# 1978-12 is needed only for the 6h accumulation window of 1979-01-01 00Z.
CONV_START_Y=1978; CONV_START_M=12
CONV_END_Y=2024;   CONV_END_M=12

step="${1:-}"

convert_one() {  # $1=year $2=month
    local y=$1 m=$2 f="$OUT_ROOT/$1/$(printf '%d_%02d' "$1" "$2").nc"
    if [[ -f "$f" ]]; then echo "skip $f"; return 0; fi
    python scripts/convert_month.py --year "$y" --month "$m" \
        --raw-root "$RAW_ROOT" --out-root "$OUT_ROOT" \
        > "$LOGDIR/convert_${y}_$(printf '%02d' "$m").log" 2>&1 \
        && echo "ok   $y-$m" || { echo "FAIL $y-$m (see $LOGDIR)"; return 1; }
}
export -f convert_one
export RAW_ROOT OUT_ROOT LOGDIR

case "$step" in
  check)
    python -c "import numpy, xarray, healpy, sklearn, netCDF4, pandas; print('numpy', numpy.__version__, '| xarray', xarray.__version__, '| healpy', healpy.__version__)"
    python scripts/verify_reference.py stats
    ls "$RAW_ROOT" | head
    ;;
  cache)
    python scripts/precompute_cache.py --cache-dir "$OUT_ROOT/_cache" --nside-high 256 --nside-low 64 --ring
    ;;
  smoke)
    convert_one 2022 1
    convert_one 1979 1
    python scripts/verify_reference.py month "$OUT_ROOT/2022/2022_01.nc" "$OUT_ROOT/1979/1979_01.nc"
    ;;
  convert)
    {
      echo "$CONV_START_Y $CONV_START_M"
      for y in $(seq $((CONV_START_Y + 1)) $CONV_END_Y); do
        for m in $(seq 1 12); do echo "$y $m"; done
      done
    } | xargs -P "$JOBS" -n 2 bash -c 'convert_one "$0" "$1"'
    ;;
  verify)
    python scripts/verify_reference.py months "$OUT_ROOT" 1979 2024
    ;;
  normalize)
    : "${NORM_DIR:?set NORM_DIR}"
    mkdir -p "$NORM_DIR"
    cp "$STATS" "$NORM_DIR/stats.json"; cp "$MASKS" "$NORM_DIR/masks.nc"
    for spec in "train $TRAIN_START $TRAIN_END" "val $VAL_START $VAL_END" "test $TEST_START $TEST_END"; do
      set -- $spec
      if [[ -f "$NORM_DIR/$1_index.json" ]]; then echo "skip $1 (index exists)"; continue; fi
      python scripts/03_normalize.py --data-dir "$OUT_ROOT" --stats "$STATS" --masks "$MASKS" \
        --out-dir "$NORM_DIR" --split "$1" --start-date "$2" --end-date "$3" \
        --max-z-clip 10 --workers "$NORM_WORKERS" 2>&1 | tee "$LOGDIR/normalize_$1.log"
    done
    ;;
  clim)
    : "${NORM_DIR:?set NORM_DIR}"; : "${CLIM_DIR:?set CLIM_DIR}"
    python scripts/05_compute_climatology.py --memmap "$NORM_DIR/train.npy" \
      --times "$NORM_DIR/train_times.npy" --index "$NORM_DIR/train_index.json" \
      --out-dir "$CLIM_DIR" --batch 128 2>&1 | tee "$LOGDIR/climatology.log"
    ;;
  final)
    : "${NORM_DIR:?set NORM_DIR}"; : "${CLIM_DIR:?set CLIM_DIR}"
    python scripts/verify_reference.py normalized "$NORM_DIR"
    python scripts/06_consistency_checks.py --norm-dir "$NORM_DIR" --clim-dir "$CLIM_DIR" \
      --max-z 10.0 --n-finite-samples 200 --n-stat-samples 400 --n-perf-reads 64 \
      2>&1 | tee "$LOGDIR/consistency.log"
    ;;
  *)
    sed -n '2,30p' "$0"; exit 1 ;;
esac
