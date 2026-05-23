#!/usr/bin/env python3
"""Compute DOY × hour climatology (mean + std) from train memmap.

For v3 we extend the v01/v02 DOY-only climatology with an explicit hour
dimension (0, 6, 12, 18 UTC). Each variable becomes (366, 4, n_pixels)
float32; one NetCDF for mean and one for std, mirroring the v01 schema
plus the new `hour` dimension.

Method (memory-bounded):
    Process the 4 hour buckets sequentially. For each hour:
        - allocate sum, sumsq (366, npix, n_vars) float32 (~17 GB each)
        - iterate the train memmap, accumulate only timesteps in this bucket
        - compute mean = sum/count, std = sqrt(sumsq/count - mean^2)
        - linear-interp DOY 60 (Feb 29) from DOY 59 & 61 (matches v01)
        - write that hour's slice into the output NetCDFs and free arrays

Peak RAM: ~34 GB (sum + sumsq) + a chunk read buffer.

Output: in --out-dir
    climatology_doy_hour_mean_nside64_v3.nc
    climatology_doy_hour_std_nside64_v3.nc

Each variable shape (366, 4, 49152) float32. Z-score-normalized space
(same transforms applied during 03_normalize).
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np


# ---------------------------------------------------------------------------
def compute_doy_hour_indices(times64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-366 DOY (1..366) and hour bucket (0..3) for each datetime64[ns].

    Fixed calendar: Feb 29 → DOY 60 always. In non-leap years, March 1
    is DOY 61 (shifted up by 1 compared to tm_yday).
    """
    import pandas as pd
    ts = pd.to_datetime(times64)
    yday = ts.dayofyear.to_numpy()
    years = ts.year.to_numpy()
    hours = ts.hour.to_numpy()
    is_leap = np.array([calendar.isleap(int(y)) for y in years])
    doy_fixed = np.where(is_leap | (yday <= 59), yday, yday + 1).astype(np.int32)
    if (hours % 6 != 0).any():
        raise ValueError("Found timesteps not aligned to 6h cadence")
    hour_idx = (hours // 6).astype(np.int32)
    return doy_fixed, hour_idx


# ---------------------------------------------------------------------------
def open_output_ncs(out_dir: Path, var_list: list[str], npix: int,
                    period: str) -> tuple[nc.Dataset, nc.Dataset, dict, dict]:
    """Create empty mean+std NetCDFs with all variables pre-declared."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_path = out_dir / "climatology_doy_hour_mean_nside64_v3.nc"
    std_path = out_dir / "climatology_doy_hour_std_nside64_v3.nc"

    def _new_ds(path: Path, title: str) -> tuple[nc.Dataset, dict]:
        ds = nc.Dataset(path, "w", format="NETCDF4")
        ds.createDimension("dayofyear", 366)
        ds.createDimension("hour", 4)
        ds.createDimension("pixel", npix)
        doy_v = ds.createVariable("dayofyear", "i4", ("dayofyear",))
        doy_v[:] = np.arange(1, 367, dtype=np.int32)
        hour_v = ds.createVariable("hour", "i4", ("hour",))
        hour_v[:] = np.array([0, 6, 12, 18], dtype=np.int32)
        hour_v.units = "UTC hour-of-day"
        pix_v = ds.createVariable("pixel", "i4", ("pixel",))
        pix_v[:] = np.arange(npix, dtype=np.int32)
        ds.title = title
        ds.data_space = "z-score normalized (power transform or asinh + standardized)"
        ds.healpix_nside = 64
        ds.healpix_npix = npix
        ds.healpix_ordering = "RING"
        ds.climatology_period = period
        ds.hour_bucket = "UTC 0/6/12/18; index 0=00Z, 1=06Z, 2=12Z, 3=18Z"
        ds.note_doy60 = "DOY 60 (Feb 29) interpolated from DOY 59 & 61"
        ds.n_variables = len(var_list)
        ds.variable_order = json.dumps(var_list)
        ds.processing_date = datetime.utcnow().isoformat()
        var_handles = {}
        for v in var_list:
            # No compression + 1-chunk-per-hour to match write pattern.
            # File size: 235 vars × 366 × 4 × 49152 × 4 bytes ≈ 6.5 GB per file.
            # The trade-off is worth it: avoids zlib dirty-buffer accumulation
            # that ate our RAM across hour passes.
            var = ds.createVariable(
                v, "f4", ("dayofyear", "hour", "pixel"),
                chunksizes=(366, 1, npix), zlib=False,
            )
            var.var_name = v
            var_handles[v] = var
        return ds, var_handles

    mean_ds, mean_vars = _new_ds(mean_path, "MONAN-AI v3 ERA5 DOY×hour climatological MEAN")
    std_ds, std_vars = _new_ds(std_path, "MONAN-AI v3 ERA5 DOY×hour climatological STD")
    return mean_ds, std_ds, mean_vars, std_vars


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", required=True, type=Path,
                   help="Path to train.npy (memmap output of 03_normalize.py)")
    p.add_argument("--times", required=True, type=Path,
                   help="Path to train_times.npy")
    p.add_argument("--index", required=True, type=Path,
                   help="Path to train_index.json (for var_list)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--batch", type=int, default=128,
                   help="Timesteps per memmap read batch (kept in float32 RAM)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "compute_climatology.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    log = logging.getLogger().info

    log("=" * 72)
    log("05_compute_climatology")
    log(f"  memmap  : {args.memmap}")
    log(f"  times   : {args.times}")
    log(f"  index   : {args.index}")
    log(f"  out_dir : {args.out_dir}")
    log("=" * 72)

    # Load index
    with open(args.index) as f:
        idx = json.load(f)
    var_list = idx["var_list"]
    n_steps_idx = idx["n_timesteps"]
    npix = idx["n_pixels"]
    n_vars = len(var_list)
    log(f"  n_steps : {n_steps_idx}, n_pixels: {npix}, n_vars: {n_vars}")

    # Open memmap (read-only)
    mm = np.load(args.memmap, mmap_mode="r")
    assert mm.shape == (n_steps_idx, npix, n_vars), \
        f"memmap shape {mm.shape} != expected ({n_steps_idx}, {npix}, {n_vars})"
    log(f"  memmap shape={mm.shape} dtype={mm.dtype}")

    # Compute DOY & hour for each timestep
    times = np.load(args.times)
    doy_arr, hour_arr = compute_doy_hour_indices(times)
    log(f"  hour distribution: " + ", ".join(
        f"{h*6:02d}Z={int((hour_arr == h).sum())}" for h in range(4)))
    log(f"  DOY range observed: [{int(doy_arr.min())}, {int(doy_arr.max())}]")

    # Period string
    period = f"{int(times.min().astype('datetime64[Y]').astype(int) + 1970)}-" \
             f"{int(times.max().astype('datetime64[Y]').astype(int) + 1970)}"
    log(f"  climatology_period: {period}")

    # Create output NetCDFs
    mean_ds, std_ds, mean_vars, std_vars = open_output_ncs(
        args.out_dir, var_list, npix, period,
    )
    log("  output NetCDFs created (empty)")

    # ---- One pass per hour bucket ----
    t_start = time.time()
    for h in range(4):
        log(f"---- hour bucket {h*6:02d}Z ----")
        t_h = time.time()
        t_steps = np.where(hour_arr == h)[0]
        log(f"    timesteps in bucket: {len(t_steps)}")
        if len(t_steps) == 0:
            log("    no timesteps; writing zeros")
            continue

        sum_arr = np.zeros((366, npix, n_vars), dtype=np.float32)
        sumsq_arr = np.zeros((366, npix, n_vars), dtype=np.float32)
        count = np.zeros(366, dtype=np.int64)

        # Batch read to amortize Python overhead
        for batch_start in range(0, len(t_steps), args.batch):
            batch_idx = t_steps[batch_start:batch_start + args.batch]
            # memmap fancy-indexed read → loads only those timesteps
            data = mm[batch_idx].astype(np.float32)  # (B, npix, n_vars)
            d_batch = doy_arr[batch_idx]
            for i, doy in enumerate(d_batch):
                idx0 = doy - 1
                sum_arr[idx0] += data[i]
                sumsq_arr[idx0] += data[i] * data[i]
                count[idx0] += 1
            if (batch_start // args.batch) % 10 == 0:
                log(f"      batch {batch_start}/{len(t_steps)} "
                    f"(elapsed={time.time()-t_h:.0f}s)")

        # Compute mean & std IN-PLACE on sum_arr and sumsq_arr to keep peak
        # RAM at ~34 GB. After this loop:
        #   sum_arr   holds the mean   (per DOY, per pix, per var)
        #   sumsq_arr holds the std    (same shape)
        log("    computing mean/std (in-place, per DOY) …")
        for d in range(366):
            n = int(count[d])
            if n == 0:
                sum_arr[d] = 0.0
                sumsq_arr[d] = 0.0
                continue
            inv_n = np.float32(1.0 / n)
            # mean = sum / n
            sum_arr[d] *= inv_n
            # variance = sumsq/n - mean^2
            sumsq_arr[d] *= inv_n
            # subtract mean^2 in place (small 73 MB temp per DOY)
            m_sq = sum_arr[d] * sum_arr[d]
            sumsq_arr[d] -= m_sq
            np.maximum(sumsq_arr[d], 0.0, out=sumsq_arr[d])
            np.sqrt(sumsq_arr[d], out=sumsq_arr[d])

        # Interpolate DOY 60 (Feb 29) from neighbours — match v01 protocol.
        # 1979-2021 has 11 leap years → DOY 60 has ~11 samples vs ~43 elsewhere.
        log(f"    interpolating DOY 60 (Feb 29; raw count={int(count[59])}) "
            "from DOY 59 & 61")
        sum_arr[59] = 0.5 * (sum_arr[58] + sum_arr[60])
        sumsq_arr[59] = 0.5 * (sumsq_arr[58] + sumsq_arr[60])

        # Write this hour's slice for every variable
        log("    writing to NetCDFs …")
        t_w = time.time()
        for j, v in enumerate(var_list):
            mean_vars[v][:, h, :] = sum_arr[:, :, j]
            std_vars[v][:, h, :] = sumsq_arr[:, :, j]
        log(f"    hour {h*6:02d}Z done (wrote {n_vars} vars in {time.time()-t_w:.0f}s, "
            f"total bucket time {time.time()-t_h:.0f}s)")

        # Flush this hour's writes to disk and force netCDF4 to release its
        # internal buffers. Critical: without sync, dirty pages accumulate
        # across hour passes and cause OOM. Also makes the file readable
        # even if a later hour fails.
        mean_ds.sync()
        std_ds.sync()

        # Free per-hour buffers before next iteration
        del sum_arr, sumsq_arr
        import gc
        gc.collect()

    mean_ds.close()
    std_ds.close()
    log("=" * 72)
    log(f"DONE in {time.time()-t_start:.0f}s")
    log(f"  mean: {args.out_dir / 'climatology_doy_hour_mean_nside64_v3.nc'}")
    log(f"  std : {args.out_dir / 'climatology_doy_hour_std_nside64_v3.nc'}")


if __name__ == "__main__":
    main()
