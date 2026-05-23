#!/usr/bin/env python3
"""Conservative downsample of the v3 NORMALIZED memmap dataset from
HEALPix nside=64 (49152 px) to nside=16 (3072 px).

Operates on the already-normalized data so that BOTH resolutions share
the EXACT SAME normalization (same stats.json, same transforms). Each
nside=16 pixel is the nanmean of its 16 nside=64 children — purely
spatial averaging in the already-z-scored space.

Note on variance: averaging 16 z-scores of approximately-iid samples
reduces the apparent std by ~sqrt(16) = 4× (so values cluster near
zero). This is intentional — we are *not* re-z-scoring the coarse data.

Inputs (from --src-dir):
    {train,val,test}.npy            (n_steps, 49152, 235) float16 memmap
    {train,val,test}_times.npy
    {train,val,test}_index.json
    stats.json                       (passed through unchanged)
    masks.nc                         (degraded via OR-of-children)

Climatology (if --clim-src given):
    climatology_doy_hour_mean_nside64_v3.nc   (366, 4, 49152) per var
    climatology_doy_hour_std_nside64_v3.nc    same

Outputs (mirror layout in --dst-dir / --clim-dst):
    same filenames; pixel dim shrinks 49152 → 3072.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monan_preproc.healpix import HierarchicalDegrader


# Default source/target nsides; can be overridden via CLI.
NSIDE_HIGH = 64
NSIDE_LOW = 16
NPIX_HIGH = 12 * NSIDE_HIGH * NSIDE_HIGH
NPIX_LOW = 12 * NSIDE_LOW * NSIDE_LOW


# ---------------------------------------------------------------------------
def degrade_memmap_split(src_dir: Path, dst_dir: Path, split: str,
                          degrader: HierarchicalDegrader,
                          batch: int, log) -> None:
    src_arr_path = src_dir / f"{split}.npy"
    src_times_path = src_dir / f"{split}_times.npy"
    src_index_path = src_dir / f"{split}_index.json"
    if not src_arr_path.exists():
        log(f"  [{split}] source memmap not found, skipping: {src_arr_path}")
        return

    log(f"==== split={split} ====")
    src = np.load(src_arr_path, mmap_mode="r")
    times = np.load(src_times_path)
    with open(src_index_path) as f:
        index = json.load(f)
    n_steps, npix_src, n_vars = src.shape
    if npix_src != NPIX_HIGH:
        raise ValueError(f"unexpected source npix {npix_src} != {NPIX_HIGH}")
    if n_steps != len(times):
        raise ValueError(f"n_steps {n_steps} != len(times) {len(times)}")
    log(f"  src shape={src.shape} dtype={src.dtype}")

    # Pre-allocate destination memmap with same dtype (float16 by default)
    dst_arr_path = dst_dir / f"{split}.npy"
    dst_arr_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_arr_path.exists():
        dst_arr_path.unlink()
    log(f"  pre-allocating dst memmap: {dst_arr_path}")
    log(f"    shape=({n_steps}, {NPIX_LOW}, {n_vars}) dtype={src.dtype}")
    dst = np.lib.format.open_memmap(
        dst_arr_path, mode="w+", dtype=src.dtype,
        shape=(n_steps, NPIX_LOW, n_vars),
    )
    est_gb = n_steps * NPIX_LOW * n_vars * src.dtype.itemsize / 1e9
    log(f"    estimated size: {est_gb:.2f} GB")

    # Iterate timesteps in batches: read (B, 49152, n_vars) → degrade → write
    t0 = time.time()
    for start in range(0, n_steps, batch):
        end = min(start + batch, n_steps)
        block_f32 = src[start:end].astype(np.float32)   # (B, 49152, n_vars)
        # Degrade per-pixel axis (axis=1). Returns float32.
        out_f32 = degrader(block_f32, pix_axis=1)        # (B, 3072, n_vars)
        dst[start:end] = out_f32.astype(src.dtype)
        if (start // batch) % 20 == 0 or end == n_steps:
            log(f"    [{split}] {end}/{n_steps} "
                f"({time.time()-t0:.0f}s, {(end/n_steps)*100:.1f}%)")
    del dst   # flush

    # times.npy: just copy (same timesteps)
    np.save(dst_dir / f"{split}_times.npy", times)

    # Update index json
    new_index = dict(index)
    new_index["n_pixels"] = NPIX_LOW
    new_index["nside"] = NSIDE_LOW
    new_index["downsample_source"] = {
        "src_npix": NPIX_HIGH, "src_nside": NSIDE_HIGH,
        "method": "hierarchical nanmean (4 children per step, ×2)",
        "note": "Applied on already-normalized data — same stats/transforms shared "
                "across resolutions; downsampled std ≈ 1/√16 of source std.",
        "date": datetime.utcnow().isoformat() + "Z",
    }
    with open(dst_dir / f"{split}_index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    log(f"  [{split}] DONE in {time.time()-t0:.0f}s "
        f"(throughput: {n_steps/((time.time()-t0)+1e-9):.0f} timesteps/s)")


# ---------------------------------------------------------------------------
def degrade_masks(src_path: Path, dst_path: Path, log) -> None:
    """variation_mask + coverage_mask: OR-aggregate over the 16 children of
    each parent pixel (a coarse pixel is 'covered' if ANY of its children is)."""
    log(f"==== masks ====")
    log(f"  src: {src_path}")
    with xr.open_dataset(src_path) as ds:
        # boolean masks of shape (variable, pixel=49152)
        var_coord = ds["variable"].values.astype(str) if "variable" in ds.coords else None
        cov = ds["coverage_mask"].values.astype(bool)
        var = ds["variation_mask"].values.astype(bool)

    if cov.shape[1] != NPIX_HIGH:
        raise ValueError(f"masks have unexpected npix={cov.shape[1]}")

    # Use degrader's children mapping to fold OR.
    # We can't reuse the float HierarchicalDegrader directly (it does nanmean).
    # Build the same children mapping via two passes:
    import healpy as hp
    # Pass 1: 64→32
    # Pass 2: 32→16
    def or_pass(mask_arr, nside_high, nside_low):
        # mask_arr: (n_vars, npix_high)
        # For each parent at nside_low, OR over its 4 children at nside_high
        npix_high = 12 * nside_high * nside_high
        npix_low = 12 * nside_low * nside_low
        # children per parent: nside_high/nside_low = 2 → 4 children in NESTED;
        # in RING we need pix2ang then ang2pix
        # Approach: convert mask to NESTED, fold groups of 4 contiguous, then convert back
        # Simpler reliable approach: for each high pixel, get its parent at low nside
        ipix_high = np.arange(npix_high)
        theta, phi = hp.pix2ang(nside_high, ipix_high, nest=False)
        parent_idx = hp.ang2pix(nside_low, theta, phi, nest=False)  # (npix_high,)
        # Reduce by parent: for each var, np.bincount(parent_idx, weights=mask)
        n_vars = mask_arr.shape[0]
        out = np.zeros((n_vars, npix_low), dtype=bool)
        for j in range(n_vars):
            # bincount with weights: sum, then >0
            s = np.bincount(parent_idx, weights=mask_arr[j].astype(np.int32),
                             minlength=npix_low)
            out[j] = s > 0
        return out

    log("  degrading coverage_mask (any-of-16) …")
    cov_low = or_pass(cov, NSIDE_HIGH, NSIDE_LOW)
    log("  degrading variation_mask (any-of-16) …")
    var_low = or_pass(var, NSIDE_HIGH, NSIDE_LOW)
    log(f"  cov: {cov.sum()} → {cov_low.sum()} pix-vars active")
    log(f"  var: {var.sum()} → {var_low.sum()} pix-vars active")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    out_ds = xr.Dataset(
        data_vars={
            "coverage_mask": (("variable", "pixel"), cov_low),
            "variation_mask": (("variable", "pixel"), var_low),
        },
        coords={
            "variable": var_coord if var_coord is not None
                         else np.arange(cov.shape[0]),
            "pixel": np.arange(NPIX_LOW, dtype=np.int32),
        },
        attrs={
            "downsample_source_nside": NSIDE_HIGH,
            "downsample_target_nside": NSIDE_LOW,
            "downsample_method": "OR over 16 children (any covered → covered)",
            "downsample_date": datetime.utcnow().isoformat() + "Z",
        },
    )
    out_ds.to_netcdf(dst_path)
    log(f"  → {dst_path}")


# ---------------------------------------------------------------------------
def degrade_climatology(src_dir: Path, dst_dir: Path,
                         degrader: HierarchicalDegrader, log) -> None:
    """Degrade both mean and std files. Each var has shape (366, 4, 49152);
    we apply the spatial degrader pixel-wise."""
    log(f"==== climatology ====")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("mean", "std"):
        fname = f"climatology_doy_hour_{kind}_nside{NSIDE_HIGH}_v3.nc"
        src_path = src_dir / fname
        if not src_path.exists():
            log(f"  [skip] not found: {src_path}")
            continue
        # New filename uses target nside in place of source nside
        dst_path = dst_dir / fname.replace(f"nside{NSIDE_HIGH}",
                                            f"nside{NSIDE_LOW}")
        log(f"  {src_path.name} → {dst_path.name}")
        if dst_path.exists():
            dst_path.unlink()

        # Use netCDF4 directly for reading (xarray was caching loaded
        # variables, accumulating ~64 GB across 235 vars and OOMing).
        import gc
        src_nc = nc.Dataset(src_path, "r")
        data_vars = [v for v in src_nc.variables
                      if v not in ("dayofyear", "hour", "pixel")]
        n_doy = src_nc.dimensions["dayofyear"].size
        n_hour = src_nc.dimensions["hour"].size
        src_npix = src_nc.dimensions["pixel"].size
        if src_npix != NPIX_HIGH:
            raise ValueError(f"unexpected clim pixel dim {src_npix}")
        t_clim = time.time()
        out_ds = nc.Dataset(dst_path, "w", format="NETCDF4")
        out_ds.createDimension("dayofyear", n_doy)
        out_ds.createDimension("hour", n_hour)
        out_ds.createDimension("pixel", NPIX_LOW)
        doy_v = out_ds.createVariable("dayofyear", "i4", ("dayofyear",))
        doy_v[:] = src_nc.variables["dayofyear"][:]
        hour_v = out_ds.createVariable("hour", "i4", ("hour",))
        hour_v[:] = src_nc.variables["hour"][:]
        try:
            hour_v.units = src_nc.variables["hour"].units
        except Exception:
            pass
        pix_v = out_ds.createVariable("pixel", "i4", ("pixel",))
        pix_v[:] = np.arange(NPIX_LOW, dtype=np.int32)
        # Copy global attrs and update
        for k in src_nc.ncattrs():
            try:
                out_ds.setncattr(k, src_nc.getncattr(k))
            except Exception:
                pass
        out_ds.setncattr("downsample_source_nside", NSIDE_HIGH)
        out_ds.setncattr("downsample_target_nside", NSIDE_LOW)
        out_ds.setncattr("downsample_method",
                          "hierarchical nanmean (4 children per step, ×2)")
        out_ds.setncattr("downsample_date",
                          datetime.utcnow().isoformat() + "Z")
        out_ds.setncattr("healpix_nside", NSIDE_LOW)
        out_ds.setncattr("healpix_npix", NPIX_LOW)
        # Create + fill each variable
        out_vars = {}
        for v in data_vars:
            out_vars[v] = out_ds.createVariable(
                v, "f4", ("dayofyear", "hour", "pixel"),
                chunksizes=(n_doy, 1, NPIX_LOW), zlib=False,
            )
            out_vars[v].var_name = v
        for j, v in enumerate(data_vars):
            arr = src_nc.variables[v][:]  # (366, 4, 49152) float32 — fresh, no cache
            if hasattr(arr, "filled"):
                arr = arr.filled(np.nan)
            low = degrader(np.asarray(arr, dtype=np.float32),
                            pix_axis=-1)  # (366, 4, 3072)
            out_vars[v][:, :, :] = low
            del arr, low
            if (j + 1) % 50 == 0 or j == len(data_vars) - 1:
                # Periodic GC to ensure netCDF4 var buffers are released
                gc.collect()
                out_ds.sync()
                log(f"    [{fname[:35]}] {j+1}/{len(data_vars)} vars "
                    f"({time.time()-t_clim:.0f}s)")
        out_ds.sync()
        out_ds.close()
        src_nc.close()
        log(f"  done: {dst_path} ({dst_path.stat().st_size/1e9:.2f} GB)")


# ---------------------------------------------------------------------------
def main() -> None:
    # Declared up front because we also reference these names while
    # building the argparse defaults below.
    global NSIDE_HIGH, NSIDE_LOW, NPIX_HIGH, NPIX_LOW
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3/normalized"))
    p.add_argument("--dst-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n16/normalized"))
    p.add_argument("--clim-src", type=Path,
                   default=Path("/scratch/local/monan_ia_v3/climatology"))
    p.add_argument("--clim-dst", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n16/climatology"))
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="HEALPix children-mapping cache "
                        "(default: <dst-dir>/_cache)")
    p.add_argument("--batch", type=int, default=200,
                   help="Timesteps per memmap read batch")
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                   help="Which splits to process; pass an empty list "
                        "(--splits) to skip all memmap degrades.")
    p.add_argument("--skip-masks", action="store_true")
    p.add_argument("--skip-clim", action="store_true")
    p.add_argument("--skip-stats-copy", action="store_true")
    p.add_argument("--nside-high", type=int, default=NSIDE_HIGH,
                   help="Source HEALPix nside (default: 64)")
    p.add_argument("--nside-low", type=int, default=NSIDE_LOW,
                   help="Target HEALPix nside (default: 16)")
    args = p.parse_args()

    # Override module-level constants from CLI so all helper functions
    # below pick up the user-specified resolutions.
    NSIDE_HIGH = int(args.nside_high)
    NSIDE_LOW = int(args.nside_low)
    NPIX_HIGH = 12 * NSIDE_HIGH * NSIDE_HIGH
    NPIX_LOW = 12 * NSIDE_LOW * NSIDE_LOW
    if NSIDE_HIGH <= NSIDE_LOW:
        raise SystemExit(f"nside-high ({NSIDE_HIGH}) must exceed "
                         f"nside-low ({NSIDE_LOW})")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger().info

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.dst_dir.parent / "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("08_degrade_normalized_n64_to_n16")
    log(f"  src     : {args.src_dir}")
    log(f"  dst     : {args.dst_dir}")
    log(f"  clim_src: {args.clim_src}")
    log(f"  clim_dst: {args.clim_dst}")
    log(f"  batch   : {args.batch}")
    log(f"  splits  : {args.splits}")
    log("=" * 72)

    log("Building HEALPix degrader nside=64→16 (cached) …")
    t0 = time.time()
    degrader = HierarchicalDegrader(
        nside_high=NSIDE_HIGH, nside_low=NSIDE_LOW, nest=False,
        cache_dir=cache_dir,
    )
    log(f"  degrader ready in {time.time()-t0:.1f}s "
        f"({len(degrader.mappings)} passes: "
        f"{[m.shape for m in degrader.mappings]})")

    # Splits
    for split in args.splits:
        degrade_memmap_split(args.src_dir, args.dst_dir, split, degrader,
                              args.batch, log)

    # Stats: copy as-is (single normalization system across resolutions)
    if not args.skip_stats_copy:
        src_stats = args.src_dir / "stats.json"
        if src_stats.exists():
            shutil.copy2(src_stats, args.dst_dir / "stats.json")
            log(f"copied stats.json (passed through unchanged)")

    # Masks: OR over 16 children
    if not args.skip_masks:
        src_masks = args.src_dir / "masks.nc"
        if src_masks.exists():
            degrade_masks(src_masks, args.dst_dir / "masks.nc", log)

    # Climatology: nanmean over 16 children
    if not args.skip_clim:
        degrade_climatology(args.clim_src, args.clim_dst, degrader, log)

    log("=" * 72)
    log("ALL DONE")
    log("=" * 72)


if __name__ == "__main__":
    main()
