#!/usr/bin/env python3
"""Hierarchical upsample n4 → n8 (parent broadcast) + residual = n8 − upsample.

The HEALPix nested hierarchy gives every nside=8 pixel exactly one parent
at nside=4 (4 children per parent). Upsampling here is the natural
piecewise-constant lift: every child inherits its parent's value.

Two outputs are produced (both at nside=8 pixel count = 768):

  1. UPSAMPLED  = parent_broadcast(n4)            ("coarse prior")
  2. RESIDUAL   = n8_observed − upsampled         ("high-frequency target")

By construction the residual averages to zero over each quartet of
children, since conservative_downsample(upsample(x)) = x. This means a
cascaded model can predict UPSAMPLED with the n4-level model and add a
high-frequency RESIDUAL prediction from a finer-scale network.

Inputs:
    --n4-dir   /scratch/local/monan_ia_v3_n4/normalized
    --n8-dir   /scratch/local/monan_ia_v3_n8/normalized
    --n4-clim  /scratch/local/monan_ia_v3_n4/climatology
    --n8-clim  /scratch/local/monan_ia_v3_n8/climatology

Outputs (mirror layout):
    --up-dir       upsampled normalized memmaps (same shape as n8)
    --up-clim-dir  upsampled climatology
    --res-dir      residual normalized memmaps (same shape as n8)
    --res-clim-dir residual climatology
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

import healpy as hp
import netCDF4 as nc
import numpy as np


# Defaults; overridden by CLI in main().
NSIDE_LOW = 4    # source ("coarse")
NSIDE_HIGH = 8   # target ("fine")
NPIX_LOW = 12 * NSIDE_LOW * NSIDE_LOW
NPIX_HIGH = 12 * NSIDE_HIGH * NSIDE_HIGH


def build_parent_index(nside_high: int, nside_low: int, nest: bool = False) -> np.ndarray:
    """For each pixel at nside_high (RING), return its parent index at nside_low (RING)."""
    ipix = np.arange(12 * nside_high * nside_high)
    theta, phi = hp.pix2ang(nside_high, ipix, nest=nest)
    parent = hp.ang2pix(nside_low, theta, phi, nest=nest)
    return parent.astype(np.int32)


def upsample_array(arr_low: np.ndarray, parent_idx: np.ndarray,
                    pix_axis: int) -> np.ndarray:
    """Broadcast low-res values to high-res via fancy indexing.

    arr_low shape (..., NPIX_LOW, ...). Returns (..., NPIX_HIGH, ...) where
    each output[..., i, ...] = arr_low[..., parent_idx[i], ...].
    """
    return np.take(arr_low, parent_idx, axis=pix_axis)


# ---------------------------------------------------------------------------
def process_split(split: str, n4_dir: Path, n8_dir: Path,
                   up_dir: Path, res_dir: Path,
                   parent_idx: np.ndarray, batch: int, log) -> None:
    log(f"==== split={split} ====")
    n4_arr_path = n4_dir / f"{split}.npy"
    n8_arr_path = n8_dir / f"{split}.npy"
    if not n4_arr_path.exists():
        log(f"  [{split}] n4 source missing, skip: {n4_arr_path}")
        return
    if not n8_arr_path.exists():
        log(f"  [{split}] n8 source missing, skip: {n8_arr_path}")
        return

    n4 = np.load(n4_arr_path, mmap_mode="r")
    n8 = np.load(n8_arr_path, mmap_mode="r")
    if n4.shape[1] != NPIX_LOW:
        raise ValueError(f"n4 npix {n4.shape[1]} != {NPIX_LOW}")
    if n8.shape[1] != NPIX_HIGH:
        raise ValueError(f"n8 npix {n8.shape[1]} != {NPIX_HIGH}")
    if n4.shape[0] != n8.shape[0]:
        raise ValueError(f"n4/n8 n_steps mismatch: {n4.shape[0]} vs {n8.shape[0]}")
    if n4.shape[2] != n8.shape[2]:
        raise ValueError(f"n4/n8 n_vars mismatch: {n4.shape[2]} vs {n8.shape[2]}")
    n_steps, _, n_vars = n8.shape
    log(f"  n4 shape={n4.shape}, n8 shape={n8.shape}")

    up_path = up_dir / f"{split}.npy"
    res_path = res_dir / f"{split}.npy"
    up_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (up_path, res_path):
        if p.exists():
            p.unlink()
    log(f"  pre-allocating up: {up_path} ({n_steps}, {NPIX_HIGH}, {n_vars}) {n8.dtype}")
    up_mm = np.lib.format.open_memmap(
        up_path, mode="w+", dtype=n8.dtype,
        shape=(n_steps, NPIX_HIGH, n_vars),
    )
    log(f"  pre-allocating res: {res_path}")
    res_mm = np.lib.format.open_memmap(
        res_path, mode="w+", dtype=n8.dtype,
        shape=(n_steps, NPIX_HIGH, n_vars),
    )
    est_gb = n_steps * NPIX_HIGH * n_vars * n8.dtype.itemsize / 1e9
    log(f"  estimated each: {est_gb:.2f} GB")

    t0 = time.time()
    for start in range(0, n_steps, batch):
        end = min(start + batch, n_steps)
        coarse = n4[start:end].astype(np.float32)       # (B, 192, V)
        observed = n8[start:end].astype(np.float32)     # (B, 768, V)
        up = upsample_array(coarse, parent_idx, pix_axis=1)  # (B, 768, V)
        res = observed - up
        up_mm[start:end] = up.astype(n8.dtype)
        res_mm[start:end] = res.astype(n8.dtype)
        if (start // batch) % 50 == 0 or end == n_steps:
            log(f"    [{split}] {end}/{n_steps} "
                f"({time.time()-t0:.0f}s, {(end/n_steps)*100:.1f}%)")
    del up_mm, res_mm

    # Times + index (both up and res share the n8 time axis)
    times = np.load(n8_dir / f"{split}_times.npy")
    np.save(up_dir / f"{split}_times.npy", times)
    np.save(res_dir / f"{split}_times.npy", times)
    with open(n8_dir / f"{split}_index.json") as f:
        n8_index = json.load(f)

    common_meta = {
        "src_low_nside": NSIDE_LOW,
        "src_high_nside": NSIDE_HIGH,
        "src_low_npix": NPIX_LOW,
        "src_high_npix": NPIX_HIGH,
        "date": datetime.utcnow().isoformat() + "Z",
    }
    for tag, out_dir, role, meta_extra in [
        ("up", up_dir, "upsample (parent broadcast)",
         {"note": "Piecewise-constant lift of nside=4 onto nside=8 children."}),
        ("res", res_dir, "residual = n8 − upsample(n4)",
         {"note": "By construction: mean over each quartet of children = 0; "
                  "captures the high-frequency complement of the n4 coarse field."}),
    ]:
        new_index = dict(n8_index)
        new_index["resampling"] = {**common_meta, "role": role, **meta_extra}
        with open(out_dir / f"{split}_index.json", "w") as f:
            json.dump(new_index, f, indent=2)

    log(f"  [{split}] DONE in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
def process_clim(n4_clim_dir: Path, n8_clim_dir: Path,
                  up_clim_dir: Path, res_clim_dir: Path,
                  parent_idx: np.ndarray, log) -> None:
    import gc
    up_clim_dir.mkdir(parents=True, exist_ok=True)
    res_clim_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("mean", "std"):
        n4_path = n4_clim_dir / f"climatology_doy_hour_{kind}_nside{NSIDE_LOW}_v3.nc"
        n8_path = n8_clim_dir / f"climatology_doy_hour_{kind}_nside{NSIDE_HIGH}_v3.nc"
        if not (n4_path.exists() and n8_path.exists()):
            log(f"  [clim {kind}] missing source, skip "
                f"(n4={n4_path.exists()}, n8={n8_path.exists()})")
            continue
        up_path = up_clim_dir / f"climatology_doy_hour_{kind}_nside{NSIDE_HIGH}_v3.nc"
        res_path = res_clim_dir / f"climatology_doy_hour_{kind}_nside{NSIDE_HIGH}_v3.nc"
        log(f"  [clim {kind}] {n4_path.name} + {n8_path.name} → up + res")
        for p in (up_path, res_path):
            if p.exists():
                p.unlink()

        nc_n4 = nc.Dataset(n4_path, "r")
        nc_n8 = nc.Dataset(n8_path, "r")
        data_vars = [v for v in nc_n4.variables
                      if v not in ("dayofyear", "hour", "pixel")]
        n_doy = nc_n4.dimensions["dayofyear"].size
        n_hour = nc_n4.dimensions["hour"].size

        def _create(out_path):
            out_ds = nc.Dataset(out_path, "w", format="NETCDF4")
            out_ds.createDimension("dayofyear", n_doy)
            out_ds.createDimension("hour", n_hour)
            out_ds.createDimension("pixel", NPIX_HIGH)
            v = out_ds.createVariable("dayofyear", "i4", ("dayofyear",))
            v[:] = nc_n4.variables["dayofyear"][:]
            v = out_ds.createVariable("hour", "i4", ("hour",))
            v[:] = nc_n4.variables["hour"][:]
            try:
                v.units = nc_n4.variables["hour"].units
            except Exception:
                pass
            v = out_ds.createVariable("pixel", "i4", ("pixel",))
            v[:] = np.arange(NPIX_HIGH, dtype=np.int32)
            for k in nc_n4.ncattrs():
                try:
                    out_ds.setncattr(k, nc_n4.getncattr(k))
                except Exception:
                    pass
            out_ds.setncattr("healpix_nside", NSIDE_HIGH)
            out_ds.setncattr("healpix_npix", NPIX_HIGH)
            out_ds.setncattr("source_low_nside", NSIDE_LOW)
            out_ds.setncattr("creation_date", datetime.utcnow().isoformat() + "Z")
            return out_ds

        up_ds = _create(up_path)
        up_ds.setncattr("derivation", "parent broadcast (upsample n4→n8)")
        res_ds = _create(res_path)
        res_ds.setncattr("derivation", "residual = clim_n8 − upsample(clim_n4)")

        up_vars = {}
        res_vars = {}
        for v in data_vars:
            up_vars[v] = up_ds.createVariable(
                v, "f4", ("dayofyear", "hour", "pixel"),
                chunksizes=(n_doy, 1, NPIX_HIGH), zlib=False,
            )
            res_vars[v] = res_ds.createVariable(
                v, "f4", ("dayofyear", "hour", "pixel"),
                chunksizes=(n_doy, 1, NPIX_HIGH), zlib=False,
            )

        t0 = time.time()
        for j, v in enumerate(data_vars):
            coarse = np.asarray(nc_n4.variables[v][:], dtype=np.float32)   # (366,4,192)
            observed = np.asarray(nc_n8.variables[v][:], dtype=np.float32)  # (366,4,768)
            up = np.take(coarse, parent_idx, axis=-1)                        # (366,4,768)
            res = observed - up
            up_vars[v][:, :, :] = up
            res_vars[v][:, :, :] = res
            del coarse, observed, up, res
            if (j + 1) % 50 == 0 or j == len(data_vars) - 1:
                gc.collect()
                up_ds.sync()
                res_ds.sync()
                log(f"    [clim {kind}] {j+1}/{len(data_vars)} vars "
                    f"({time.time()-t0:.0f}s)")

        up_ds.sync(); up_ds.close()
        res_ds.sync(); res_ds.close()
        nc_n4.close(); nc_n8.close()
        log(f"  done: {up_path.name} ({up_path.stat().st_size/1e6:.1f} MB), "
            f"{res_path.name} ({res_path.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
def main() -> None:
    # Top of function so default= expressions below can read the globals.
    global NSIDE_LOW, NSIDE_HIGH, NPIX_LOW, NPIX_HIGH
    p = argparse.ArgumentParser()
    p.add_argument("--n4-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n4/normalized"))
    p.add_argument("--n8-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8/normalized"))
    p.add_argument("--up-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8_from_n4/normalized"))
    p.add_argument("--res-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8_residual/normalized"))
    p.add_argument("--n4-clim", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n4/climatology"))
    p.add_argument("--n8-clim", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8/climatology"))
    p.add_argument("--up-clim-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8_from_n4/climatology"))
    p.add_argument("--res-clim-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8_residual/climatology"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    p.add_argument("--skip-clim", action="store_true")
    p.add_argument("--nside-low", type=int, default=NSIDE_LOW,
                   help="Source (coarse) HEALPix nside (default: 4)")
    p.add_argument("--nside-high", type=int, default=NSIDE_HIGH,
                   help="Target (fine) HEALPix nside (default: 8)")
    args = p.parse_args()

    NSIDE_LOW = int(args.nside_low)
    NSIDE_HIGH = int(args.nside_high)
    NPIX_LOW = 12 * NSIDE_LOW * NSIDE_LOW
    NPIX_HIGH = 12 * NSIDE_HIGH * NSIDE_HIGH
    if NSIDE_HIGH <= NSIDE_LOW:
        raise SystemExit(f"nside-high ({NSIDE_HIGH}) must exceed "
                         f"nside-low ({NSIDE_LOW})")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger().info

    args.up_dir.mkdir(parents=True, exist_ok=True)
    args.res_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("09_upsample_n4_to_n8_residual")
    log(f"  n4_dir  : {args.n4_dir}")
    log(f"  n8_dir  : {args.n8_dir}")
    log(f"  up_dir  : {args.up_dir}")
    log(f"  res_dir : {args.res_dir}")
    log(f"  batch   : {args.batch}")
    log(f"  splits  : {args.splits}")
    log("=" * 72)

    parent_idx = build_parent_index(NSIDE_HIGH, NSIDE_LOW, nest=False)
    log(f"  parent index built (NPIX_HIGH={len(parent_idx)} → max parent "
        f"= {parent_idx.max()}/{NPIX_LOW - 1})")

    # Process splits
    for split in args.splits:
        process_split(split, args.n4_dir, args.n8_dir,
                       args.up_dir, args.res_dir, parent_idx,
                       args.batch, log)

    # Stats/masks: keep n8 stats verbatim (single normalization system),
    # copy n8 masks to both (high-res mask region is unchanged).
    for tag, out_dir in [("up", args.up_dir), ("res", args.res_dir)]:
        for fname in ("stats.json", "masks.nc"):
            src = args.n8_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
                log(f"copied {fname} → {tag}/")

    # Climatology
    if not args.skip_clim:
        process_clim(args.n4_clim, args.n8_clim,
                      args.up_clim_dir, args.res_clim_dir,
                      parent_idx, log)

    log("=" * 72)
    log("ALL DONE")
    log("=" * 72)


if __name__ == "__main__":
    main()
