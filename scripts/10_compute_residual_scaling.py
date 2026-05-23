#!/usr/bin/env python3
"""Compute per-variable scaling for the residual dataset.

The residual = n8_obs − upsample(n4) already has mean ≈ 0 per quartet by
construction. What varies wildly across variables is its scale: some
fields (precipitation, divergence) have residuals nearly as large as the
full signal, while smooth large-scale fields (geopotential, MSLP) have
tiny residuals. Training a single model on this mix is unstable unless
each channel is brought back to roughly unit variance.

This script computes a single scalar per variable:

    scaling_std[v] = sqrt( mean( residual[:, masked_pix, v]^2 ) )

(restricted to variation_mask pixels so that always-zero pixels do not
deflate the estimate). At training time:

    residual_scaled  = residual  / scaling_std
    residual_back    = scaled    * scaling_std

mean is also recorded for verification (it should be ~0 everywhere).

Output: <residual-dir>/residual_scaling.json
    {
      "per_var": {var: {"std": ..., "mean": ..., "n_obs": ...}, ...},
      "metadata": { ... }
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--residual-dir", type=Path,
                   default=Path("/scratch/local/monan_ia_v3_n8_residual/normalized"))
    p.add_argument("--split", default="train",
                   help="Which split to base the scaling on (default: train)")
    p.add_argument("--batch", type=int, default=256,
                   help="Timesteps per memmap read batch")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default: <residual-dir>/residual_scaling.json)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger().info

    out_path = args.output or (args.residual_dir / "residual_scaling.json")

    log("=" * 72)
    log("10_compute_residual_scaling")
    log(f"  residual_dir: {args.residual_dir}")
    log(f"  split       : {args.split}")
    log(f"  batch       : {args.batch}")
    log(f"  output      : {out_path}")
    log("=" * 72)

    arr_path = args.residual_dir / f"{args.split}.npy"
    times_path = args.residual_dir / f"{args.split}_times.npy"
    index_path = args.residual_dir / f"{args.split}_index.json"
    masks_path = args.residual_dir / "masks.nc"

    arr = np.load(arr_path, mmap_mode="r")
    times = np.load(times_path)
    with open(index_path) as f:
        ix = json.load(f)
    var_list = ix["var_list"]
    n_steps, npix, n_vars = arr.shape
    log(f"  memmap shape={arr.shape} dtype={arr.dtype}")
    log(f"  time range: {times[0]} .. {times[-1]}")

    with xr.open_dataset(masks_path) as md:
        if "variable" in md.coords:
            mask_vars = list(md["variable"].values.astype(str))
            order = [mask_vars.index(v) for v in var_list]
            vmask = md["variation_mask"].values.astype(bool)[order]
        else:
            vmask = md["variation_mask"].values.astype(bool)
    if vmask.shape != (n_vars, npix):
        raise SystemExit(f"variation_mask shape {vmask.shape} != ({n_vars}, {npix})")
    n_pix_per_var = vmask.sum(axis=1).astype(np.int64)  # (n_vars,)
    log(f"  vars with at least 1 masked pixel: "
        f"{int((n_pix_per_var > 0).sum())}/{n_vars}")

    vmask_T_f32 = vmask.T.astype(np.float32)  # (npix, n_vars)

    sum_per_var = np.zeros(n_vars, dtype=np.float64)
    sumsq_per_var = np.zeros(n_vars, dtype=np.float64)
    count_per_var = np.zeros(n_vars, dtype=np.int64)

    t0 = time.time()
    for start in range(0, n_steps, args.batch):
        end = min(start + args.batch, n_steps)
        block = arr[start:end].astype(np.float32)   # (B, npix, n_vars)
        b = end - start
        masked = block * vmask_T_f32[None]          # (B, npix, n_vars)
        sum_per_var += masked.sum(axis=(0, 1))
        sumsq_per_var += (masked * masked).sum(axis=(0, 1))
        count_per_var += b * n_pix_per_var
        if (start // args.batch) % 50 == 0 or end == n_steps:
            log(f"    {end}/{n_steps} timesteps ({time.time()-t0:.0f}s, "
                f"{(end/n_steps)*100:.1f}%)")

    log(f"  accumulation done in {time.time()-t0:.0f}s")

    # mean and std per var, safe for empty masks
    safe_count = np.where(count_per_var > 0, count_per_var, 1)
    mean_per_var = (sum_per_var / safe_count).astype(np.float64)
    var_per_var = (sumsq_per_var / safe_count - mean_per_var ** 2)
    var_per_var = np.maximum(var_per_var, 0.0)
    std_per_var = np.sqrt(var_per_var)

    # For vars with no masked pixels, scaling = 1.0 (no-op)
    std_per_var = np.where(count_per_var > 0, std_per_var, 1.0)

    # Sanity numbers
    nz = std_per_var > 0
    log(f"  std summary across {n_vars} vars: "
        f"min={std_per_var.min():.5f}, "
        f"median={np.median(std_per_var):.5f}, "
        f"max={std_per_var.max():.5f}")
    log(f"  |mean| summary (should be ~0):     "
        f"max={np.abs(mean_per_var).max():.5f}")

    output = {
        "metadata": {
            "split": args.split,
            "method": "global RMS over variation_mask pixels of residual",
            "note": (
                "Use as: residual_scaled = residual / per_var.std. "
                "Inverse: residual = residual_scaled * per_var.std. "
                "mean is recorded for verification only; "
                "by-construction the residual is mean-zero over each "
                "HEALPix nside=4 quartet."
            ),
            "n_timesteps_used": int(n_steps),
            "n_pixels": int(npix),
            "n_variables": int(n_vars),
            "date": datetime.utcnow().isoformat() + "Z",
            "source_memmap": str(arr_path),
            "source_masks": str(masks_path),
        },
        "per_var": {
            var_list[j]: {
                "std": float(std_per_var[j]),
                "mean": float(mean_per_var[j]),
                "n_obs": int(count_per_var[j]),
                "n_pixels_in_mask": int(n_pix_per_var[j]),
            }
            for j in range(n_vars)
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"  wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

    # Show top/bottom-5 vars by scaling
    pairs = sorted(((var_list[j], float(std_per_var[j])) for j in range(n_vars)),
                    key=lambda x: x[1])
    log("  --- 5 smallest scalings (vars with tiny residuals): ---")
    for v, s in pairs[:5]:
        log(f"    {v:<20} std={s:.5f}")
    log("  --- 5 largest scalings (vars with big residuals):   ---")
    for v, s in pairs[-5:]:
        log(f"    {v:<20} std={s:.5f}")
    log("=" * 72)
    log("DONE")
    log("=" * 72)


if __name__ == "__main__":
    main()
