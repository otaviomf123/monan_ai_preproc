#!/usr/bin/env python3
"""Compute v3 dataset statistics + dual-mask + per-variable transforms.

Three phases (note new ORDER: B → A → C):

  B. **Per-pixel Welford → DUAL masks** (multiprocess)
     Each worker streams its variable chunk across all files. Per-pixel:
        - count of finite values across timesteps
        - mean (Welford)
        - std (Welford)
     Two masks built from this per variable:
        coverage_mask = 1 where pixel had ANY finite value (variable is
                          geographically defined here).  Used during
                          DENORMALIZE to restore NaN where variable is not
                          defined (e.g., SST over land).
        variation_mask = 1 where std > abs_thr OR cv > rel_thr (pixel
                          has meaningful variation across time).  Used
                          for STATS computation (avoids mean/std being
                          polluted by structural zeros, e.g., snow in
                          tropics, cloud-water at 10 hPa).

  A. **Transform discovery on masked samples**
     For each variable, sample N pixels from variation_mask=1 only, find
     optimal transform:
        - Try signed_power 1.0, 1/2, ..., 1/8; pick smallest power that
          achieves max|z| ≤ target.
        - If none works (sparse / structural zeros), fall back to asinh
          with scale = 90th percentile of |non-zero samples|. Asinh tames
          extreme outliers logarithmically without amplifying zeros.
        - Last resort: identity.

  C. **Global stats** (multiprocess)
     Each worker accumulates global Welford (mean, std) over masked pixels
     × timesteps, with the chosen transform applied. Stats are computed
     ONLY on variation_mask=1 pixels.

Output:
    {out_dir}/healpix_statistics_v3.json    -- per-variable stats + transform
    {out_dir}/healpix_masks_v3.nc           -- both masks per variable
    {out_dir}/compute_stats.log

NaN handling policy:
    Forward (normalize): NaN → 0 in z-scored output (model adapts).
    Inverse (denormalize): use coverage_mask to restore NaN where the
        variable is not geographically defined; pixels inside coverage
        but outside variation are recoverable as 0 (or near-0) physical
        values via the transform inverse.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monan_preproc.stats import (
    GlobalWelford,
    IncrementalStatistics,
    apply_transform,
    build_mask_from_per_pixel_stats,
    find_optimal_transform,
)


# ---------------------------------------------------------------------------
def discover_files(data_dir: Path,
                   start_year: int | None = None,
                   end_year: int | None = None) -> list[Path]:
    files: list[Path] = []
    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith("_"):
            continue
        try:
            y = int(year_dir.name)
        except ValueError:
            continue
        if start_year is not None and y < start_year:
            continue
        if end_year is not None and y > end_year:
            continue
        for f in sorted(year_dir.glob(f"{y}_*.nc")):
            files.append(f)
    return files


def get_var_list_npix(sample_file: Path) -> tuple[list[str], int]:
    with xr.open_dataset(sample_file) as ds:
        vars_ = sorted(ds.data_vars)
        npix = int(ds.sizes["pixel"])
    return vars_, npix


def chunk_vars_balanced(var_list: list[str], n_workers: int) -> list[list[str]]:
    chunks: list[list[str]] = [[] for _ in range(n_workers)]
    for i, v in enumerate(var_list):
        chunks[i % n_workers].append(v)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------
def _worker_per_pixel_stats(args):
    """Per-pixel Welford for a chunk of variables across all files (raw values)."""
    var_chunk, files, npix = args
    accs = {v: IncrementalStatistics(npix) for v in var_chunk}
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                for v in var_chunk:
                    if v not in ds.data_vars:
                        continue
                    arr = ds[v].values.astype(np.float64)
                    accs[v].update_batch(arr)
        except Exception as e:
            print(f"WARN per-pixel {f.name}: {e}", file=sys.stderr)
    return accs


def _worker_sample_for_transforms(args):
    """Sample finite values from variation_mask=1 pixels, per variable.

    Returns dict {var: 1D array of samples drawn from valid pixels only}.
    """
    var_chunk, files, variation_masks, n_per_file, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    samples: dict[str, list[np.ndarray]] = {v: [] for v in var_chunk}
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                for v in var_chunk:
                    if v not in ds.data_vars:
                        continue
                    mask = variation_masks[v].astype(bool)
                    if not mask.any():
                        continue
                    arr = ds[v].values  # (n_t, npix)
                    # restrict to mask=1 pixels then flatten + filter NaN
                    sub = arr[:, mask].ravel()
                    finite = sub[np.isfinite(sub)]
                    if finite.size == 0:
                        continue
                    n = min(n_per_file, finite.size)
                    if finite.size > n_per_file:
                        idx = rng.choice(finite.size, n, replace=False)
                        samples[v].append(finite[idx])
                    else:
                        samples[v].append(finite)
        except Exception as e:
            print(f"WARN sample {f.name}: {e}", file=sys.stderr)
    return {v: np.concatenate(s) for v, s in samples.items() if s}


def _worker_global_stats(args):
    """Global Welford on variation_mask=1 pixels with chosen transform applied."""
    var_chunk, files, variation_masks, transforms = args
    accs = {v: GlobalWelford() for v in var_chunk}
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                for v in var_chunk:
                    if v not in ds.data_vars:
                        continue
                    arr = ds[v].values.astype(np.float64)
                    arr_t = apply_transform(arr, transforms[v])
                    accs[v].update_masked(arr_t, variation_masks[v].astype(bool))
        except Exception as e:
            print(f"WARN global {f.name}: {e}", file=sys.stderr)
    return accs


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_b_per_pixel_and_masks(
    files: list[Path], var_list: list[str], npix: int,
    abs_std_thr: float, rel_std_thr: float,
    n_workers: int, log,
) -> tuple[dict, dict, dict]:
    """Phase B: per-pixel Welford → coverage + variation masks.

    Returns:
        coverage_mask : dict var -> (npix,) float32 (1.0 where any finite obs exists)
        variation_mask: dict var -> (npix,) float32 (1.0 where std passes thresholds)
        n_valid       : dict var -> int (count of variation pixels)
    """
    log("=== Phase B: per-pixel Welford → dual masks ===")
    chunks = chunk_vars_balanced(var_list, n_workers)
    log(f"  {len(chunks)} workers, ~{(len(var_list)+len(chunks)-1)//len(chunks)} vars each, "
        f"{len(files)} files …")
    args = [(c, files, npix) for c in chunks]
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_per_pixel_stats, args)
    log(f"  per-pixel pass done in {time.time()-t0:.1f}s")

    per_pixel: dict[str, IncrementalStatistics] = {}
    for d in results:
        per_pixel.update(d)

    coverage = {}
    variation = {}
    n_valid = {}
    for v in var_list:
        if v not in per_pixel:
            coverage[v] = np.zeros(npix, dtype=np.float32)
            variation[v] = np.zeros(npix, dtype=np.float32)
            n_valid[v] = 0
            continue
        # COVERAGE: pixel had at least one finite observation across timesteps
        cov = (per_pixel[v].count >= 1).astype(np.float32)
        coverage[v] = cov
        # VARIATION: pixel has std-based variation
        m, s = per_pixel[v].get_mean_std()
        var_mask = build_mask_from_per_pixel_stats(m, s, abs_std_thr, rel_std_thr)
        variation[v] = var_mask
        n_valid[v] = int((var_mask > 0.5).sum())

    # Diagnostic summary
    cov_counts = sorted(int((c > 0.5).sum()) for c in coverage.values())
    var_counts = sorted(n_valid.values())
    log(f"  coverage_mask : min={cov_counts[0]}, "
        f"median={cov_counts[len(cov_counts)//2]}, max={cov_counts[-1]}")
    log(f"  variation_mask: min={var_counts[0]}, "
        f"median={var_counts[len(var_counts)//2]}, max={var_counts[-1]}")
    return coverage, variation, n_valid


def phase_a_transforms(
    files: list[Path], var_list: list[str],
    variation_masks: dict, max_z: float,
    n_test_files: int, n_per_file: int,
    n_workers: int, log,
) -> dict:
    """Phase A: discover optimal transform per variable using masked samples."""
    log("=== Phase A: transform discovery (masked samples) ===")
    rng = np.random.default_rng(42)
    pick = rng.choice(len(files), min(n_test_files, len(files)), replace=False)
    test_files = [files[i] for i in sorted(pick)]
    log(f"  sampling up to {n_per_file} pts/var from variation pixels in "
        f"{len(test_files)} files with {n_workers} workers …")
    chunks = chunk_vars_balanced(var_list, n_workers)
    args = [
        (c, test_files,
         {v: variation_masks[v] for v in c},
         n_per_file, 1000 + i)
        for i, c in enumerate(chunks)
    ]
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_sample_for_transforms, args)
    log(f"  sampling done in {time.time()-t0:.1f}s")

    samples: dict[str, np.ndarray] = {}
    for r in results:
        samples.update(r)

    transforms: dict[str, dict] = {}
    for v in var_list:
        if v not in samples or samples[v].size < 100:
            transforms[v] = {"type": "identity", "achieved_max_z": float("nan")}
        else:
            transforms[v] = find_optimal_transform(samples[v], max_z=max_z)

    n_pow = sum(1 for t in transforms.values() if t["type"] == "power")
    n_asinh = sum(1 for t in transforms.values() if t["type"] == "asinh")
    n_id = sum(1 for t in transforms.values() if t["type"] == "identity")
    n_meet = sum(
        1 for v in var_list
        if transforms[v]["achieved_max_z"] <= max_z
        or np.isnan(transforms[v]["achieved_max_z"])
    )
    log(f"  transforms chosen: {n_pow} power, {n_asinh} asinh, {n_id} identity")
    log(f"  {n_meet}/{len(var_list)} achieve max|z|≤{max_z}")

    # Show worst offenders
    worst = sorted(
        ((v, transforms[v]["achieved_max_z"], transforms[v]["type"])
         for v in var_list
         if not np.isnan(transforms[v]["achieved_max_z"])),
        key=lambda x: -x[1],
    )[:10]
    if worst and worst[0][1] > max_z:
        log(f"  worst residual max|z| (after best transform):")
        for v, z, t in worst:
            if z > max_z:
                log(f"    {v:<22} type={t:<8} max|z|={z:.2f}")

    return transforms


def phase_c_global(
    files: list[Path], var_list: list[str],
    variation_masks: dict, transforms: dict,
    n_workers: int, log,
) -> dict:
    """Global Welford on variation pixels with chosen transform applied."""
    log("=== Phase C: global stats (variation-masked, transformed) ===")
    chunks = chunk_vars_balanced(var_list, n_workers)
    args = [
        (c, files,
         {v: variation_masks[v] for v in c},
         {v: transforms[v] for v in c})
        for c in chunks
    ]
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_global_stats, args)
    log(f"  global pass done in {time.time()-t0:.1f}s")
    out = {}
    for d in results:
        out.update(d)
    return out


# ---------------------------------------------------------------------------
def write_masks_netcdf(out_path: Path, var_list: list[str], npix: int,
                       coverage: dict, variation: dict,
                       extra_attrs: dict) -> None:
    """Save dual masks as NetCDF4 with two 2D arrays (variable × pixel).

    Layout:
        coverage_mask : (variable, pixel) float32 in {0.0, 1.0}
        variation_mask: (variable, pixel) float32 in {0.0, 1.0}

    Use:
        - For STATS computation: variation_mask
        - For DENORMALIZATION (restoring NaN): coverage_mask (where 0 ⇒ NaN)
    """
    cov_arr = np.stack([coverage[v] for v in var_list], axis=0)         # (V, P)
    var_arr = np.stack([variation[v] for v in var_list], axis=0)        # (V, P)

    ds = xr.Dataset(
        data_vars={
            "coverage_mask": (("variable", "pixel"), cov_arr,
                              {"long_name": "Pixel had ≥1 finite observation across "
                                            "all timesteps; use to restore NaN on denorm",
                               "units": "1",
                               "valid_range": [0, 1]}),
            "variation_mask": (("variable", "pixel"), var_arr,
                               {"long_name": "Pixel passes std/cv variation threshold; "
                                             "used for stats computation",
                                "units": "1",
                                "valid_range": [0, 1]}),
        },
        coords={
            "variable": ("variable", np.array(var_list, dtype="U24")),
            "pixel": ("pixel", np.arange(npix, dtype=np.int32)),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "MONAN-IA v3 per-variable masks (coverage + variation)",
            "nside": 64,
            "npix": int(npix),
            "ordering": "RING",
            "n_variables": len(var_list),
            "package": "monan_ia_preproc 0.1.0",
            "creation_date": datetime.utcnow().isoformat() + "Z",
            **extra_attrs,
        },
    )
    enc = {
        "coverage_mask":  {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"},
        "variation_mask": {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=enc, format="NETCDF4")
    ds.close()


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/beegfs/otavio.feitosa/era5_healpix_v3")
    p.add_argument("--out-dir", default=None,
                   help="default = {data-dir}/_stats/")
    p.add_argument("--max-z", type=float, default=5.0)
    p.add_argument("--n-test-files", type=int, default=60)
    p.add_argument("--n-per-file", type=int, default=50000)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--abs-std-threshold", type=float, default=1e-3)
    p.add_argument("--rel-std-threshold", type=float, default=0.01)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "compute_stats.log"),
        ],
    )
    log = logging.getLogger().info

    log("=" * 72)
    log("MONAN-IA v3 stats + dual-masks computation (B → A → C)")
    log("=" * 72)
    log(f"  data_dir : {data_dir}")
    log(f"  out_dir  : {out_dir}")
    log(f"  max |z|  : {args.max_z}")
    log(f"  workers  : {args.workers}")

    files = discover_files(data_dir, args.start_year, args.end_year)
    if not files:
        raise SystemExit(f"No files found under {data_dir}")
    var_list, npix = get_var_list_npix(files[0])
    log(f"  files    : {len(files)}")
    log(f"  vars     : {len(var_list)}")
    log(f"  pixels   : {npix}")

    # Phase B FIRST — masks come before transform discovery
    t0 = time.time()
    coverage, variation, n_valid = phase_b_per_pixel_and_masks(
        files, var_list, npix,
        args.abs_std_threshold, args.rel_std_threshold,
        args.workers, log,
    )
    log(f"  Phase B total: {time.time()-t0:.1f}s")

    # Phase A — transform discovery on masked pixels
    t0 = time.time()
    transforms = phase_a_transforms(
        files, var_list, variation, args.max_z,
        args.n_test_files, args.n_per_file, args.workers, log,
    )
    log(f"  Phase A total: {time.time()-t0:.1f}s")

    # Phase C — global stats with chosen transforms
    t0 = time.time()
    global_acc = phase_c_global(
        files, var_list, variation, transforms, args.workers, log,
    )
    log(f"  Phase C total: {time.time()-t0:.1f}s")

    # Build output JSON
    statistics = {}
    for v in var_list:
        if v in global_acc:
            mean, std = global_acc[v].get_mean_std()
            vmin = float(global_acc[v].vmin) if np.isfinite(global_acc[v].vmin) else None
            vmax = float(global_acc[v].vmax) if np.isfinite(global_acc[v].vmax) else None
            count = int(global_acc[v].count)
        else:
            mean = std = 0.0
            vmin = vmax = None
            count = 0
        t = transforms[v]
        entry = {
            "mean": mean,
            "std": std,
            "vmin_after_transform": vmin,
            "vmax_after_transform": vmax,
            "n_observations": count,
            "valid_pixels": n_valid.get(v, 0),
            "total_pixels": npix,
            "transform_type": t["type"],
            "achieved_max_z_sample": (
                None if np.isnan(t.get("achieved_max_z", float("nan")))
                else float(t["achieved_max_z"])
            ),
        }
        if t["type"] == "power":
            entry["transform_p"] = t["p"]
        elif t["type"] == "asinh":
            entry["transform_scale"] = t["scale"]
        statistics[v] = entry

    output = {
        "metadata": {
            "nside": 64,
            "npix": npix,
            "ordering": "RING",
            "n_variables": len(var_list),
            "n_files": len(files),
            "max_z_target": args.max_z,
            "abs_std_threshold": args.abs_std_threshold,
            "rel_std_threshold": args.rel_std_threshold,
            "creation_date": datetime.utcnow().isoformat() + "Z",
            "data_dir": str(data_dir),
            "package": "monan_ia_preproc 0.1.0",
            "phase_order": "B (masks) → A (transforms on masked samples) → C (global stats)",
            "nan_policy": "Forward: NaN→0 in z-scored output. Inverse: use coverage_mask "
                          "to restore NaN where variable not geographically defined.",
        },
        "statistics": statistics,
    }
    json_path = out_dir / "healpix_statistics_v3.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"  Wrote stats : {json_path}")

    # Save dual masks
    mask_path = out_dir / "healpix_masks_v3.nc"
    write_masks_netcdf(
        mask_path, var_list, npix, coverage, variation,
        extra_attrs={
            "abs_std_threshold": args.abs_std_threshold,
            "rel_std_threshold": args.rel_std_threshold,
            "n_files_scanned": len(files),
            "phase_order": "B (masks) → A (transforms on masked samples) → C (global stats)",
        },
    )
    log(f"  Wrote masks : {mask_path}")

    # Final summary
    n_pow = sum(1 for v in var_list if statistics[v]["transform_type"] == "power")
    n_asinh = sum(1 for v in var_list if statistics[v]["transform_type"] == "asinh")
    n_id = sum(1 for v in var_list if statistics[v]["transform_type"] == "identity")
    n_target = sum(
        1 for v in var_list
        if (statistics[v]["achieved_max_z_sample"] is not None
            and statistics[v]["achieved_max_z_sample"] <= args.max_z)
    )
    log("=" * 72)
    log(f"  Summary ({len(var_list)} vars):")
    log(f"    transforms: {n_pow} power, {n_asinh} asinh, {n_id} identity")
    log(f"    {n_target}/{len(var_list)} achieve max|z|≤{args.max_z} (sample-based)")
    log("=" * 72)


if __name__ == "__main__":
    main()
