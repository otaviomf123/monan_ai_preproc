#!/usr/bin/env python3
"""End-to-end consistency checks for the v3 preprocessed dataset.

Runs a battery of checks before training and prints PASS/FAIL per check
with a final summary. Designed to catch:

  - missing files / wrong shapes
  - var_list inconsistency across train/val/test/stats/clim
  - 6h cadence violations, time gaps, train/val/test leakage
  - NaN/Inf or out-of-clip values (sampled)
  - per-variable mean ≈ 0 / std ≈ 1 after z-score (sampled)
  - **silent zeros** (e.g. a variable missing from a monthly file leaving
    its slice as zeros in the memmap) — done as a per-month, per-var
    "all-zero-on-variation-mask" scan that would expose the bugs in
    `normalize_one_timestep` (`if v not in arrs_per_var: continue`)
  - climatology shape, DOY-60 interpolation, finiteness
  - memmap read throughput (sanity for DataLoader)

Usage:
    python scripts/06_consistency_checks.py \\
        --norm-dir /scratch/local/monan_ia_v3/normalized \\
        --clim-dir /scratch/local/monan_ia_v3/climatology
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

class CheckRunner:
    def __init__(self, log) -> None:
        self.log = log
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn) -> bool:
        self.log(f"\n=== {name} ===")
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        self.log(f"  → {status}: {msg}")
        self.results.append((name, ok, msg))
        return ok

    def summary(self) -> int:
        self.log("\n" + "=" * 72)
        self.log("SUMMARY")
        self.log("=" * 72)
        n_pass = sum(1 for _, ok, _ in self.results if ok)
        for name, ok, msg in self.results:
            status = "PASS" if ok else "FAIL"
            self.log(f"  [{status}] {name}: {msg}")
        self.log("=" * 72)
        self.log(f"  {n_pass}/{len(self.results)} checks passed")
        self.log("=" * 72)
        return 0 if n_pass == len(self.results) else 1


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_files_exist(paths: dict[str, Path]):
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        return False, f"missing: {missing}"
    sizes = {k: p.stat().st_size for k, p in paths.items()}
    return True, f"all present ({len(paths)} files); sizes (GB): " + \
                 ", ".join(f"{k}={v/1e9:.2f}" for k, v in sizes.items())


def check_var_list_consistency(norm_dir: Path, clim_dir: Path):
    """train/val/test/stats/clim mean/std must all expose the same 235 vars
    in the same order."""
    refs = {}

    # train index has the canonical order
    with open(norm_dir / "train_index.json") as f:
        train_idx = json.load(f)
    canonical = train_idx["var_list"]
    refs["train_index"] = canonical

    for s in ("val", "test"):
        with open(norm_dir / f"{s}_index.json") as f:
            refs[f"{s}_index"] = json.load(f)["var_list"]

    with open(norm_dir / "stats.json") as f:
        stats = json.load(f)
    refs["stats"] = sorted(stats["statistics"].keys())

    for f in ("climatology_doy_hour_mean_nside64_v3.nc",
              "climatology_doy_hour_std_nside64_v3.nc"):
        with xr.open_dataset(clim_dir / f) as ds:
            non_coord = [v for v in ds.data_vars if v not in ds.coords]
            refs[f] = sorted(non_coord)

    canonical_sorted = sorted(canonical)
    bad = []
    for k, v in refs.items():
        if k.endswith("_index"):
            if v != canonical:
                bad.append(f"{k}: order or content differs")
        else:
            if v != canonical_sorted:
                missing = set(canonical_sorted) - set(v)
                extra = set(v) - set(canonical_sorted)
                bad.append(f"{k}: missing={list(missing)[:3]}, extra={list(extra)[:3]}")
    if bad:
        return False, "; ".join(bad)
    return True, f"{len(canonical)} vars consistent across all sources"


def check_shape_dtype(norm_dir: Path, split: str):
    with open(norm_dir / f"{split}_index.json") as f:
        idx = json.load(f)
    arr = np.load(norm_dir / f"{split}.npy", mmap_mode="r")
    expected = (idx["n_timesteps"], idx["n_pixels"], idx["n_variables"])
    if arr.shape != expected:
        return False, f"shape {arr.shape} != expected {expected}"
    if str(arr.dtype) != idx["dtype"]:
        return False, f"dtype {arr.dtype} != expected {idx['dtype']}"
    return True, f"shape={arr.shape} dtype={arr.dtype}"


def check_time_axis(norm_dir: Path, split: str):
    with open(norm_dir / f"{split}_index.json") as f:
        idx = json.load(f)
    times = np.load(norm_dir / f"{split}_times.npy")
    n = len(times)
    if n != idx["n_timesteps"]:
        return False, f"len(times)={n} != index n_timesteps={idx['n_timesteps']}"

    diffs = np.diff(times.astype("datetime64[h]")).astype("timedelta64[h]").astype(int)
    if (diffs <= 0).any():
        return False, f"non-monotonic at indices {np.where(diffs<=0)[0][:5]}"
    bad = np.where(diffs != 6)[0]
    if len(bad) > 0:
        # Allow gaps at year boundaries if present (but flag them)
        return False, (f"{len(bad)} non-6h gaps; first at idx {bad[0]} "
                       f"(diff={diffs[bad[0]]}h between {times[bad[0]]} and {times[bad[0]+1]})")
    expected = idx["start_date"]
    actual_start = str(times[0].astype("datetime64[D]"))
    if not actual_start.startswith(expected):
        return False, f"start {actual_start} != index {expected}"
    return True, (f"n={n}, strictly +6h from {times[0]} to {times[-1]}")


def check_split_disjoint(norm_dir: Path):
    splits = {}
    for s in ("train", "val", "test"):
        splits[s] = np.load(norm_dir / f"{s}_times.npy")

    overlaps = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        sa = set(splits[a].astype("datetime64[h]").astype(int).tolist())
        sb = set(splits[b].astype("datetime64[h]").astype(int).tolist())
        inter = sa & sb
        if inter:
            overlaps.append(f"{a}∩{b}={len(inter)}")
    if overlaps:
        return False, "; ".join(overlaps)
    return True, (f"disjoint: train [{splits['train'][0]}..{splits['train'][-1]}] "
                  f"({len(splits['train'])}), "
                  f"val [{splits['val'][0]}..{splits['val'][-1]}] "
                  f"({len(splits['val'])}), "
                  f"test [{splits['test'][0]}..{splits['test'][-1]}] "
                  f"({len(splits['test'])})")


def check_finite_and_clip(norm_dir: Path, split: str, max_z: float, n_samples: int):
    arr = np.load(norm_dir / f"{split}.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(arr.shape[0], size=min(n_samples, arr.shape[0]),
                            replace=False)
    sample_idx.sort()
    n_nan = 0
    n_inf = 0
    over_clip = 0
    abs_max = 0.0
    for i in sample_idx:
        x = arr[i].astype(np.float32)  # (npix, n_vars)
        n_nan += int(np.isnan(x).sum())
        n_inf += int(np.isinf(x).sum())
        am = float(np.abs(x).max())
        abs_max = max(abs_max, am)
        over_clip += int((np.abs(x) > max_z + 1e-3).sum())
    msg = (f"sampled {len(sample_idx)} timesteps; "
           f"NaN={n_nan} Inf={n_inf} over-clip(|z|>{max_z})={over_clip}, "
           f"|z|_max={abs_max:.4f}")
    if n_nan or n_inf or over_clip > 0:
        return False, msg
    return True, msg


def check_global_stats_per_var(norm_dir: Path, split: str, n_samples: int):
    """For each var, sampled mean ~ 0, std ~ 1."""
    arr = np.load(norm_dir / f"{split}.npy", mmap_mode="r")
    with open(norm_dir / f"{split}_index.json") as f:
        idx = json.load(f)
    var_list = idx["var_list"]

    rng = np.random.default_rng(1)
    sample_idx = rng.choice(arr.shape[0], size=min(n_samples, arr.shape[0]),
                            replace=False)
    sample_idx.sort()

    # Pull samples once: (S, npix, nvar) → reduce over (S, npix)
    accum_sum = np.zeros(len(var_list), dtype=np.float64)
    accum_sumsq = np.zeros(len(var_list), dtype=np.float64)
    n_samp = 0
    for i in sample_idx:
        x = arr[i].astype(np.float32)
        accum_sum += x.sum(axis=0)
        accum_sumsq += (x * x).sum(axis=0)
        n_samp += x.shape[0]

    means = accum_sum / n_samp
    stds = np.sqrt(np.maximum(accum_sumsq / n_samp - means ** 2, 0))

    # Tolerances are loose because sample is only ~1% and variation_mask
    # excludes many pixels for some vars (so mean stays at 0 over those).
    bad_mean = []
    bad_std = []
    for j, v in enumerate(var_list):
        if abs(means[j]) > 0.4:  # large mean drift hint
            bad_mean.append((v, float(means[j])))
        if not (0.05 <= stds[j] <= 2.0):  # a real variable should be in this band
            bad_std.append((v, float(stds[j])))

    msg = (f"{len(var_list)} vars; mean range=[{means.min():.4f}, {means.max():.4f}], "
           f"std range=[{stds.min():.4f}, {stds.max():.4f}]")
    if bad_mean or bad_std:
        details = ""
        if bad_mean:
            details += f"; {len(bad_mean)} vars |mean|>0.4 (e.g. {bad_mean[:3]})"
        if bad_std:
            details += f"; {len(bad_std)} vars std∉[0.05,2.0] (e.g. {bad_std[:3]})"
        return False, msg + details
    return True, msg


def check_silent_zeros_per_month(norm_dir: Path, split: str, log=print,
                                  log_every: int = 60):
    """For each (month, variable), compute fraction of EXACTLY-zero values
    across pixels in the variation_mask. A real variable should not be
    100 % zero on a month inside its variation mask — that would mean the
    variable was missing from that month's NetCDF and the normalizer's
    `if v not in arrs_per_var: continue` left zeros behind.

    Fully vectorized per-month (no inner Python loop over vars).
    """
    arr = np.load(norm_dir / f"{split}.npy", mmap_mode="r")
    times = np.load(norm_dir / f"{split}_times.npy")
    with open(norm_dir / f"{split}_index.json") as f:
        idx = json.load(f)
    var_list = idx["var_list"]

    masks_path = norm_dir / "masks.nc"
    with xr.open_dataset(masks_path) as md:
        if "variation_mask" not in md.data_vars:
            return False, "variation_mask not in masks.nc"
        vmask = md["variation_mask"].values.astype(bool)  # (n_vars, n_pix)
        if "variable" in md.coords:
            mask_vars = list(md["variable"].values.astype(str))
            order = [mask_vars.index(v) for v in var_list]
            vmask = vmask[order]
    if vmask.shape != (len(var_list), arr.shape[1]):
        return False, f"variation_mask shape {vmask.shape} mismatch"

    n_in_mask = vmask.sum(axis=1).astype(np.int64)  # (n_vars,)
    vmask_T = vmask.T.astype(np.int8)  # (n_pix, n_vars), saves RAM vs bool*int

    months = times.astype("datetime64[M]")
    unique_months = np.unique(months)
    n_vars = len(var_list)

    suspicious: list[tuple[str, str, float, int]] = []
    t0 = time.time()
    for k, ym in enumerate(unique_months):
        idx_in_month = np.where(months == ym)[0]
        T = len(idx_in_month)
        # Read raw float16 (no cast — bool compare works on f16)
        block = arr[idx_in_month]  # (T, npix, nvar) float16
        is_zero = (block == 0)  # (T, npix, nvar) bool

        # Per-var sum of zeros in mask, vectorized:
        # sum_T  = is_zero.sum(axis=0) → (npix, nvar) uint16/int32
        # masked = (sum_T * vmask_T).sum(axis=0) → (nvar,)
        sum_T = is_zero.sum(axis=0, dtype=np.int32)         # (npix, nvar)
        masked_sum = (sum_T * vmask_T).sum(axis=0).astype(np.int64)  # (nvar,)
        denom = T * n_in_mask  # (nvar,)
        zfrac = np.where(n_in_mask > 0,
                          masked_sum.astype(np.float64) / np.maximum(denom, 1),
                          0.0)
        bad_vars = np.where(zfrac > 0.99)[0]
        for j in bad_vars:
            if n_in_mask[j] > 0:
                suspicious.append((str(ym.astype("datetime64[D]"))[:7],
                                   var_list[int(j)], float(zfrac[j]),
                                   int(n_in_mask[j])))

        if (k + 1) % log_every == 0 or k == len(unique_months) - 1:
            log(f"    [silent_zeros/{split}] {k+1}/{len(unique_months)} months "
                f"({time.time()-t0:.0f}s, {len(suspicious)} suspicious so far)")

    msg = f"scanned {len(unique_months)} months × {n_vars} vars"
    if suspicious:
        head = suspicious[:15]
        return False, (msg + f"; {len(suspicious)} (month,var) cells with "
                       f">99% zeros on variation_mask. first 15: {head}")
    return True, msg + "; no var-month is >99% zero on its variation_mask"


def check_climatology(clim_dir: Path, expected_npix: int, expected_n_vars: int):
    msgs = []
    for f in ("climatology_doy_hour_mean_nside64_v3.nc",
              "climatology_doy_hour_std_nside64_v3.nc"):
        with xr.open_dataset(clim_dir / f) as ds:
            if dict(ds.dims) != {"dayofyear": 366, "hour": 4, "pixel": expected_npix}:
                return False, f"{f}: bad dims {dict(ds.dims)}"
            data_vars = [v for v in ds.data_vars if v not in ds.coords]
            if len(data_vars) != expected_n_vars:
                return False, f"{f}: n_vars={len(data_vars)} != {expected_n_vars}"
            # Spot check: sample 5 vars across hours; finite, ranges sane
            spot = data_vars[::max(1, len(data_vars) // 5)][:5]
            for v in spot:
                a = ds[v].values
                if not np.isfinite(a).all():
                    return False, f"{f}/{v}: contains non-finite values"
                # DOY 60 should be ≈ avg(DOY 59, 61) per our interpolation
                d59 = a[58]
                d60 = a[59]
                d61 = a[60]
                expected = 0.5 * (d59 + d61)
                err = float(np.max(np.abs(d60 - expected)))
                if err > 1e-5:
                    return False, f"{f}/{v}: DOY 60 deviates from interp (max err {err:.2e})"
            msgs.append(f"{f.split('.')[0][-15:]}: dims OK, {len(data_vars)} vars, "
                        f"DOY60 interp verified on {len(spot)} sampled vars")
    return True, " | ".join(msgs)


def check_memmap_read_perf(norm_dir: Path, n_random: int):
    arr = np.load(norm_dir / "train.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    idx_random = rng.integers(0, arr.shape[0], size=n_random)

    t0 = time.time()
    bytes_read = 0
    for i in idx_random:
        x = arr[i]  # forces page-in
        bytes_read += x.nbytes
        # touch a few pixels to ensure read isn't lazy
        _ = float(x[0, 0])
    dt = time.time() - t0
    mb_per_s = bytes_read / 1e6 / dt
    return True, (f"{n_random} random timestep reads in {dt:.2f}s "
                  f"({mb_per_s:.1f} MB/s, ~{n_random/dt:.0f} samples/s)")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norm-dir", required=True, type=Path)
    p.add_argument("--clim-dir", required=True, type=Path)
    p.add_argument("--max-z", type=float, default=10.0)
    p.add_argument("--n-finite-samples", type=int, default=200)
    p.add_argument("--n-stat-samples", type=int, default=400)
    p.add_argument("--n-perf-reads", type=int, default=64)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger().info

    log("=" * 72)
    log("06_consistency_checks")
    log(f"  norm_dir: {args.norm_dir}")
    log(f"  clim_dir: {args.clim_dir}")
    log("=" * 72)

    cr = CheckRunner(log)

    # collect expected sizes
    paths = {}
    for s in ("train", "val", "test"):
        paths[f"{s}.npy"] = args.norm_dir / f"{s}.npy"
        paths[f"{s}_times.npy"] = args.norm_dir / f"{s}_times.npy"
        paths[f"{s}_index.json"] = args.norm_dir / f"{s}_index.json"
    paths["stats.json"] = args.norm_dir / "stats.json"
    paths["masks.nc"] = args.norm_dir / "masks.nc"
    paths["clim_mean"] = args.clim_dir / "climatology_doy_hour_mean_nside64_v3.nc"
    paths["clim_std"] = args.clim_dir / "climatology_doy_hour_std_nside64_v3.nc"

    cr.run("files_exist", lambda: check_files_exist(paths))
    cr.run("var_list_consistency",
           lambda: check_var_list_consistency(args.norm_dir, args.clim_dir))
    for s in ("train", "val", "test"):
        cr.run(f"shape_dtype/{s}", lambda s=s: check_shape_dtype(args.norm_dir, s))
        cr.run(f"time_axis/{s}",   lambda s=s: check_time_axis(args.norm_dir, s))
    cr.run("split_disjoint", lambda: check_split_disjoint(args.norm_dir))
    for s in ("train", "val", "test"):
        cr.run(f"finite_and_clip/{s}",
               lambda s=s: check_finite_and_clip(args.norm_dir, s,
                                                  args.max_z, args.n_finite_samples))
    cr.run("global_stats_per_var/train",
           lambda: check_global_stats_per_var(args.norm_dir, "train",
                                              args.n_stat_samples))
    cr.run("silent_zeros_per_month/train",
           lambda: check_silent_zeros_per_month(args.norm_dir, "train", log))
    cr.run("silent_zeros_per_month/val",
           lambda: check_silent_zeros_per_month(args.norm_dir, "val", log,
                                                 log_every=12))
    cr.run("silent_zeros_per_month/test",
           lambda: check_silent_zeros_per_month(args.norm_dir, "test", log,
                                                 log_every=12))

    # for clim, learn npix and n_vars from train
    with open(args.norm_dir / "train_index.json") as f:
        train_idx = json.load(f)
    cr.run("climatology",
           lambda: check_climatology(args.clim_dir,
                                     train_idx["n_pixels"],
                                     train_idx["n_variables"]))
    cr.run("memmap_read_perf",
           lambda: check_memmap_read_perf(args.norm_dir, args.n_perf_reads))

    rc = cr.summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
