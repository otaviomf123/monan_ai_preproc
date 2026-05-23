#!/usr/bin/env python3
"""Normalize v3 monthly HEALPix NetCDFs → big-memmap split files.

Per split (train/val/test), writes:
    {out_dir}/{split}.npy           memmap-able (n_steps, npix, n_vars) float16
    {out_dir}/{split}_times.npy     datetime64 (n_steps,)
    {out_dir}/{split}_index.json    metadata (var_list, transforms, date range, ...)

Forward normalization per variable v:
    y     = apply_transform(x, transforms[v])   (signed_power or asinh; identity)
    z     = (y - mean_v) / std_v
    z     = clip(z, -max_z_clip, +max_z_clip)
    z     = where NaN: 0.0
    out   = z.astype(float16)

Inverse (denormalization, run downstream by the model caller):
    y     = z.astype(float32) * std_v + mean_v
    x     = invert_transform(y, transforms[v])
    x     = where coverage_mask == 0: NaN

Output supports memmap natively (via `np.load(path, mmap_mode='r')`).

Usage:
    python scripts/03_normalize.py \\
        --data-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3 \\
        --stats    .../healpix_statistics_v3.json \\
        --masks    .../healpix_masks_v3.nc \\
        --out-dir  /scratch/local/monan_ia_v3/normalized \\
        --split    train --start-date 1979-01-01 --end-date 2021-11-30 \\
        --max-z-clip 10 --workers 16
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monan_preproc.stats import apply_transform


# ---------------------------------------------------------------------------
def parse_date(s: str) -> np.datetime64:
    return np.datetime64(s)


def discover_files_in_range(
    data_dir: Path, start: np.datetime64, end: np.datetime64,
) -> list[Path]:
    """List v3 monthly NetCDFs whose month overlaps [start, end] (inclusive end)."""
    end_inclusive = end + np.timedelta64(1, "D")  # treat end as inclusive day
    files: list[Path] = []
    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith("_"):
            continue
        try:
            y = int(year_dir.name)
        except ValueError:
            continue
        for f in sorted(year_dir.glob(f"{y}_*.nc")):
            try:
                m = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            mo_start = np.datetime64(f"{y:04d}-{m:02d}-01")
            mo_end_y, mo_end_m = (y + 1, 1) if m == 12 else (y, m + 1)
            mo_end = np.datetime64(f"{mo_end_y:04d}-{mo_end_m:02d}-01")
            if mo_end <= start or mo_start >= end_inclusive:
                continue
            files.append(f)
    return files


def get_dataset_basics(sample_file: Path) -> tuple[list[str], int]:
    """Return (sorted_var_list, npix) — same convention as stats output."""
    with xr.open_dataset(sample_file) as ds:
        return sorted(ds.data_vars), int(ds.sizes["pixel"])


# ---------------------------------------------------------------------------
def normalize_one_timestep(
    arrs_per_var: dict, var_list: list[str],
    means: np.ndarray, stds: np.ndarray, transforms: list[dict],
    max_z_clip: float,
) -> np.ndarray:
    """Apply transform + z-score + clip + float16 cast for ONE timestep.

    arrs_per_var: dict var_name -> (npix,) float64 raw values for this timestep.
    Returns (npix, n_vars) float16, var order matches var_list.
    """
    npix = next(iter(arrs_per_var.values())).shape[0]
    out = np.zeros((npix, len(var_list)), dtype=np.float16)
    for j, v in enumerate(var_list):
        if v not in arrs_per_var:
            continue
        raw = arrs_per_var[v]
        y = apply_transform(raw, transforms[j])
        std = stds[j] if stds[j] > 1e-15 else 1.0
        z = (y - means[j]) / std
        z = np.clip(z, -max_z_clip, max_z_clip)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, j] = z.astype(np.float16)
    return out


def _worker_normalize_files(args):
    """Worker: open assigned monthly files, normalize timesteps in range, write
    to the shared memmap at the right indices.

    args = (worker_id, log_dir, out_path, shape, dtype_str, files_and_offsets,
            var_list, means, stds, transforms, max_z_clip,
            start_dt64, end_dt64)
    Returns: list of (file_name, n_written)
    """
    (worker_id, log_dir, out_path, shape, dtype_str, files_and_offsets,
     var_list, means, stds, transforms, max_z_clip,
     start_dt64, end_dt64) = args

    log_path = Path(log_dir) / f"worker_{worker_id:02d}.log"
    def wlog(msg: str) -> None:
        with open(log_path, "a") as fh:
            fh.write(f"{datetime.utcnow().isoformat()} [w{worker_id:02d} pid={os.getpid()}] {msg}\n")

    wlog(f"START — assigned {len(files_and_offsets)} files")

    # Open shared memmap for partial writes (writes go to disjoint slices)
    mm = np.lib.format.open_memmap(out_path, mode="r+", dtype=np.dtype(dtype_str),
                                    shape=shape)
    wlog("memmap opened")

    report = []
    t0 = time.time()
    for i_f, (f_path, offset_idx) in enumerate(files_and_offsets):
        try:
            with xr.open_dataset(f_path) as ds:
                times = ds["time"].values
                mask = (times >= start_dt64) & (times <= end_dt64)
                idx = np.where(mask)[0]
                if idx.size == 0:
                    report.append((f_path.name, 0))
                    continue

                data_per_var = {}
                for v in var_list:
                    if v in ds.data_vars:
                        data_per_var[v] = ds[v].values

                cur = offset_idx
                for t_idx in idx:
                    per_var_t = {v: data_per_var[v][t_idx].astype(np.float64)
                                  for v in data_per_var}
                    norm = normalize_one_timestep(
                        per_var_t, var_list, means, stds, transforms, max_z_clip,
                    )
                    mm[cur] = norm
                    cur += 1

                report.append((f_path.name, len(idx)))
                if (i_f + 1) % 5 == 0 or i_f == len(files_and_offsets) - 1:
                    dt = time.time() - t0
                    wlog(f"progress: {i_f+1}/{len(files_and_offsets)} files "
                         f"({dt:.0f}s elapsed, last={f_path.name})")
        except Exception as e:
            wlog(f"WARN {f_path.name}: {e}")
            report.append((f_path.name, 0))

    # No explicit flush — let `del mm` + process exit handle it.
    # (Explicit msync of large overlapping regions in many procs can stall.)
    del mm
    wlog(f"DONE — {sum(n for _, n in report)} timesteps in {time.time()-t0:.0f}s")
    return report


# ---------------------------------------------------------------------------
def build_writes_index(
    files: list[Path], start: np.datetime64, end: np.datetime64,
) -> tuple[list[tuple[Path, int]], np.ndarray]:
    """Determine, for each monthly file, how many timesteps fall in [start, end]
    and at what offset they should be written to the output array.

    Returns:
        files_and_offsets: list of (Path, output_start_idx)
        all_times: np.ndarray of all timesteps in order, length n_total
    """
    files_and_offsets = []
    all_times: list[np.datetime64] = []
    cur = 0
    for f in files:
        with xr.open_dataset(f) as ds:
            times = ds["time"].values
        mask = (times >= start) & (times <= end)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        files_and_offsets.append((f, cur))
        all_times.extend(times[mask].tolist())
        cur += cnt
    return files_and_offsets, np.array(all_times, dtype="datetime64[ns]")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path,
                   help="root with v3 monthly NetCDFs in {YYYY}/{YYYY}_{MM}.nc")
    p.add_argument("--stats", required=True, type=Path,
                   help="healpix_statistics_v3.json from Phase 2")
    p.add_argument("--masks", required=True, type=Path,
                   help="healpix_masks_v3.nc (will be copied to out-dir)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--start-date", required=True, type=parse_date,
                   help="inclusive start, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, type=parse_date,
                   help="inclusive end, YYYY-MM-DD")
    p.add_argument("--max-z-clip", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=8,
                   help="Process pool size (spawn ctx). 8 is robust on 1 NVMe.")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"normalize_{args.split}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    log = logging.getLogger().info

    log("=" * 72)
    log(f"03_normalize: split={args.split}")
    log(f"  data_dir : {args.data_dir}")
    log(f"  stats    : {args.stats}")
    log(f"  masks    : {args.masks}")
    log(f"  out_dir  : {args.out_dir}")
    log(f"  range    : {args.start_date} → {args.end_date}")
    log(f"  max|z|   : {args.max_z_clip}")
    log(f"  dtype    : {args.dtype}")
    log(f"  workers  : {args.workers}")
    log("=" * 72)

    # Load stats
    with open(args.stats) as f:
        stats_blob = json.load(f)
    stats_dict = stats_blob["statistics"]

    # Discover files in range
    files = discover_files_in_range(args.data_dir, args.start_date, args.end_date)
    log(f"  files in range: {len(files)}")
    if not files:
        raise SystemExit("No v3 files found in date range")

    # Var list from first file (sorted, matches stats key order)
    var_list, npix = get_dataset_basics(files[0])
    log(f"  vars     : {len(var_list)}")
    log(f"  pixels   : {npix}")

    # Build per-var arrays for fast lookup in workers
    means = np.array([stats_dict[v]["mean"] for v in var_list], dtype=np.float64)
    stds = np.array([stats_dict[v]["std"] for v in var_list], dtype=np.float64)
    # Transforms: list of dicts in var_list order
    transforms = []
    for v in var_list:
        s = stats_dict[v]
        t = {"type": s["transform_type"]}
        if s["transform_type"] == "power":
            t["p"] = float(s["transform_p"])
        elif s["transform_type"] == "asinh":
            t["scale"] = float(s["transform_scale"])
        transforms.append(t)
    n_pow = sum(1 for t in transforms if t["type"] == "power")
    n_asinh = sum(1 for t in transforms if t["type"] == "asinh")
    n_id = sum(1 for t in transforms if t["type"] == "identity")
    log(f"  transforms: {n_pow} power, {n_asinh} asinh, {n_id} identity")

    # Single canonical date window — used by both index builder and workers.
    # end-date is INCLUSIVE for the entire day (T00, T06, T12, T18).
    start_dt64 = args.start_date.astype("datetime64[ns]")
    end_dt64 = (args.end_date.astype("datetime64[ns]")
                + np.timedelta64(23, "h")
                + np.timedelta64(59, "m")
                + np.timedelta64(59, "s"))

    # Build write index (assigns each file to an output offset)
    log("  building write index (scanning time axes) …")
    t0 = time.time()
    files_and_offsets, all_times = build_writes_index(
        files, start_dt64, end_dt64,
    )
    n_total = len(all_times)
    log(f"  total timesteps: {n_total} (index built in {time.time()-t0:.1f}s)")
    if n_total == 0:
        raise SystemExit("No timesteps fell in date range")

    # Pre-allocate output memmap
    out_npy = args.out_dir / f"{args.split}.npy"
    log(f"  pre-allocating memmap: {out_npy}")
    log(f"    shape=({n_total}, {npix}, {len(var_list)}) dtype={args.dtype}")
    nbytes_est = n_total * npix * len(var_list) * np.dtype(args.dtype).itemsize
    log(f"    estimated size: {nbytes_est/1e9:.1f} GB")
    t0 = time.time()
    mm = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype=np.dtype(args.dtype),
        shape=(n_total, npix, len(var_list)),
    )
    del mm  # close in main; workers reopen
    log(f"    allocated in {time.time()-t0:.1f}s")

    # Partition files across workers (contiguous slices, balanced)
    n_workers = max(1, min(args.workers, len(files_and_offsets)))
    chunks: list[list[tuple[Path, int]]] = [[] for _ in range(n_workers)]
    for i, fo in enumerate(files_and_offsets):
        chunks[i % n_workers].append(fo)
    chunks = [c for c in chunks if c]
    n_workers = len(chunks)
    log(f"  partitioned {len(files_and_offsets)} files across {n_workers} workers "
        f"(sizes: {[len(c) for c in chunks]})")

    # Per-worker log dir
    worker_log_dir = args.out_dir / f"normalize_{args.split}_worker_logs"
    if worker_log_dir.exists():
        shutil.rmtree(worker_log_dir)
    worker_log_dir.mkdir()

    # Dispatch
    worker_args = [
        (wid, str(worker_log_dir), str(out_npy),
         (n_total, npix, len(var_list)), args.dtype,
         c,
         var_list, means, stds, transforms,
         float(args.max_z_clip),
         start_dt64, end_dt64)
        for wid, c in enumerate(chunks)
    ]
    log(f"  spawning {n_workers} workers (spawn context, per-worker logs in "
        f"{worker_log_dir.name}/) …")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    reports: list = [None] * n_workers
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        future_to_wid = {ex.submit(_worker_normalize_files, wa): wa[0]
                         for wa in worker_args}
        done_count = 0
        for fut in as_completed(future_to_wid):
            wid = future_to_wid[fut]
            try:
                reports[wid] = fut.result()
                done_count += 1
                log(f"  worker {wid:02d} finished "
                    f"({done_count}/{n_workers}, elapsed={time.time()-t0:.0f}s)")
            except Exception as e:
                log(f"  worker {wid:02d} FAILED: {e}")
                reports[wid] = []
    elapsed = time.time() - t0
    log(f"  normalization done in {elapsed:.1f}s "
        f"({nbytes_est/1e9/elapsed:.2f} GB/s effective)")

    # Verify counts
    written = 0
    for r in reports:
        if r is None:
            continue
        for _, n_w in r:
            written += n_w
    log(f"  wrote {written}/{n_total} timesteps "
        f"({'OK' if written == n_total else 'MISMATCH!'})")
    if written != n_total:
        raise SystemExit(f"FATAL: only {written}/{n_total} timesteps written — "
                         "check worker logs and DO NOT trust output")

    # Save times array
    times_path = args.out_dir / f"{args.split}_times.npy"
    np.save(times_path, all_times)
    log(f"  wrote times : {times_path}")

    # Save index JSON
    index = {
        "split": args.split,
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "n_timesteps": int(n_total),
        "n_pixels": int(npix),
        "n_variables": len(var_list),
        "dtype": args.dtype,
        "max_z_clip": float(args.max_z_clip),
        "ordering": "RING",
        "nside": 64,
        "var_list": var_list,
        "n_files_consumed": len(files_and_offsets),
        "creation_date": datetime.utcnow().isoformat() + "Z",
        "package": "monan_ia_preproc 0.1.0",
        "stats_source": str(args.stats),
        "masks_source": str(args.masks),
        "data_source": str(args.data_dir),
    }
    index_path = args.out_dir / f"{args.split}_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    log(f"  wrote index : {index_path}")

    # Copy stats + masks into out_dir (first call only — idempotent)
    out_stats = args.out_dir / "stats.json"
    if not out_stats.exists():
        shutil.copy(args.stats, out_stats)
        log(f"  copied stats: {out_stats}")
    out_masks = args.out_dir / "masks.nc"
    if not out_masks.exists():
        shutil.copy(args.masks, out_masks)
        log(f"  copied masks: {out_masks}")

    log("=" * 72)
    log(f"DONE split={args.split} | {n_total} steps, {nbytes_est/1e9:.1f} GB on disk")
    log("=" * 72)


if __name__ == "__main__":
    main()
