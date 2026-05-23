#!/usr/bin/env python3
"""Conservative downsampling of v3 monthly HEALPix NetCDFs from nside=64
to nside=16 using `HierarchicalDegrader` (NaN-aware nanmean of 4 children
per step, applied twice: 64→32→16).

On HEALPix all pixels have equal area, so plain nanmean over the 16
children of each parent IS area-weighted (conservative for both
intensive and per-area extensive quantities).

Inputs (per month):
    {src_dir}/{YYYY}/{YYYY}_{MM}.nc       235 vars × (n_t, 49152) float32

Outputs (mirrored layout):
    {dst_dir}/{YYYY}/{YYYY}_{MM}.nc       235 vars × (n_t, 3072)  float32

Defaults:
    src_dir = /mnt/beegfs/otavio.feitosa/era5_healpix_v3
    dst_dir = /mnt/beegfs/otavio.feitosa/era5_healpix_v3_n16

Strict checks (same spirit as `convert.py`):
    - source must expose exactly 235 atmospheric vars (drop healpix coords)
    - output written atomically (`.tmp.nc` → rename); strict 235-var check
      before rename
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monan_preproc.healpix import HierarchicalDegrader, healpix_coords
from monan_preproc.convert import write_output

EXPECTED_N_VARS = 235
COORD_VARS = {"healpix_lat", "healpix_lon"}


def discover_files(src_dir: Path) -> list[Path]:
    files: list[Path] = []
    for year_dir in sorted(src_dir.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith("_"):
            continue
        try:
            int(year_dir.name)
        except ValueError:
            continue
        files.extend(sorted(year_dir.glob(f"{year_dir.name}_*.nc")))
    return files


def degrade_one_file(src_path: Path, dst_root: Path,
                      degrader: HierarchicalDegrader, log,
                      force: bool = False, dst_nside: int = 16) -> str:
    """Returns 'ok' | 'skip' | 'fail:<reason>'."""
    y = src_path.parent.name
    name = src_path.name
    dst_path = dst_root / y / name
    if dst_path.exists() and not force:
        return "skip"

    t0 = time.time()
    try:
        with xr.open_dataset(src_path) as ds:
            data_vars = [v for v in ds.data_vars if v not in COORD_VARS]
            if len(data_vars) != EXPECTED_N_VARS:
                return f"fail:bad_n_vars_in={len(data_vars)}"
            target_times = ds["time"].values
            all_vars: dict[str, np.ndarray] = {}
            all_attrs: dict[str, dict] = {}
            for v in data_vars:
                raw = ds[v].values  # (n_t, 49152) float32 (or float64)
                if raw.ndim != 2 or raw.shape[-1] != 49152:
                    return f"fail:bad_shape:{v}={raw.shape}"
                out = degrader(raw.astype(np.float32, copy=False), pix_axis=-1)
                # out shape: (n_t, 3072)
                all_vars[v] = out
                # Preserve per-var attrs from source + augment regrid note
                attrs = dict(ds[v].attrs)
                attrs["regrid"] = (
                    f"{attrs.get('regrid', '')}; "
                    f"downsampled nanmean nside=64→16"
                ).lstrip("; ")
                all_attrs[v] = attrs

        if len(all_vars) != EXPECTED_N_VARS:
            return f"fail:bad_n_vars_out={len(all_vars)}"

        # Atomic write
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_tmp = dst_path.with_suffix(".tmp.nc")
        if dst_tmp.exists():
            dst_tmp.unlink()
        write_output(
            dst_tmp,
            target_times=target_times,
            all_vars=all_vars,
            all_attrs=all_attrs,
            nside=dst_nside,
            nest=False,
            extra_attrs={
                "downsample_source_nside": 64,
                "downsample_method": "hierarchical nanmean (4 children per step, ×2)",
                "downsample_note": (
                    "Conservative on HEALPix equal-area grid; NaN-aware for "
                    "partial-coverage fields (SST, soil, sea ice)."
                ),
                "downsample_date": datetime.utcnow().isoformat() + "Z",
            },
        )
        dst_tmp.rename(dst_path)
        sz = dst_path.stat().st_size / 1e6
        log(f"  [{name}] OK ({sz:.1f} MB, {time.time()-t0:.1f}s)")
        return "ok"
    except Exception as e:
        return f"fail:exception:{type(e).__name__}:{e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path,
                   default=Path("/mnt/beegfs/otavio.feitosa/era5_healpix_v3"))
    p.add_argument("--dst-dir", type=Path,
                   default=Path("/mnt/beegfs/otavio.feitosa/era5_healpix_v3_n16"))
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Cache dir for HEALPix children mapping "
                        "(default: <dst_dir>/_cache)")
    p.add_argument("--shard", default="0/1",
                   help="Shard spec 'i/N' for SLURM-array parallelism "
                        "(this task processes files where idx %% N == i)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing destination files")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger().info

    i_str, n_str = args.shard.split("/")
    shard_i, shard_n = int(i_str), int(n_str)
    if not (0 <= shard_i < shard_n):
        raise SystemExit(f"bad shard: {args.shard}")

    cache_dir = args.cache_dir or (args.dst_dir / "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("07_degrade_n64_to_n16")
    log(f"  src     : {args.src_dir}")
    log(f"  dst     : {args.dst_dir}")
    log(f"  cache   : {cache_dir}")
    log(f"  shard   : {shard_i}/{shard_n}")
    log("=" * 72)

    # Build degrader once (caches children mapping)
    log("  building degrader nside=64→16 (cached) …")
    t0 = time.time()
    degrader = HierarchicalDegrader(
        nside_high=64, nside_low=16, nest=False, cache_dir=cache_dir,
    )
    log(f"    degrader ready in {time.time()-t0:.1f}s "
        f"({len(degrader.mappings)} pass(es): "
        f"{[c.shape for c in degrader.mappings]})")

    # Sanity check: 3072 parents, each with 4 children at the intermediate step
    n_pix_low = 12 * 16 * 16
    for k, m in enumerate(degrader.mappings):
        n_parents, n_children = m.shape
        log(f"    pass {k}: parents={n_parents}, children/parent={n_children}")

    files = discover_files(args.src_dir)
    log(f"  discovered {len(files)} source files")
    my_files = [f for k, f in enumerate(files) if k % shard_n == shard_i]
    log(f"  this shard processes {len(my_files)} files")

    counts: dict[str, int] = {"ok": 0, "skip": 0, "fail": 0}
    fails: list[tuple[str, str]] = []
    t_start = time.time()
    for k, src in enumerate(my_files):
        result = degrade_one_file(src, args.dst_dir, degrader, log,
                                   force=args.force)
        if result == "ok":
            counts["ok"] += 1
        elif result == "skip":
            counts["skip"] += 1
            log(f"  [{src.name}] skip (exists)")
        else:
            counts["fail"] += 1
            fails.append((src.name, result))
            log(f"  [{src.name}] FAIL: {result}")
        if (k + 1) % 20 == 0:
            log(f"  ---- shard {shard_i}/{shard_n} progress: "
                f"{k+1}/{len(my_files)} "
                f"({time.time()-t_start:.0f}s elapsed) ----")

    log("=" * 72)
    log(f"DONE shard {shard_i}/{shard_n} in {time.time()-t_start:.0f}s — "
        f"ok={counts['ok']}, skip={counts['skip']}, fail={counts['fail']}")
    if fails:
        log("  failures:")
        for name, why in fails:
            log(f"    {name}: {why}")
    log("=" * 72)
    sys.exit(0 if counts["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
