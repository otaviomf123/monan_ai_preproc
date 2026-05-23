#!/usr/bin/env python3
"""Pre-compute the BallTree regrid index + hierarchical-degrade children
mappings, save to cache_dir. Run once before launching the array job —
all parallel workers then load from cache instead of recomputing.

Usage:
    python scripts/precompute_cache.py \\
        --cache-dir /mnt/beegfs/otavio.feitosa/era5_healpix_v3/_cache \\
        --nside-high 256 --nside-low 64 --ring
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monan_preproc.cache import (
    load_or_build_children,
    load_or_build_regrid_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--nside-high", type=int, default=256)
    p.add_argument("--nside-low", type=int, default=64)
    p.add_argument("--ring", action="store_true",
                   help="Use RING ordering (default). Use --nested to override.")
    p.add_argument("--nested", action="store_true")
    # ERA5 1° regular grid (181 lat × 360 lon, 90→-90, 0→359)
    p.add_argument("--n-lat", type=int, default=181)
    p.add_argument("--n-lon", type=int, default=360)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ring and args.nested:
        raise SystemExit("Pick one: --ring OR --nested")
    nest = args.nested  # default RING
    ordering = "NESTED" if nest else "RING"

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== precompute cache ===")
    print(f"  cache_dir : {args.cache_dir}")
    print(f"  ordering  : {ordering}")
    print(f"  source    : {args.n_lat}×{args.n_lon} regular lat-lon")
    print(f"  nside     : {args.nside_high} → {args.nside_low}")
    print()

    # ERA5 1° grid: lat from 90 to -90 (north→south), lon from 0 to 359
    lat = np.linspace(90.0, -90.0, args.n_lat, dtype=np.float32)
    lon = np.linspace(0.0, 359.0, args.n_lon, dtype=np.float32)

    t0 = time.time()
    idx = load_or_build_regrid_indices(
        cache_dir=args.cache_dir,
        lat=lat, lon=lon,
        nside_high=args.nside_high,
        nest=nest,
        verbose=True,
    )
    t1 = time.time()
    print(f"  regrid index ready: {idx.size} entries  [{t1-t0:.1f}s]")

    t0 = time.time()
    children = load_or_build_children(
        cache_dir=args.cache_dir,
        nside_high=args.nside_high,
        nside_low=args.nside_low,
        nest=nest,
        verbose=True,
    )
    t1 = time.time()
    print(f"  degrade children ready: {len(children)} steps  [{t1-t0:.1f}s]")

    print(f"\nCache files:")
    for f in sorted(args.cache_dir.glob("*.npz")):
        print(f"  {f.name}: {f.stat().st_size/1e6:.2f} MB")
    print("\nDONE — workers can now load these instead of recomputing.")


if __name__ == "__main__":
    main()
