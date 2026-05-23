#!/usr/bin/env python3
"""CLI entry point: convert one (year, month) ERA5 → HEALPix nside=64.

Usage:
    python -m monan_preproc.scripts.convert_month --year 1951 --month 1
    python scripts/convert_month.py --year 1951 --month 1 --overwrite
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable when running this file directly from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monan_preproc.convert import convert_month


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERA5 → HEALPix nside=64 monthly converter")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument(
        "--raw-root",
        type=str,
        default="/mnt/beegfs/otavio.feitosa/era5",
        help="Root with main + data_additional_levels + data_hydrometeors + data_ozone_extended",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="/mnt/beegfs/otavio.feitosa/era5_healpix_v3",
    )
    p.add_argument("--nside-high", type=int, default=256)
    p.add_argument("--nside-low", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    out = convert_month(
        year=args.year,
        month=args.month,
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        nside_high=args.nside_high,
        nside_low=args.nside_low,
        overwrite=args.overwrite,
    )
    print(f"WROTE: {out}")


if __name__ == "__main__":
    main()
