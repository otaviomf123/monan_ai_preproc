#!/usr/bin/env python3
"""Sanity-check a converted month: variable count, NaN coverage, value ranges,
and physical plausibility for a few key variables.

Usage:
    python scripts/validate_output.py /mnt/beegfs/otavio.feitosa/era5_healpix_v3/1951/1951_01.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# (var name → (expected_min, expected_max) in physical units after preproc)
EXPECTED_RANGES = {
    "t2m":            (170.0, 340.0),       # K
    "msl":            (94000.0, 110000.0),  # Pa
    "sst":            (270.0, 310.0),       # K (or NaN over land)
    "siconc":         (0.0, 1.0),
    "u10":            (-100.0, 100.0),      # m/s
    "v10":            (-100.0, 100.0),
    "tp":             (0.0, 500.0),         # mm/6h
    "tcwv":           (0.0, 100.0),         # kg/m²
    "sshf":           (-1000.0, 1000.0),    # W/m² (mean over 6h)
    "tisr":           (0.0, 1500.0),        # W/m²
    "t_500hpa":       (200.0, 290.0),
    "t_850hpa":       (220.0, 320.0),
    "z_500hpa":       (4.0e4, 6.5e4),       # m²/s² (geopotential, polar winter can be very low)
    "u_250hpa":       (-150.0, 150.0),
    "r_1000hpa":      (-5.0, 150.0),        # ECMWF reports up to ~140% (ice supersat) and small negatives
    "r_500hpa":       (-5.0, 150.0),
    "o3_50hpa":       (0.0, 5e-5),
    "o3_500hpa":      (0.0, 5e-7),
    "clwc_850hpa":    (0.0, 5e-3),
    "ciwc_300hpa":    (0.0, 5e-3),
    "cc_500hpa":      (0.0, 1.0),
}


def _summary(arr: np.ndarray) -> dict:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {"all_nan": True}
    return {
        "n": int(arr.size),
        "n_nan": int(arr.size - a.size),
        "pct_nan": float(100 * (arr.size - a.size) / arr.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str)
    args = p.parse_args()

    ds = xr.open_dataset(args.path)
    print(f"FILE: {args.path}")
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  attrs: nside={ds.attrs.get('nside')}, ordering={ds.attrs.get('ordering')}, "
          f"n_variables={ds.attrs.get('n_variables')}")
    print(f"  variable counts: dyn={ds.attrs.get('n_pressure_dyn')}, "
          f"o3_ext={ds.attrs.get('n_o3_extended')}, "
          f"hydromet={ds.attrs.get('n_hydrometeors')}, "
          f"surf_inst={ds.attrs.get('n_surface_inst')}, "
          f"surf_accum={ds.attrs.get('n_surface_accum')}")
    print()
    print(f"Total variables in file: {len(ds.data_vars)}")
    print()

    print(f"{'VARIABLE':<22}{'pct_nan':>9}{'min':>14}{'max':>14}{'mean':>14}  STATUS")
    print("-" * 90)
    fails = 0
    for v, (lo, hi) in EXPECTED_RANGES.items():
        if v not in ds.data_vars:
            print(f"{v:<22}                                              MISSING")
            fails += 1
            continue
        s = _summary(ds[v].values)
        if s.get("all_nan"):
            print(f"{v:<22} ALL NAN")
            fails += 1
            continue
        ok = (s["min"] >= lo) and (s["max"] <= hi)
        flag = "OK" if ok else "OUT-OF-RANGE"
        if not ok:
            fails += 1
        print(
            f"{v:<22}{s['pct_nan']:>8.2f}%{s['min']:>14.4g}{s['max']:>14.4g}{s['mean']:>14.4g}"
            f"  {flag}  (expect {lo}..{hi})"
        )

    # Limited-domain sanity:
    print()
    print("Limited-domain checks:")
    print("  sst: NaN over land (ERA5 fill); siconc and soil/snow vars = 0 over off-domain")
    for v in ("sst", "siconc", "swvl1", "stl1", "sd"):
        if v not in ds.data_vars:
            continue
        s = _summary(ds[v].values)
        print(f"  {v}: pct_nan={s.get('pct_nan', 0.0):.1f}%, "
              f"min={s.get('min', float('nan')):.4g}, "
              f"max={s.get('max', float('nan')):.4g}, "
              f"mean={s.get('mean', float('nan')):.4g}")

    print()
    if fails:
        print(f"FAIL: {fails} issues")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
