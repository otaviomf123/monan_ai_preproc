#!/usr/bin/env python3
"""Re-download 3 wind files WITHOUT i10fg.

Previous redownload (era5_redownload_fixes.py) requested all 5 wind vars
including 10m_wind_gust_since_previous_post_processing (i10fg). Mixing this
"accumulator-like" variable with pure-instantaneous u/v components triggered
CDS to package the response as a ZIP archive instead of a single NetCDF —
which xarray cannot read.

Since i10fg is excluded from the v3 schema (6h-window max gust, not a
snapshot), we just redownload these 3 wind files with only u10/v10/u100/v100.

Run from /mnt/beegfs/otavio.feitosa/era5/ in the `tf` conda env.
"""
from __future__ import annotations

import calendar
import time
from pathlib import Path

import cdsapi


WIND_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]

TARGETS = [
    (1954, 5),
    (1968, 11),
    (1972, 9),
]


def main() -> None:
    client = cdsapi.Client()
    for year, month in TARGETS:
        n_days = calendar.monthrange(year, month)[1]
        target = Path(f"{year}/era5_surface_wind_{year}_{month:02d}.nc")
        target_tmp = target.with_suffix(".nc.tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target_tmp.exists():
            target_tmp.unlink()

        print(f"[{year}-{month:02d}] requesting 4 wind vars (no i10fg) → .tmp ...")
        t0 = time.time()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     WIND_VARS,
                "year":         str(year),
                "month":        f"{month:02d}",
                "day":          [f"{d:02d}" for d in range(1, n_days + 1)],
                "time":         ["00:00", "06:00", "12:00", "18:00"],
                "grid":         ["1.0", "1.0"],
                "format":       "netcdf",
            },
            str(target_tmp),
        )
        sz = target_tmp.stat().st_size / 1e6
        if target.exists():
            target.unlink()
        target_tmp.rename(target)
        elapsed = time.time() - t0
        print(f"[{year}-{month:02d}] DONE: {target} ({sz:.1f} MB) in {elapsed:.1f}s")

        # Verify it's NetCDF, not ZIP
        import xarray as xr
        ds = xr.open_dataset(target)
        vs = sorted(list(ds.data_vars))
        print(f"  vars verified: {vs}")
        ds.close()


if __name__ == "__main__":
    main()
