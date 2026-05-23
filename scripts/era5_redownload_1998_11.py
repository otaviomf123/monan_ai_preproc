#!/usr/bin/env python3
"""One-shot re-download of era5_pressure_additional_levels_1998_11.nc.

Original CDS request failed on 2026-04-24 with a transient MARS server error
("Failed to write chunked newline (Broken pipe)"). The .status was marked as
errored but the request was never re-issued. This script retries it.

Run from /mnt/beegfs/otavio.feitosa/era5/ in the `tf` conda env.
"""
import time
from pathlib import Path

import cdsapi
import xarray as xr


def main() -> None:
    target = Path("data_additional_levels/1998/era5_pressure_additional_levels_1998_11.nc")
    target_tmp = target.with_suffix(".nc.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_tmp.exists():
        target_tmp.unlink()

    print("Re-downloading additional_levels 1998-11 (5 vars × 2 levels)")
    print(f"  target: {target}")

    t0 = time.time()
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "relative_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": ["250", "950"],
            "year":  "1998",
            "month": "11",
            "day":   [f"{d:02d}" for d in range(1, 31)],
            "time":  ["00:00", "06:00", "12:00", "18:00"],
            "grid":  ["1.0", "1.0"],
            "format": "netcdf",
        },
        str(target_tmp),
    )

    sz = target_tmp.stat().st_size / 1e6
    target_tmp.rename(target)
    elapsed = time.time() - t0
    print(f"DONE: {target} ({sz:.1f} MB) in {elapsed:.1f}s")

    # Verify
    print("Verifying contents:")
    ds = xr.open_dataset(target)
    vs = sorted(list(ds.data_vars))
    levels = sorted([int(x) for x in ds.pressure_level.values])
    tdim = "valid_time" if "valid_time" in ds.sizes else "time"
    n_t = ds.sizes.get(tdim, "?")
    print(f"  vars   : {vs}")
    print(f"  levels : {levels}")
    print(f"  n_time : {n_t}")
    ds.close()


if __name__ == "__main__":
    main()
