#!/usr/bin/env python3
"""Targeted re-download of 6 broken surface source files identified during
the 2001-2008 conversion (job 66270). Same pattern as the original 8 broken
files in 1950-2000: incomplete var sets from older download script.

Safe: download to <target>.tmp first; only rename to final name on success
(does NOT delete the broken file unless the new one downloaded OK).

Wind request EXCLUDES i10fg to prevent CDS from packaging as ZIP.

Run from /mnt/beegfs/otavio.feitosa/era5/ in conda env `tf`.
"""
from __future__ import annotations

import calendar
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore

import cdsapi


GROUPS = {
    "boundary_layer": [
        "boundary_layer_height",
        "convective_available_potential_energy",
        "convective_inhibition",
        "k_index",
        "total_totals_index",
    ],
    "clouds": [
        "cloud_base_height",
        "high_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "total_cloud_cover",
    ],
    "water_column": [
        "total_column_cloud_ice_water",
        "total_column_cloud_liquid_water",
        "total_column_ozone",
        "total_column_rain_water",
        "total_column_snow_water",
        "total_column_water",
        "total_column_water_vapour",
    ],
    "wind": [
        # NO i10fg — mixing with instantaneous u/v triggers CDS ZIP packaging.
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
    ],
}

TODO = {
    (2002, 6):  ["boundary_layer",            "water_column"],
    (2007, 4):  ["boundary_layer", "clouds",  "water_column"],
    (2008, 12): [                                            "wind"],
}

OUTPUT_ROOT = Path(".")
MAX_PARALLEL = 5


def download_one(year, month, group, sem, client, log):
    sem.acquire()
    tag = f"{year}-{month:02d} {group}"
    target = OUTPUT_ROOT / str(year) / f"era5_surface_{group}_{year}_{month:02d}.nc"
    target_tmp = target.with_suffix(".nc.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_tmp.exists():
        target_tmp.unlink()
    try:
        n_days = calendar.monthrange(year, month)[1]
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     GROUPS[group],
                "year":         str(year),
                "month":        f"{month:02d}",
                "day":          [f"{d:02d}" for d in range(1, n_days + 1)],
                "time":         ["00:00", "06:00", "12:00", "18:00"],
                "grid":         ["1.0", "1.0"],
                "format":       "netcdf",
            },
            str(target_tmp),
        )
        new_sz = target_tmp.stat().st_size / 1e6
        if target.exists():
            target.unlink()
        target_tmp.rename(target)
        log(f"[{tag}] DONE ({new_sz:.1f} MB, {len(GROUPS[group])} vars)")
        return tag, True, ""
    except Exception as e:
        log(f"[{tag}] FAIL: {e}")
        try:
            if target_tmp.exists():
                target_tmp.unlink()
        except Exception:
            pass
        return tag, False, str(e)
    finally:
        sem.release()


def main():
    import logging
    from datetime import datetime
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"era5_redownload_fixes_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s %(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    log = logging.getLogger().info

    log("=" * 60)
    log(f"ERA5 re-download fixes round 2 (6 broken sources for 2001-2008)")
    log("=" * 60)

    tasks = [(y, m, g) for (y, m), gs in TODO.items() for g in gs]
    log(f"Total: {len(tasks)} requests, max {MAX_PARALLEL} concurrent")

    sem = Semaphore(MAX_PARALLEL)
    client = cdsapi.Client()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = [ex.submit(download_one, y, m, g, sem, client, log)
                for y, m, g in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    n_ok = sum(1 for _, ok, _ in results if ok)
    log("=" * 60)
    log(f"FINAL: {n_ok}/{len(results)} OK, {len(results)-n_ok} failed")
    if n_ok < len(results):
        for tag, ok, err in results:
            if not ok:
                log(f"  {tag}: {err}")


if __name__ == "__main__":
    main()
