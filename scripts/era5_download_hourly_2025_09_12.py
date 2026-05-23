#!/usr/bin/env python3
"""Targeted download of 2025-09 .. 2025-12 hourly files (7 file types × 4 months
= 28 CDS requests).

The general `era5_download_hourly.py` script has a hardcoded
    end_month = 8 if year == 2025 else 12
which was correct when written (Aug 2025 was the latest), but is now outdated.
This script fills the gap surgically without modifying the original.

Safety: atomic .tmp → rename only on success. Run from /mnt/beegfs/otavio.feitosa/era5/.
"""
from __future__ import annotations

import calendar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Semaphore

import cdsapi


HOURLY_GROUPS = {
    "precipitation": [
        "total_precipitation",
    ],
    "evaporation": [
        "evaporation",
        "potential_evaporation",
    ],
    "heat_flux": [
        "surface_sensible_heat_flux",
        "surface_latent_heat_flux",
    ],
    "radiation_solar": [
        "surface_net_solar_radiation",
        "surface_solar_radiation_downwards",
        "top_net_solar_radiation",
        "toa_incident_solar_radiation",
        "surface_solar_radiation_downward_clear_sky",
    ],
    "radiation_thermal": [
        "surface_net_thermal_radiation",
        "surface_thermal_radiation_downwards",
        "top_net_thermal_radiation",
        "surface_thermal_radiation_downward_clear_sky",
    ],
    "runoff": [
        "surface_runoff",
        "sub_surface_runoff",
        "runoff",
    ],
    "snow": [
        "snowfall",
        "snowmelt",
    ],
}

YEAR = 2025
MONTHS = [9, 10, 11, 12]
MAX_PARALLEL = 8  # hourly files are big; be conservative
HOURS_24 = [f"{h:02d}:00" for h in range(24)]


def download_one(year, month, group, vars_, sem, client, log):
    sem.acquire()
    tag = f"{year}-{month:02d} {group}"
    target = Path(str(year)) / f"era5_surface_hourly_{group}_{year}_{month:02d}.nc"
    target_tmp = target.with_suffix(".nc.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_tmp.exists():
        target_tmp.unlink()
    try:
        if target.exists():
            log(f"[{tag}] already exists — skipping")
            return tag, True, "exists"
        n_days = calendar.monthrange(year, month)[1]
        log(f"[{tag}] requesting {len(vars_)} vars × 24h × {n_days}d from CDS …")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     vars_,
                "year":         str(year),
                "month":        f"{month:02d}",
                "day":          [f"{d:02d}" for d in range(1, n_days + 1)],
                "time":         HOURS_24,
                "grid":         ["1.0", "1.0"],
                "format":       "netcdf",
            },
            str(target_tmp),
        )
        sz = target_tmp.stat().st_size / 1e6
        target_tmp.rename(target)
        log(f"[{tag}] DONE ({sz:.1f} MB)")
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
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"hourly_2025_09_12_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s %(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    log = logging.getLogger().info
    log("=" * 60)
    log(f"ERA5 hourly fill: 2025-09..12 ({len(MONTHS)} months × {len(HOURLY_GROUPS)} groups)")
    log("=" * 60)

    tasks = []
    for m in MONTHS:
        for group, vars_ in HOURLY_GROUPS.items():
            tasks.append((YEAR, m, group, vars_))
    log(f"Total tasks: {len(tasks)} (max {MAX_PARALLEL} concurrent)")

    sem = Semaphore(MAX_PARALLEL)
    client = cdsapi.Client()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = [ex.submit(download_one, y, m, g, v, sem, client, log)
                for (y, m, g, v) in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_skip = sum(1 for _, ok, why in results if ok and why == "exists")
    log("=" * 60)
    log(f"FINAL: {n_ok}/{len(results)} OK ({n_skip} were skipped existing), "
        f"{len(results)-n_ok} failed")
    for tag, ok, err in results:
        if not ok:
            log(f"  {tag}: {err}")


if __name__ == "__main__":
    main()
