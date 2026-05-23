#!/usr/bin/env python3
"""
Re-download the 18 ERA5 surface files that have INCOMPLETE variable sets.

Problem: 8 specific months were downloaded by an older version of
`era5_download.py` that requested fewer variables per file_type than the
current schema. Files exist on disk and open OK, but lack vars like cbh,
u100, v100, the convective indices (cape/cin/kx/totalx), or the column
hydrometeor integrals (tciw/tclw/tco3/tcrw/tcsw).

Strategy:
  1. Delete the broken file (so we don't carry stale partial data forward).
  2. Issue a fresh CDS request with the FULL var set for that file_type
     (matching what newer 2010+ files contain).
  3. Save to the original path.

Run from the cluster's `/mnt/beegfs/otavio.feitosa/era5/` directory (the
output dir convention) using the `tf` conda env that has `cdsapi`.
"""
from __future__ import annotations

import calendar
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Semaphore

import cdsapi


# ---------------------------------------------------------------------------
# Where to write (relative to script execution dir)
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(".")

# CDS short-name → full variable set per file_type (match newer working files)
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
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "10m_wind_gust_since_previous_post_processing",
    ],
}

# (year, month) → list of file_types that need re-download for that month
TODO = {
    (1954, 3):  ["boundary_layer", "clouds",      "water_column"],
    (1954, 4):  ["boundary_layer",                "water_column"],
    (1954, 5):  [                                                "wind"],
    (1956, 8):  ["boundary_layer", "clouds",      "water_column"],
    (1962, 7):  ["boundary_layer", "clouds",      "water_column"],
    (1968, 11): [                                                "wind"],
    (1972, 9):  ["boundary_layer", "clouds",      "water_column", "wind"],
    (1974, 10): [                  "clouds"],
}

MAX_PARALLEL = 5


# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"era5_redownload_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-10s %(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def download_one(year: int, month: int, group: str,
                 sem: Semaphore, client: cdsapi.Client,
                 logger: logging.Logger) -> tuple[str, bool, str]:
    """Re-download with the full var set; safe against partial failures.

    SAFETY: download to <target>.tmp first. Only after the request succeeds
    do we replace the existing broken file (atomic rename). If CDS errors
    out (403, network, quota), the original file is untouched.
    """
    sem.acquire()
    tag = f"{year}-{month:02d} {group}"
    try:
        n_days = calendar.monthrange(year, month)[1]
        days = [f"{d:02d}" for d in range(1, n_days + 1)]
        year_str, month_str = str(year), f"{month:02d}"
        out_dir = OUTPUT_ROOT / year_str
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"era5_surface_{group}_{year_str}_{month_str}.nc"
        target_tmp = target.with_suffix(".nc.tmp")

        request = {
            "product_type": "reanalysis",
            "variable":     GROUPS[group],
            "year":         year_str,
            "month":        month_str,
            "day":          days,
            "time":         ["00:00", "06:00", "12:00", "18:00"],
            "grid":         ["1.0", "1.0"],
            "format":       "netcdf",
        }
        logger.info(f"[{tag}] requesting {len(GROUPS[group])} vars from CDS → .tmp ...")
        # Clean any leftover from a prior aborted run
        if target_tmp.exists():
            target_tmp.unlink()

        client.retrieve("reanalysis-era5-single-levels", request, str(target_tmp))

        # Only swap if download succeeded
        new_sz = target_tmp.stat().st_size / 1e6
        if target.exists():
            old_sz = target.stat().st_size / 1e6
            target.unlink()
            logger.info(f"[{tag}] replacing old file ({old_sz:.1f} MB → {new_sz:.1f} MB)")
        target_tmp.rename(target)
        logger.info(f"[{tag}] DONE ({new_sz:.1f} MB)")
        return (tag, True, "")
    except Exception as e:
        logger.error(f"[{tag}] FAIL: {e}")
        # Clean up any partial .tmp so re-runs aren't confused
        try:
            if target_tmp.exists():
                target_tmp.unlink()
        except Exception:
            pass
        return (tag, False, str(e))
    finally:
        sem.release()


def main() -> None:
    logger = setup_logging()
    logger.info("=" * 72)
    logger.info("ERA5 re-download of 18 broken surface files (8 months × var groups)")
    logger.info("=" * 72)

    # Flatten task list
    tasks: list[tuple[int, int, str]] = []
    for (y, m), groups in TODO.items():
        for g in groups:
            tasks.append((y, m, g))
    logger.info(f"Total tasks: {len(tasks)} (max {MAX_PARALLEL} concurrent CDS requests)")

    sem = Semaphore(MAX_PARALLEL)
    client = cdsapi.Client()

    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = [ex.submit(download_one, y, m, g, sem, client, logger)
                   for (y, m, g) in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_err = len(results) - n_ok
    logger.info("=" * 72)
    logger.info(f"FINAL: {n_ok}/{len(results)} OK, {n_err} failed")
    if n_err:
        logger.info("Failures:")
        for tag, ok, err in results:
            if not ok:
                logger.info(f"  {tag}: {err}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
