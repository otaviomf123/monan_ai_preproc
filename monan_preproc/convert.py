"""Per-month ERA5 → HEALPix nside=64 conversion.

Pipeline for one (year, month):

    1. Load all relevant raw NetCDF files (instantaneous surface, accumulated
       hourly surface, pressure levels, additional levels, hydrometeors, ozone
       extended).
    2. Regrid each variable from regular 1° lat-lon to HEALPix nside=256
       (BallTree nearest-neighbor).
    3. Degrade nside=256 → nside=64 hierarchically with `nanmean` over the 4
       NESTED children at each step (256 → 128 → 64). This preserves
       limited-domain variables (sst, soil moisture, sea ice) cleanly because
       ocean/land masks propagate as NaN through the averaging.
    4. Expand pressure variables `r/t/u/v/z` over the canonical 17-level grid
       (combines main pressure files + additional_levels for 250+950).
    5. Concatenate o3: stratosphere (10/30/50) + ozone_extended (14 trop levels).
    6. Filter hydrometeor levels per `HYDROMETEOR_KEEP_LEVELS`.
    7. Sum hourly accumulators over each [t-6h, t] window with previous-month
       boundary handling (factor applied per variable: m → mm or J/m² → W/m²).
    8. Write a single NetCDF per month with all variables on HEALPix nside=64,
       RING ordering (default; NESTED selectable via `nest=True`), plus metadata.

    Strict schema: the output must contain ALL `catalog.total_var_count()`
    expected variables. If any source file is missing/incomplete, the call
    raises RuntimeError and writes nothing — preserving dataset uniformity.

Note: derived variables (e.g., q from r/t/p, psfc from msl+z, skin_T composite)
are intentionally NOT computed here. They are produced on-the-fly downstream
(model dataloader, PyCRTM caller, etc.) so the preproc output stays a faithful
projection of the raw ERA5 fields onto HEALPix.
"""
from __future__ import annotations

import calendar
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import healpy as hp
import numpy as np
import xarray as xr

from . import catalog
from .healpix import HierarchicalDegrader, LatLonToHEALPix, healpix_coords

logger = logging.getLogger(__name__)

NSIDE_HIGH = 256
NSIDE_LOW = 64
DROP_VARS_GLOBAL = {"expver", "number"}

# Provenance string applied to every output variable.
_REGRID_NOTE = (
    "BallTree NN regular 1°→nside=256, then nested-nanmean degrade →nside=64"
)


def _capture_attrs(da: "xr.DataArray", extras: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Pull printable attrs from the source DataArray and merge with extras.

    Anything coercible to str is kept (units, long_name, standard_name,
    GRIB_paramId, etc.). The mandatory `regrid` provenance attr is always
    set unless overridden in `extras`.
    """
    out: Dict[str, str] = {}
    for k, v in da.attrs.items():
        try:
            out[k] = v if isinstance(v, str) else str(v)
        except Exception:
            continue
    if extras:
        out.update({k: (v if isinstance(v, str) else str(v)) for k, v in extras.items()})
    out.setdefault("regrid", _REGRID_NOTE)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    drop = [v for v in ds.variables if ds[v].dtype == object or v in DROP_VARS_GLOBAL]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def _time_dim(ds: xr.Dataset) -> str:
    for n in ("valid_time", "time"):
        if n in ds.dims:
            return n
    raise ValueError(f"No time dimension in {list(ds.dims)}")


def _latlon(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    lat = ds["latitude"].values if "latitude" in ds else ds["lat"].values
    lon = ds["longitude"].values if "longitude" in ds else ds["lon"].values
    return lat, lon


def _level_dim(ds: xr.Dataset) -> Optional[str]:
    for n in ("pressure_level", "level"):
        if n in ds.dims:
            return n
    return None


def target_times_for(year: int, month: int) -> np.ndarray:
    """Return 6h target timestamps (00/06/12/18Z) covering the given month."""
    n_days = calendar.monthrange(year, month)[1]
    out = []
    for day in range(1, n_days + 1):
        for hour in (0, 6, 12, 18):
            out.append(np.datetime64(f"{year}-{month:02d}-{day:02d}T{hour:02d}:00"))
    return np.array(out)


def _select_6h(
    da: xr.DataArray, times: np.ndarray, target_times: np.ndarray, tol_minutes: int = 30
) -> np.ndarray:
    """Select rows from da matching target_times (within `tol_minutes`)."""
    out = np.full((len(target_times),) + da.shape[1:], np.nan, dtype=np.float32)
    for i, t6h in enumerate(target_times):
        idx = np.where(times == t6h)[0]
        if len(idx) == 0:
            diffs = np.abs((times - t6h).astype("timedelta64[m]").astype(float))
            if diffs.size > 0:
                k = int(np.argmin(diffs))
                if diffs[k] <= tol_minutes:
                    idx = [k]
        if len(idx):
            out[i] = da.values[idx[0]].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Pressure (main + additional levels), o3 (strato + extended), hydrometeors
# ---------------------------------------------------------------------------


def _process_pressure_file(
    path: Path,
    target_times: np.ndarray,
    regridder: LatLonToHEALPix,
    degrader: HierarchicalDegrader,
    keep_vars: Sequence[str],
    keep_levels: Sequence[int],
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], Dict[str, str]]]:
    """Open one pressure-level NetCDF, regrid+degrade vars at requested levels.

    Returns
    -------
    data : dict (var, level) → (n_times, npix_low)
    attrs : dict (var, level) → CF-style attrs (source `units`, `long_name`,
            `standard_name`, etc., plus `pressure_level` and `regrid`).
    """
    out_data: Dict[Tuple[str, int], np.ndarray] = {}
    out_attrs: Dict[Tuple[str, int], Dict[str, str]] = {}
    ds = _open(path)
    plev = _level_dim(ds)
    if plev is None:
        ds.close()
        return out_data, out_attrs
    times = ds[_time_dim(ds)].values
    lat, lon = _latlon(ds)
    file_levels = [int(x) for x in ds[plev].values]
    for v in keep_vars:
        if v not in ds.data_vars:
            continue
        da = ds[v]
        if plev not in da.dims:
            continue
        base_attrs = _capture_attrs(da)
        for lvl in keep_levels:
            if lvl not in file_levels:
                continue
            data = _select_6h(da.sel({plev: lvl}), times, target_times)  # (T, nlat, nlon)
            grid_high = regridder.regrid(data, lat, lon)                 # (T, npix_256)
            grid_low = degrader(grid_high, pix_axis=-1)                  # (T, npix_64)
            out_data[(v, int(lvl))] = grid_low
            attrs = dict(base_attrs)
            attrs["pressure_level"] = f"{lvl} hPa"
            if "long_name" in attrs:
                attrs["long_name"] = f"{attrs['long_name']} at {lvl} hPa"
            out_attrs[(v, int(lvl))] = attrs
    ds.close()
    return out_data, out_attrs


def process_pressure(
    year: int,
    month: int,
    target_times: np.ndarray,
    raw_dir: Path,
    additional_dir: Path,
    regridder: LatLonToHEALPix,
    degrader: HierarchicalDegrader,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    """Process pressure-level dynamics + strato o3 over the available levels.

    Returns dict var_NNNhpa → (T, npix_low) and matching attrs dict for r/t/u/v/z
    (15 main + 2 additional) and o3 (3 strato levels). Tropospheric/UT o3 comes
    from process_o3_extended.
    """
    out_data: Dict[str, np.ndarray] = {}
    out_attrs: Dict[str, Dict[str, str]] = {}

    # ── main pressure files (15 levels, includes o3 only at stratosphere) ──
    for stem, levels in catalog.PRESSURE_LEVEL_FILES.items():
        path = raw_dir / f"{stem}_{year}_{month:02d}.nc"
        if not path.exists():
            logger.warning("missing pressure file: %s", path.name)
            continue
        keep_vars = list(catalog.PRESSURE_VARS)
        if "stratosphere" in stem:
            keep_vars.append("o3")
        d, a = _process_pressure_file(
            path, target_times, regridder, degrader, keep_vars, levels
        )
        for (v, lvl), arr in d.items():
            name = f"{v}_{lvl}hpa"
            out_data[name] = arr
            out_attrs[name] = a[(v, lvl)]

    # ── additional levels (250 + 950) ──
    add_path = additional_dir / f"{catalog.ADDITIONAL_LEVELS_FILE}_{year}_{month:02d}.nc"
    if add_path.exists():
        d, a = _process_pressure_file(
            add_path,
            target_times,
            regridder,
            degrader,
            catalog.PRESSURE_VARS,
            catalog.ADDITIONAL_LEVELS,
        )
        for (v, lvl), arr in d.items():
            name = f"{v}_{lvl}hpa"
            out_data[name] = arr
            out_attrs[name] = a[(v, lvl)]
    else:
        logger.warning("missing additional_levels: %s", add_path.name)

    return out_data, out_attrs


def process_o3_extended(
    year: int,
    month: int,
    target_times: np.ndarray,
    ozone_dir: Path,
    regridder: LatLonToHEALPix,
    degrader: HierarchicalDegrader,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    """Process tropospheric/UT ozone (14 levels) into o3_NNNhpa entries."""
    out_data: Dict[str, np.ndarray] = {}
    out_attrs: Dict[str, Dict[str, str]] = {}
    path = ozone_dir / f"{catalog.OZONE_EXTENDED_FILE}_{year}_{month:02d}.nc"
    if not path.exists():
        logger.warning("missing ozone_extended: %s (will leave non-strato o3 levels empty)",
                       path.name)
        return out_data, out_attrs
    ds = _open(path)
    plev = _level_dim(ds)
    if plev is None:
        ds.close()
        return out_data, out_attrs
    times = ds[_time_dim(ds)].values
    lat, lon = _latlon(ds)
    # Ozone CDS variable: ozone_mass_mixing_ratio → short name 'o3'
    if "o3" not in ds.data_vars:
        ds.close()
        return out_data, out_attrs
    da = ds["o3"]
    base_attrs = _capture_attrs(da)
    file_levels = [int(x) for x in ds[plev].values]
    for lvl in catalog.OZONE_EXTENDED_LEVELS:
        if lvl not in file_levels:
            continue
        data = _select_6h(da.sel({plev: lvl}), times, target_times)
        grid_high = regridder.regrid(data, lat, lon)
        name = f"o3_{lvl}hpa"
        out_data[name] = degrader(grid_high, pix_axis=-1)
        attrs = dict(base_attrs)
        attrs["pressure_level"] = f"{lvl} hPa"
        if "long_name" in attrs:
            attrs["long_name"] = f"{attrs['long_name']} at {lvl} hPa"
        out_attrs[name] = attrs
    ds.close()
    return out_data, out_attrs


def process_hydrometeors(
    year: int,
    month: int,
    target_times: np.ndarray,
    hydromet_dir: Path,
    regridder: LatLonToHEALPix,
    degrader: HierarchicalDegrader,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    """Process crwc/cswc/clwc/ciwc/cc, dropping always-zero (var, level) combos."""
    out_data: Dict[str, np.ndarray] = {}
    out_attrs: Dict[str, Dict[str, str]] = {}
    for stem, vars_in_file in catalog.HYDROMETEOR_FILES.items():
        path = hydromet_dir / f"{stem}_{year}_{month:02d}.nc"
        if not path.exists():
            logger.warning("missing hydrometeor file: %s", path.name)
            continue
        for v in vars_in_file:
            keep = catalog.HYDROMETEOR_KEEP_LEVELS[v]
            d, a = _process_pressure_file(
                path, target_times, regridder, degrader, [v], keep
            )
            for (vv, lvl), arr in d.items():
                name = f"{vv}_{lvl}hpa"
                out_data[name] = arr
                attrs = dict(a[(vv, lvl)])
                # Document the physical level cutoff applied to clwc/crwc
                if vv in ("clwc", "crwc"):
                    n_kept = len(catalog.HYDROMETEOR_KEEP_LEVELS[vv])
                    attrs["level_filter"] = (
                        f"kept {n_kept}/17 levels; high levels excluded "
                        "(no liquid water at T < -38 deg C)"
                    )
                out_attrs[name] = attrs
    return out_data, out_attrs


# ---------------------------------------------------------------------------
# Surface — instantaneous and accumulated
# ---------------------------------------------------------------------------


def process_surface_instantaneous(
    year: int,
    month: int,
    target_times: np.ndarray,
    raw_dir: Path,
    regridder: LatLonToHEALPix,
    degrader: HierarchicalDegrader,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    out_data: Dict[str, np.ndarray] = {}
    out_attrs: Dict[str, Dict[str, str]] = {}
    for stem, keep_vars in catalog.SURFACE_INSTANTANEOUS.items():
        path = raw_dir / f"{stem}_{year}_{month:02d}.nc"
        if not path.exists():
            logger.warning("missing surface file: %s", path.name)
            continue
        ds = _open(path)
        times = ds[_time_dim(ds)].values
        lat, lon = _latlon(ds)
        for v in keep_vars:
            if v not in ds.data_vars:
                continue
            data = _select_6h(ds[v], times, target_times)
            grid_high = regridder.regrid(data, lat, lon)
            out_data[v] = degrader(grid_high, pix_axis=-1)
            out_attrs[v] = _capture_attrs(ds[v])
        ds.close()
    return out_data, out_attrs


def process_surface_accumulated(
    year: int,
    month: int,
    target_times: np.ndarray,
    raw_dir: Path,
    regridder: LatLonToHEALPix,
    degrader: HierarchicalDegrader,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    """Sum hourly accumulators over each [t-6h, t] window per target_time.

    Falls back to previous month's last hours for the first 06Z target.
    Output `attrs` reflect the post-conversion units (mm or W m**-2) and
    document the accumulation step.
    """
    out_data: Dict[str, np.ndarray] = {}
    out_attrs: Dict[str, Dict[str, str]] = {}
    if month == 1:
        py, pm = year - 1, 12
    else:
        py, pm = year, month - 1
    cutoff = np.datetime64(
        f"{py}-{pm:02d}-{calendar.monthrange(py, pm)[1]:02d}T19:00"
    )

    for stem, var_specs in catalog.SURFACE_ACCUM_FILES.items():
        path = raw_dir / f"{stem}_{year}_{month:02d}.nc"
        if not path.exists():
            logger.warning("missing accum file: %s", path.name)
            continue
        ds = _open(path)
        tdim = _time_dim(ds)
        t_curr = ds[tdim].values
        lat, lon = _latlon(ds)

        prev = raw_dir.parent / str(py) / f"{stem}_{py}_{pm:02d}.nc"
        if prev.exists():
            ds_prev = _open(prev)
            tdim_p = _time_dim(ds_prev)
            t_prev = ds_prev[tdim_p].values
            mask = t_prev >= cutoff
            t_prev = t_prev[mask]
        else:
            ds_prev = None
            t_prev = np.array([], dtype="datetime64[ns]")

        for v, spec in var_specs.items():
            if v not in ds.data_vars:
                continue
            factor = float(spec["factor"])
            output_units = str(spec["units"])
            accum_note = str(spec["accumulation"])

            arr_curr = ds[v].values
            if t_prev.size and ds_prev is not None and v in ds_prev.data_vars:
                arr_prev = ds_prev[v].isel({tdim_p: mask}).values
                all_t = np.concatenate([t_prev, t_curr])
                all_x = np.concatenate([arr_prev, arr_curr], axis=0)
            else:
                all_t = t_curr
                all_x = arr_curr

            final_npix = hp.nside2npix(degrader.nside_low)
            res_low = np.full((len(target_times), final_npix), np.nan, dtype=np.float32)

            for i, t6h in enumerate(target_times):
                t_start = t6h - np.timedelta64(5, "h")
                m = (all_t >= t_start) & (all_t <= t6h)
                if not m.any():
                    continue
                summed = all_x[m].sum(axis=0, keepdims=True)
                grid_high = regridder.regrid(summed, lat, lon)
                grid_low = degrader(grid_high, pix_axis=-1)
                res_low[i] = (grid_low[0] * factor).astype(np.float32)
            out_data[v] = res_low

            # Capture source attrs and OVERRIDE units to reflect conversion.
            attrs = _capture_attrs(
                ds[v],
                extras={
                    "units": output_units,
                    "accumulation": accum_note,
                    "time_window": "[t - 6h, t]",
                    "cell_methods": "time: sum" if "mm" in output_units else "time: mean",
                },
            )
            # Keep the ORIGINAL units alongside, for traceability.
            if "units" in ds[v].attrs:
                attrs["original_units"] = str(ds[v].attrs["units"])
            out_attrs[v] = attrs
        ds.close()
        if ds_prev is not None:
            ds_prev.close()
    return out_data, out_attrs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_output(
    out_path: Path,
    target_times: np.ndarray,
    all_vars: Dict[str, np.ndarray],
    all_attrs: Optional[Dict[str, Dict[str, str]]] = None,
    nside: int = NSIDE_LOW,
    nest: bool = False,
    extra_attrs: Optional[Dict] = None,
) -> None:
    """Persist the processed month as a single compressed NetCDF.

    Layout: dims=(time, pixel), one variable per array, NESTED ordering,
    HEALPix lat/lon as auxiliary coordinates. Per-variable attrs (units,
    long_name, standard_name, pressure_level, accumulation note, etc.) are
    attached when provided in `all_attrs`.
    """
    npix = hp.nside2npix(nside)
    hp_lat, hp_lon = healpix_coords(nside, nest=nest)
    ordering_str = "NESTED" if nest else "RING"

    ds = xr.Dataset(
        data_vars={name: (("time", "pixel"), arr) for name, arr in sorted(all_vars.items())},
        coords={
            "time": target_times,
            "pixel": np.arange(npix, dtype=np.int32),
            "healpix_lat": ("pixel", hp_lat),
            "healpix_lon": ("pixel", hp_lon),
        },
        attrs={
            "nside": int(nside),
            "npix": int(npix),
            "ordering": ordering_str,
            "regrid_method": _REGRID_NOTE,
            "n_variables": int(len(all_vars)),
            "creation_date": datetime.utcnow().isoformat() + "Z",
            "source": "ERA5 (CDS) — surface + pressure + additional_levels + "
                      "hydrometeors + ozone_extended",
            "frequency": "6h instantaneous (00/06/12/18Z); accumulators summed over [t-6h, t]",
            "package": "monan_ia_preproc 0.1.0",
            "Conventions": "CF-1.8",
        },
    )
    if extra_attrs:
        ds.attrs.update(extra_attrs)

    # Per-variable attrs (preserves source CDS metadata + pipeline notes)
    if all_attrs:
        for name, attrs in all_attrs.items():
            if name in ds.data_vars:
                ds[name].attrs.update(attrs)

    # Coord attrs
    ds["time"].attrs.update({
        "long_name": "Valid time",
        "standard_name": "time",
    })
    ds["pixel"].attrs.update({
        "long_name": "HEALPix pixel index",
        "ordering": ordering_str,
        "nside": int(nside),
    })
    ds["healpix_lat"].attrs.update({"units": "degrees_north", "long_name": "HEALPix pixel latitude",
                                     "standard_name": "latitude"})
    ds["healpix_lon"].attrs.update({"units": "degrees_east",  "long_name": "HEALPix pixel longitude",
                                     "standard_name": "longitude"})

    # Compression / chunking tuned for the model dataloader access pattern
    # (read a small contiguous time window per sample):
    #   - zlib level 5 + byte shuffle: ~50–55% ratio on float32, decompression
    #     speed is roughly independent of level (5 ≈ 4 in read cost) so we
    #     trade a tiny write-time bump for a slightly better archive size.
    #   - chunksizes=(4, npix): one day (4 6h-steps) per chunk → reading any
    #     single timestep loads ~768 KB instead of the full ~24 MB variable.
    #     Balanced for both autoregressive 1-step and short-rollout windows.
    n_times = len(target_times)
    chunk_t = min(4, n_times)
    encoding = {
        v: {
            "zlib": True,
            "complevel": 5,
            "shuffle": True,
            "dtype": "float32",
            "chunksizes": (chunk_t, npix),
        }
        for v in ds.data_vars
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=encoding, format="NETCDF4", engine="netcdf4")
    ds.close()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def convert_month(
    year: int,
    month: int,
    raw_root: Path,
    out_root: Path,
    nside_high: int = NSIDE_HIGH,
    nside_low: int = NSIDE_LOW,
    nest: bool = False,
    overwrite: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Convert one month to HEALPix nside_low and write a single NetCDF.

    raw_root layout expected
    ------------------------
        {raw_root}/{YYYY}/era5_*_{YYYY}_{MM}.nc          (main)
        {raw_root}/data_additional_levels/{YYYY}/...     (250+950)
        {raw_root}/data_hydrometeors/{YYYY}/...          (crwc/cswc/clwc/ciwc/cc)
        {raw_root}/data_ozone_extended/{YYYY}/...        (o3 14 levels)

    Ordering defaults to RING (`nest=False`) — matches `monan_ai` model and
    DA tooling. Set `nest=True` for NESTED ordering.

    `cache_dir` (default `out_root/_cache/`) holds the BallTree regrid index
    and hierarchical-degrade children mapping — computed once across all
    parallel workers, then reloaded.

    Output: {out_root}/{YYYY}/{YYYY}_{MM}.nc
    """
    out_path = out_root / str(year) / f"{year}_{month:02d}.nc"
    if out_path.exists() and not overwrite:
        logger.info("skip existing: %s", out_path)
        return out_path

    target_times = target_times_for(year, month)
    logger.info("[%d-%02d] target=%d 6h timestamps", year, month, len(target_times))

    if cache_dir is None:
        cache_dir = out_root / "_cache"

    regridder = LatLonToHEALPix(nside=nside_high, nest=nest, cache_dir=cache_dir)
    degrader = HierarchicalDegrader(nside_high=nside_high, nside_low=nside_low,
                                     nest=nest, cache_dir=cache_dir)

    main_dir = raw_root / str(year)
    add_dir = raw_root / "data_additional_levels" / str(year)
    hyd_dir = raw_root / "data_hydrometeors" / str(year)
    o3_dir = raw_root / "data_ozone_extended" / str(year)

    logger.info("[%d-%02d] processing pressure dynamics + strato o3 ...", year, month)
    pressure_named, pressure_attrs = process_pressure(
        year, month, target_times, main_dir, add_dir, regridder, degrader
    )
    logger.info("  pressure vars: %d", len(pressure_named))

    logger.info("[%d-%02d] processing o3 extended ...", year, month)
    o3_named, o3_attrs = process_o3_extended(
        year, month, target_times, o3_dir, regridder, degrader
    )
    logger.info("  o3 extended vars: %d", len(o3_named))

    logger.info("[%d-%02d] processing hydrometeors + cloud cover ...", year, month)
    hydromet, hydromet_attrs = process_hydrometeors(
        year, month, target_times, hyd_dir, regridder, degrader
    )
    logger.info("  hydromet vars: %d", len(hydromet))

    logger.info("[%d-%02d] processing surface instantaneous ...", year, month)
    surf_inst, surf_inst_attrs = process_surface_instantaneous(
        year, month, target_times, main_dir, regridder, degrader
    )
    logger.info("  surface inst vars: %d", len(surf_inst))

    logger.info("[%d-%02d] processing surface accumulated ...", year, month)
    surf_accum, surf_accum_attrs = process_surface_accumulated(
        year, month, target_times, main_dir, regridder, degrader
    )
    logger.info("  surface accum vars: %d", len(surf_accum))

    all_vars: Dict[str, np.ndarray] = {}
    all_vars.update(pressure_named)
    all_vars.update(o3_named)
    all_vars.update(hydromet)
    all_vars.update(surf_inst)
    all_vars.update(surf_accum)

    all_attrs: Dict[str, Dict[str, str]] = {}
    all_attrs.update(pressure_attrs)
    all_attrs.update(o3_attrs)
    all_attrs.update(hydromet_attrs)
    all_attrs.update(surf_inst_attrs)
    all_attrs.update(surf_accum_attrs)

    # ---- Strict schema check: must have ALL expected vars or refuse to write ----
    expected_vars = set(
        catalog.expected_pressure_var_names()
        + catalog.expected_surface_var_names()
    )
    actual_vars = set(all_vars.keys())
    missing = expected_vars - actual_vars
    extra = actual_vars - expected_vars
    if missing or extra:
        raise RuntimeError(
            f"[{year}-{month:02d}] schema mismatch: have {len(actual_vars)} vars, "
            f"expected {len(expected_vars)} ({catalog.total_var_count()}). "
            f"Missing {len(missing)} ({sorted(missing)[:10]}{'...' if len(missing)>10 else ''}), "
            f"extra {len(extra)} ({sorted(extra)[:10]}{'...' if len(extra)>10 else ''}). "
            "Refusing to write partial file. Fix source data and rerun with --overwrite."
        )
    logger.info("[%d-%02d] schema OK: %d vars match catalog", year, month, len(actual_vars))

    logger.info("[%d-%02d] writing %d vars to %s ...", year, month, len(all_vars), out_path)
    write_output(
        out_path=out_path,
        target_times=target_times,
        all_vars=all_vars,
        all_attrs=all_attrs,
        nside=nside_low,
        nest=nest,
        extra_attrs={
            "year": int(year),
            "month": int(month),
            "ordering_arg": "RING" if not nest else "NESTED",
            "n_pressure_dyn": len(pressure_named),
            "n_o3_extended": len(o3_named),
            "n_hydrometeors": len(hydromet),
            "n_surface_inst": len(surf_inst),
            "n_surface_accum": len(surf_accum),
        },
    )
    logger.info("[%d-%02d] DONE → %s", year, month, out_path)
    return out_path
