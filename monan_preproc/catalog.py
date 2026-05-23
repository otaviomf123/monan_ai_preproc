"""ERA5 file catalog for v3 preprocessing.

Defines: which file types exist, what variables each contains, accumulation
factors for hourly accumulators, pressure-level groupings, and the final
exclusion list (vegetation quasi-static + temperature_extremes + precip type).

Reference period scanned to build this: 2020 + 2003 (hydrometeors) + 1951.
Source files live under: /mnt/beegfs/otavio.feitosa/era5/
"""
from __future__ import annotations

# Pressure levels of the canonical 17-level vertical grid for v3.
# Ordering: top-of-atmosphere → surface (decreasing pressure altitude).
PRESSURE_LEVELS = [
    10, 30, 50,                                   # stratosphere
    100, 150, 200, 250,                           # UTLS / upper troposphere
    300, 400, 500,                                # mid troposphere
    600, 700, 750,                                # lower troposphere
    850, 925, 950, 1000,                          # boundary layer / surface
]

# ---------------------------------------------------------------------------
# Pressure-level files (each contains r/t/u/v/z over a SUBSET of levels)
# ---------------------------------------------------------------------------
PRESSURE_LEVEL_FILES = {
    "era5_pressure_boundary_layer":     [850, 925, 1000],
    "era5_pressure_lower_troposphere":  [600, 700, 750],
    "era5_pressure_mid_troposphere":    [300, 400, 500],
    "era5_pressure_upper_troposphere":  [100, 150, 200],
    "era5_pressure_stratosphere":       [10, 30, 50],
}
PRESSURE_VARS = ["r", "t", "u", "v", "z"]
# Stratosphere file ALSO contains o3 at its 3 levels.
STRATO_O3_LEVELS = [10, 30, 50]

# ---------------------------------------------------------------------------
# Additional levels (250 hPa + 950 hPa for r/t/u/v/z) — under data_additional_levels/
# ---------------------------------------------------------------------------
ADDITIONAL_LEVELS_FILE = "era5_pressure_additional_levels"
ADDITIONAL_LEVELS = [250, 950]

# ---------------------------------------------------------------------------
# Hydrometeors + cloud cover — under data_hydrometeors/
# 17-level files. Per-variable level filtering applied later.
# ---------------------------------------------------------------------------
HYDROMETEOR_FILES = {
    "era5_pressure_hydro_precip":    ["crwc", "cswc"],
    "era5_pressure_hydro_nonprecip": ["clwc", "ciwc"],
    "era5_pressure_cloud_fraction":  ["cc"],
}

# Levels to KEEP per hydrometeor variable (rest are physically zero).
# Determined by 4-month multi-sample analysis (2003, 2010 Jan/Jul).
HYDROMETEOR_KEEP_LEVELS = {
    "crwc": [200, 250, 300, 400, 500, 600, 700, 750, 850, 925, 950, 1000],   # 12
    "cswc": PRESSURE_LEVELS,                                                  # 17
    "clwc": [250, 300, 400, 500, 600, 700, 750, 850, 925, 950, 1000],         # 11
    "ciwc": PRESSURE_LEVELS,                                                  # 17
    "cc":   PRESSURE_LEVELS,                                                  # 17
}

# ---------------------------------------------------------------------------
# Ozone extended — under data_ozone_extended/ (14 tropospheric/UT levels)
# Concatenate with strato o3 (10/30/50) → o3 over full 17-level grid.
# ---------------------------------------------------------------------------
OZONE_EXTENDED_FILE = "era5_pressure_ozone_extended"
OZONE_EXTENDED_LEVELS = [100, 150, 200, 250, 300, 400, 500, 600, 700, 750, 850, 925, 950, 1000]

# ---------------------------------------------------------------------------
# Surface — instantaneous (6-hourly snapshots: 00/06/12/18Z), in main era5/{YEAR}/
# Each entry maps file basename → list of variables to KEEP from that file.
# ---------------------------------------------------------------------------
SURFACE_INSTANTANEOUS = {
    "era5_surface_boundary_layer":   ["blh", "cape", "cin", "kx", "totalx"],
    "era5_surface_clouds":           ["cbh", "hcc", "lcc", "mcc", "tcc"],
    "era5_surface_ocean_ice":        ["istl1", "istl2", "istl3", "istl4", "siconc"],
    "era5_surface_pressure":         ["msl"],                            # sp explicitly excluded
    "era5_surface_snow":             ["sd"],
    "era5_surface_snow_processes":   ["rsn"],
    "era5_surface_soil_moisture":    ["swvl1", "swvl2", "swvl3", "swvl4"],
    "era5_surface_soil_temperature": ["stl1", "stl2", "stl3", "stl4"],
    "era5_surface_sst":              ["sst"],
    "era5_surface_temperature":      ["d2m", "t2m"],
    "era5_surface_water_column":     ["tciw", "tclw", "tco3", "tcrw", "tcsw", "tcw", "tcwv"],
    # i10fg (instantaneous 10m wind gust) is the MAXIMUM gust over the past 6h
    # window, not a true snapshot — same statistical nature as mn2t/mx2t which
    # are already excluded. Excluded for the same reason.
    "era5_surface_wind":             ["u10", "u100", "v10", "v100"],
}

# Surface files explicitly skipped per user decision.
SURFACE_SKIPPED = [
    "era5_surface_temperature_extremes",  # mn2t, mx2t — 6h window max/min, not snapshots
    "era5_surface_precipitation_type",    # ptype — categorical
    "era5_surface_vegetation",            # quasi-static, low value/channel
]
# Inside era5_surface_wind, i10fg is also a 6h-window MAX (not a snapshot)
# and is therefore excluded from SURFACE_INSTANTANEOUS["era5_surface_wind"].

# ---------------------------------------------------------------------------
# Surface — accumulated (hourly files, summed over each [t-6h, t] window).
# Per-var spec: {factor, units (post-conversion), accumulation note}.
# factor is applied AFTER summation; the resulting field replaces source units.
# ---------------------------------------------------------------------------
_M_TO_MM     = {"factor": 1000.0,         "units": "mm",       "accumulation": "6h sum (m → mm)"}
_M_TO_MM_WE  = {"factor": 1000.0,         "units": "mm w.e.",  "accumulation": "6h sum (m water equivalent → mm)"}
_J_TO_WM2    = {"factor": 1.0 / 21600.0,  "units": "W m**-2",  "accumulation": "6h mean (J m**-2 → W m**-2)"}

SURFACE_ACCUM_FILES = {
    "era5_surface_hourly_precipitation":     {"tp":    _M_TO_MM},
    "era5_surface_hourly_evaporation":       {"e":     _M_TO_MM,
                                              "pev":   _M_TO_MM},
    "era5_surface_hourly_runoff":            {"ro":    _M_TO_MM,
                                              "sro":   _M_TO_MM,
                                              "ssro":  _M_TO_MM},
    "era5_surface_hourly_snow":              {"sf":    _M_TO_MM_WE,
                                              "smlt":  _M_TO_MM_WE},
    "era5_surface_hourly_heat_flux":         {"sshf":  _J_TO_WM2,
                                              "slhf":  _J_TO_WM2},
    "era5_surface_hourly_radiation_solar":   {"ssr":   _J_TO_WM2,
                                              "ssrd":  _J_TO_WM2,
                                              "ssrdc": _J_TO_WM2,
                                              "tisr":  _J_TO_WM2,
                                              "tsr":   _J_TO_WM2},
    "era5_surface_hourly_radiation_thermal": {"str":   _J_TO_WM2,
                                              "strd":  _J_TO_WM2,
                                              "strdc": _J_TO_WM2,
                                              "ttr":   _J_TO_WM2},
}

# ---------------------------------------------------------------------------
# Invariant fields (orography, land-sea mask, soil/veg type) — single 0.25° file.
# Used downstream for psfc derivation, surface emissivity in PyCRTM, etc.
# ---------------------------------------------------------------------------
INVARIANT_FILE = "era5_invariant.nc"
INVARIANT_VARS = ["z", "lsm", "slt", "sdor", "slor", "isor", "anor", "cl",
                  "cvl", "cvh", "tvl", "tvh", "fal"]

# ---------------------------------------------------------------------------
# Final variable count (only what comes directly from ERA5; no derived vars):
#   pressure r/t/u/v/z × 17                              = 85
#   o3 × 17                                              = 17
#   crwc 12 + cswc 17 + clwc 11 + ciwc 17 + cc 17        = 74
#   surface instantaneous                                = 40
#   surface accumulated                                  = 19
#   ─────────────────────────────────────────────────────────
#   total                                                = 235
#
# Derived quantities (q from r/t/p, psfc from msl+z, skin_T composite, etc.)
# are computed downstream on-the-fly — NOT stored here. The package retains a
# helper module `monan_preproc.derive` for whoever does that derivation.
# ---------------------------------------------------------------------------


def expected_pressure_var_names() -> list[str]:
    """List of expanded pressure-level var names (var_NNNhpa)."""
    names = []
    for v in PRESSURE_VARS:
        for lvl in PRESSURE_LEVELS:
            names.append(f"{v}_{lvl}hpa")
    for lvl in PRESSURE_LEVELS:
        names.append(f"o3_{lvl}hpa")
    for var, levels in HYDROMETEOR_KEEP_LEVELS.items():
        for lvl in levels:
            names.append(f"{var}_{lvl}hpa")
    return names


def expected_surface_var_names() -> list[str]:
    """List of all surface variable names (instantaneous + accumulated)."""
    inst = [v for vs in SURFACE_INSTANTANEOUS.values() for v in vs]
    accum = [v for vs in SURFACE_ACCUM_FILES.values() for v in vs.keys()]
    return inst + accum


def total_var_count() -> int:
    return len(expected_pressure_var_names()) + len(expected_surface_var_names())
