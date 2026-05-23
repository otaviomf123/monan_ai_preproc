# MONAN-IA preproc (v3)

Preprocessing-only project for the MONAN-IA v3 dataset.

Converts raw ERA5 NetCDF files on the regular 1° lat-lon grid into HEALPix
`nside=64` (NESTED ordering) NetCDF files — one per month, **236 variables**,
ready for downstream normalization, climatology, and training.

## What this does

1. **Reads raw ERA5 NetCDF** in `/mnt/beegfs/otavio.feitosa/era5/` and three
   sub-trees (additional levels, hydrometeors, ozone extended).
2. **Selects 6h instantaneous timestamps** (00/06/12/18Z) for instantaneous
   variables and **sums hourly accumulators** over each `[t-6h, t]` window
   (with previous-month boundary handling: pulls last 5 hours of the prior
   month for the day-1 00Z target).
3. **Regrids each variable** from regular 1° → HEALPix `nside=256` via
   BallTree nearest-neighbor with haversine metric.
4. **Hierarchically degrades** `nside=256 → 128 → 64` via `np.nanmean` over
   the 4 NESTED children at each step. NaN-aware: limited-domain variables
   (sst, soil moisture, sea ice) propagate masks correctly through coastlines.
5. **Concatenates ozone** (3 stratospheric levels + 14 from extended download)
   and **additional pressure levels** (250 + 950 hPa) onto the canonical
   17-level vertical grid.
6. **Filters hydrometeor levels** per-variable to drop physically always-zero
   combinations (e.g., `clwc` above 250 hPa where T < −38°C → liquid water
   nucleates homogeneously to ice).
7. **Writes one NetCDF per month** with full CF-1.8 metadata: HEALPix
   coordinates, source CDS attrs preserved (units, long_name, standard_name,
   GRIB IDs), pipeline provenance attributes, and override of units for
   accumulated quantities (m → mm, J/m² → W/m²).

## What this does NOT do

- **No derived variables.** `q` (specific humidity), `psfc` (surface pressure),
  skin temperature, etc., are computed on-the-fly downstream — not stored.
  A helper module `monan_preproc.derive` is provided for whoever needs them.
- **No normalization.** Output is in raw physical units. Power transform +
  z-score is a separate downstream step.
- **No climatology.** Per-DOY climatology is a separate downstream step.

## Variable inventory (236 total)

| Group | Count | Detail |
|---|---|---|
| Pressure dynamics | 85 | `r/t/u/v/z` × 17 levels |
| Ozone | 17 | strato (3) + extended (14) → full 17-level profile |
| Hydrometeors + cloud cover | 74 | `crwc`(12) + `cswc`(17) + `clwc`(11) + `ciwc`(17) + `cc`(17) |
| Surface instantaneous | 41 | incl. `msl`; excludes `sp`, `mn2t`, `mx2t`, `ptype`, vegetation |
| Surface accumulated | 19 | precip, evap, fluxes, radiation, runoff, snow accums |

### Vertical grid (17 levels, hPa)

`[10, 30, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 750, 850, 925, 950, 1000]`

Same grid for `r/t/u/v/z`, `o3`, and all hydrometeors. Built from:
- 15 levels in main `era5_pressure_*.nc` files
- 2 added from `era5_pressure_additional_levels_*.nc` (250 and 950)

### Hydrometeor level exclusions (11 combos dropped)

Validated via 4-month multi-sample analysis (Jan/Jul × 2003/2010): always
exactly zero. Physically: liquid water nucleates homogeneously to ice when
T < −38°C, which corresponds to ~250 hPa in midlatitudes.

| Variable | Levels excluded | Levels kept |
|---|---|---|
| `clwc` | 10, 30, 50, 100, 150, 200 (6) | 250–1000 (11) |
| `crwc` | 10, 30, 50, 100, 150 (5) | 200–1000 (12) |
| `cswc`, `ciwc`, `cc` | (none) | all 17 |

Documented per-variable via the `level_filter` attribute.

### Variable naming convention

- Surface: `<var>` (e.g., `t2m`, `msl`, `sst`, `tp`)
- Pressure: `<var>_<level>hpa` (e.g., `t_500hpa`, `o3_50hpa`, `clwc_850hpa`)

## NetCDF output format

- **Format**: NETCDF4 (HDF5-backed), CF-1.8 conventions
- **Dimensions**: `(time=N×4, pixel=49152)`
- **Coords**: `time` (datetime64), `pixel` (int32), `healpix_lat` (deg N),
  `healpix_lon` (deg E)
- **Compression**: `zlib level 5 + shuffle filter` — ~50% ratio on float32
- **Chunking**: `(4, 49152)` — one day (4 timesteps) per chunk → fast random
  access for the autoregressive model dataloader

### Per-variable metadata (preserved + augmented)

Every variable carries:

- **From CDS source** (preserved unchanged): `units` (raw), `long_name`,
  `standard_name` (when ECMWF set it), `GRIB_paramId`, `GRIB_shortName`
- **Pressure-level vars** add: `pressure_level: "NNN hPa"` and the level
  appended to `long_name`
- **Hydrometeors clwc/crwc** add: `level_filter: "kept N/17 levels; high
  levels excluded (no liquid water at T < -38 deg C)"`
- **Accumulated vars** override `units` with the post-conversion units
  (`mm`, `mm w.e.`, `W m**-2`) and add: `original_units`, `accumulation`
  (e.g., `"6h sum (m → mm)"`), `cell_methods` (`"time: sum"` / `"time: mean"`),
  `time_window: "[t - 6h, t]"`
- **All variables** add: `regrid: "BallTree NN regular 1°→nside=256, then
  nested-nanmean degrade →nside=64"`

### Dataset-level attributes

`Conventions=CF-1.8`, `nside=64`, `npix=49152`, `ordering=NESTED`,
`regrid_method=...`, `n_variables=236`, per-category counts, source,
package version, ISO creation date, `year`, `month`.

## Usage

```bash
# Single month (interactive)
python scripts/convert_month.py --year 1951 --month 1

# SLURM (smoke test of one month)
sbatch slurm/submit_test_1951_jan.sh

# SLURM (full year as array job, 4 tasks parallel)
sbatch slurm/submit_test_1951_array.sh

# Sanity-check an output file
python scripts/validate_output.py /mnt/beegfs/otavio.feitosa/era5_healpix_v3/1951/1951_01.nc
```

## Output layout

```
/mnt/beegfs/otavio.feitosa/era5_healpix_v3/
└── {YYYY}/
    └── {YYYY}_{MM}.nc           # one month, 236 vars on HEALPix nside=64 NESTED
```

Approximate size: **~2.3 GB per month** (zlib-5 compressed, float32).
Full archive 1950–2025: **~2.1 TB**.

## Project layout

```
monan_ia_preproc/
├── README.md                    (this file)
├── monan_preproc/               # importable package
│   ├── __init__.py
│   ├── catalog.py               # variable lists, levels, accum factors
│   ├── healpix.py               # LatLonToHEALPix + HierarchicalDegrader (NESTED)
│   ├── convert.py               # main pipeline (regrid + degrade + write)
│   └── derive.py                # downstream helpers (q from r,t,p)
├── scripts/
│   ├── convert_month.py         # CLI entry point (--year, --month, --overwrite)
│   └── validate_output.py       # sanity audit of an output NetCDF
└── slurm/
    ├── submit_test_1951_jan.sh         # single month
    └── submit_test_1951_array.sh       # full year as array job
```

## Status

Pipeline validated end-to-end with full year 1951 (12 months × 236 vars).
All array tasks COMPLETED. Output verified for:

- Variable counts (236 total, all categories matching catalog)
- Vertical grid coverage (17 levels for `r/t/u/v/z` and `o3`; physical
  filtering for `clwc`/`crwc`)
- Limited-domain NaN handling (`sst` 25% NaN over land, `siconc` 0 over land
  per ECMWF convention, `swvl1`/`stl1` use 0 sentinel over ocean)
- NESTED ordering empirically verified (lat/lon coords match
  `hp.pix2ang(nside, ipix, nest=True)`; spatially adjacent pixels have ~1 K
  spread in `t2m`)
- CF-1.8 metadata coverage: 100% have units + long_name; 100% have
  GRIB provenance; 19/19 accumulators have explicit conversion documentation;
  23/23 filtered-hydrometeor vars carry `level_filter` attribute

Per-month wall time: ~5 min on a single CPU core (4 GB RAM).
Full year as `--array=1-12%4`: ~16 min wall clock.
