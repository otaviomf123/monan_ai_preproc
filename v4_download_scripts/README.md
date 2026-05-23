# ERA5 download scripts — v3 reference + v4 (0.25°) ready copies

Snapshot of the production download stack used to build the v3 dataset
(at 1° resolution), plus a parallel set of scripts already tuned to
0.25° for the upcoming v4. Same 235 variables, same per-month NetCDF
layout — only the grid changes.

## Layout

```
v4_download_scripts/
├── README.md                           this file
├── production/                          v3 reference (1° as-on-cluster)
│   ├── era5_download.py                pressure-level + surface 6h (main)
│   ├── era5_download_additional_levels.py   missing pressure levels (200, 250, 750, 925, 950)
│   ├── era5_download_hourly.py         hourly accumulators (TP, evap, runoff, radiation, snow, etc.)
│   ├── era5_download_hourly_2025_09_12.py   gap-fill for late-2025 months
│   ├── era5_download_hydrometeors.py   CLWC / CIWC / CRWC / CSWC / CC on pressure levels
│   └── era5_download_ozone_extended.py o3 on the full vertical (extends short-stack o3)
├── v4_grid_025/                         v4 ready copies (0.25° applied, vars unchanged)
│   └── (same six scripts, only the CDS `grid` parameter differs)
├── v4_grid_025_cleaned/                 ★ recommended starting point for v4
│   ├── era5_download.py                 surface_groups trimmed to catalog needs
│   └── (other 5 scripts identical to v4_grid_025/)
├── fixes/                               one-off utilities (kept for reference)
│   ├── era5_redownload_fixes.py
│   ├── era5_redownload_fixes_2.py
│   ├── era5_redownload_fixes_3.py
│   ├── era5_redownload_1998_11.py
│   └── era5_redownload_wind_no_gust.py
└── logs_sample/                         one historical log so the format is
                                          obvious without running anything
```

## v3 vs v4 — what actually changes

| Aspect              | v3 (1°)        | v4 (0.25°)          | factor |
|---------------------|----------------|---------------------|--------|
| `grid` CDS param    | `['1.0','1.0']`| `['0.25','0.25']`   | — |
| Pixels / 2-D field  | 181 × 361 = 65 341 | 721 × 1440 = 1 038 240 | **15.9×** |
| Bytes / 2-D field   | 261 KB (f32)   | 4.2 MB (f32)        | **15.9×** |
| 1 month (124 ts × 235 vars × f32 raw) | ~7.5 GB | ~120 GB | **15.9×** |
| Monthly NetCDF (zlib 5) | ~500–700 MB | ~7–11 GB         | ~15× |
| Full 1950-2025 raw  | ~1–2 TB        | **~16–30 TB**       | ~15× |
| CDS request time    | ~30 s / file   | ~3–8 min / file     | ~10× |

**Variables, levels, time cadence, file layout, and download orchestration
are identical between v3 and v4** — only `grid` changes inside the
`client.retrieve(...)` dict in each script.

## Where the grid is set (single line per script)

```bash
grep -nE "'grid'|\"grid\"" v4_grid_025/*.py
```
Should print one or two `'grid': ['0.25', '0.25']` per file. If you ever
need to roll back to 1° just sed the literal back.

## Running on a new cluster

Each script is self-contained; only requires `cdsapi` (and the user's
`~/.cdsapirc` credentials). Suggested setup:

```bash
# 1. Pick a host with fast outbound + lots of disk (v4 needs ~20 TB headroom)
# 2. Setup CDS client
conda create -n era5dl python=3.11 cdsapi -y
conda activate era5dl
mkdir ~/.cds && cp $HOME/.cdsapirc ~/   # or paste your key

# 3. Set output root (default is CWD/data_<group>/)
cd /scratch/your_user/era5_v4

# 4. Run each script. They're all idempotent — re-running skips files
#    that already exist with non-trivial size. Order does not matter
#    (different CDS endpoints), but you can interleave:
nohup python era5_download.py                        > log_main.txt    2>&1 &
nohup python era5_download_hourly.py                 > log_hourly.txt  2>&1 &
nohup python era5_download_hydrometeors.py           > log_hydro.txt   2>&1 &
nohup python era5_download_additional_levels.py      > log_addl.txt    2>&1 &
nohup python era5_download_ozone_extended.py         > log_o3.txt      2>&1 &
```

CDS rate-limits per user, so 5 parallel scripts already saturate the
queue — adding more threads inside any of them won't speed things up.
Expect a full v4 download to take **3–6 weeks of wall time** on a single
CDS account, dominated by CDS server queueing.

## Output layout (same for v3 and v4)

```
data_<group>/
└── {YYYY}/
    ├── era5_{group}_pressure_{YYYY}_{MM}.nc       (pressure-level vars)
    ├── era5_{group}_surface_{YYYY}_{MM}.nc        (instantaneous surface)
    ├── era5_surface_hourly_{group}_{YYYY}_{MM}.nc (accumulated, hourly)
    └── …
```

The preprocessing pipeline (`monan_ia_preproc/monan_preproc/convert.py`)
discovers all these files automatically — no schema change needed for v4
beyond pointing `--src-dir` at the new download root.

## What `v4_grid_025_cleaned/` changes vs `v4_grid_025/`

Both download the EXACT 235 variables that `monan_preproc/catalog.py` keeps —
nothing more, nothing less. The original v3 scripts requested ~12 surface
vars that the catalog silently drops downstream; v4_grid_025_cleaned/ stops
requesting them at the CDS layer too, saving ~165 GB of v4 transfer.

| Group | Action | Reason |
|---|---|---|
| `temperature_extremes` | REMOVED entirely | mn2t/mx2t are 6h window stats, not snapshots — `SURFACE_SKIPPED` |
| `precipitation_type` | REMOVED entirely | ptype categorical, low signal — `SURFACE_SKIPPED` |
| `vegetation` | REMOVED entirely | quasi-static, low value/channel — `SURFACE_SKIPPED` |
| `wind` | `[u10, v10, i10fg]` → `[u10, v10, u100, v100]` | i10fg = 6h max gust (triggers CDS ZIP bug); u100/v100 needed by catalog |
| `boundary_layer` | `[blh, bld]` → `[blh, cape, cin, kx, totalx]` | bld unused by catalog; convective indices used |
| `pressure` | unchanged (kept sp + msl) | both used downstream (psfc derivation, MSL field) |
| hydrometeors | unchanged (17 levels per file) | union of crwc/cswc and clwc/ciwc needs is still 17; ~150 GB vertical waste is accepted to avoid 1800 extra CDS requests. Note documented in the script |

The corrected wind + boundary_layer schema was actually pioneered by
`fixes/era5_redownload_fixes.py` and applied per-file to v3 after the
initial run discovered the gap. The cleaned scripts bake those fixes in
upfront for v4.

## Notes / pitfalls

- **Wind ZIP issue**: when `i10fg` (10-m gust) is requested alongside
  `u10/v10`, CDS sometimes ships the bundle as ZIP instead of NetCDF.
  `era5_download.py` already excludes `i10fg` from the wind group — keep
  that out for v4 too.
- **Late-2025 embargo**: months Sep–Dec 2025 were rejected during v3
  ingestion (CDS embargo on recent reanalysis). Re-run
  `era5_download_hourly_2025_09_12.py` once they're released.
- **Disk pre-check**: at 0.25° one corrupted/half-written 11 GB file
  costs 22 GB (download + .tmp). All production scripts use atomic
  `.tmp → rename` writes; keep that pattern in any fork.
- **Strict-check downstream**: `convert.py` enforces 235 vars per
  monthly NetCDF. Any download script that drops a variable group will
  surface as a hard preprocessing failure — desired behavior.
