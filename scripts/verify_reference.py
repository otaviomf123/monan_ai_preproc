#!/usr/bin/env python3
"""Verify a locally rebuilt v3 dataset against the reference exported from ncempa.

Reference files live in `reference/` (committed to git):
    reference/stats/healpix_statistics_v3.json   frozen normalization stats (md5 checked)
    reference/stats/healpix_masks_v3.nc          frozen coverage/variation masks
    reference/index/{train,val,test}_index.json  split metadata of the cluster build
    reference/fingerprints/monthly_fingerprint.json     per-var stats of t0 for 1979-01 and 2022-01
    reference/fingerprints/normalized_fingerprint.json  per-var stats of t0 of val/test memmaps

Subcommands:
    stats                         check the frozen stats/masks are intact (md5)
    month  <file.nc> [<file.nc>]  compare converted monthly files (step 1) with the fingerprint
    months <out_root> <y0> <y1>   structural check of every monthly file in a year range
    normalized <norm_dir>         compare split shapes/var_list/dates + val/test t0 fingerprint

Exit code 0 = all PASS, 1 = at least one FAIL. WARN does not fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "reference"
STATS_MD5 = "372443cef4ee4fc97709b5f8e5947a88"
N_VARS = 235
NPIX = 49152

_fail = 0


def report(status: str, msg: str) -> None:
    global _fail
    if status == "FAIL":
        _fail += 1
    print(f"[{status}] {msg}")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref_var_list() -> list[str]:
    with open(REF / "index" / "train_index.json") as f:
        return json.load(f)["var_list"]


# ---------------------------------------------------------------------------
def cmd_stats(_args) -> None:
    p = REF / "stats" / "healpix_statistics_v3.json"
    got = md5(p)
    report("PASS" if got == STATS_MD5 else "FAIL", f"stats md5 {got} (expected {STATS_MD5})")
    with open(p) as f:
        st = json.load(f)
    names = sorted(st["statistics"].keys())
    report("PASS" if names == ref_var_list() else "FAIL",
           f"stats var_list ({len(names)}) matches reference train_index var_list")
    m = REF / "stats" / "healpix_masks_v3.nc"
    report("PASS" if m.exists() else "FAIL", f"masks file present: {m}")


def _compare_month(path: Path, ref: dict) -> None:
    import xarray as xr

    ds = xr.open_dataset(path)
    tag = path.name
    names = sorted(ds.data_vars)
    report("PASS" if len(names) == N_VARS else "FAIL", f"{tag}: {len(names)} vars (expected {N_VARS})")
    report("PASS" if names == sorted(ref["vars"]) else "FAIL", f"{tag}: variable names match reference")
    report("PASS" if ds.attrs.get("ordering") == "RING" else "FAIL",
           f"{tag}: ordering={ds.attrs.get('ordering')} (expected RING)")
    report("PASS" if ds.sizes.get("pixel") == NPIX else "FAIL", f"{tag}: pixel={ds.sizes.get('pixel')}")
    report("PASS" if int(ds.sizes["time"]) == ref["n_time"] else "FAIL",
           f"{tag}: n_time={int(ds.sizes['time'])} (expected {ref['n_time']})")
    report("PASS" if str(ds.time.values[0]) == ref["time0"] else "FAIL",
           f"{tag}: time0={ds.time.values[0]} (expected {ref['time0']})")

    worst, n_warn, n_bad = 0.0, 0, 0
    bad = []
    for v in names:
        if v not in ref["vars"]:
            continue
        r = ref["vars"][v]
        x = ds[v].isel(time=0).values.astype(np.float64)
        mean, std = float(np.nanmean(x)), float(np.nanstd(x))
        nanf = float(np.isnan(x).mean())
        scale = max(abs(r["nanstd_t0"]), 1e-12)
        dm = abs(mean - r["nanmean_t0"]) / scale
        ds_ = abs(std - r["nanstd_t0"]) / scale
        d = max(dm, ds_, abs(nanf - r["nan_frac_t0"]) * 10)
        worst = max(worst, d)
        if d > 1e-2:
            n_bad += 1
            bad.append((v, d))
        elif d > 1e-4:
            n_warn += 1
    bad.sort(key=lambda t: -t[1])
    if n_bad:
        report("FAIL", f"{tag}: {n_bad} vars differ > 1% of std from reference; worst: {bad[:8]}")
    elif n_warn:
        report("WARN", f"{tag}: {n_warn} vars differ 0.01-1% of std (source/version noise?); worst={worst:.2e}")
    else:
        report("PASS", f"{tag}: all vars match reference t0 stats (worst rel diff {worst:.2e})")


def cmd_month(args) -> None:
    with open(REF / "fingerprints" / "monthly_fingerprint.json") as f:
        fp = json.load(f)
    for p in map(Path, args.files):
        key = p.stem  # YYYY_MM
        if key not in fp:
            report("FAIL", f"no reference fingerprint for {key}; available: {list(fp)}")
            continue
        _compare_month(p, fp[key])


def cmd_months(args) -> None:
    import xarray as xr

    root = Path(args.out_root)
    missing, wrong = [], []
    for y in range(args.y0, args.y1 + 1):
        for m in range(1, 13):
            p = root / str(y) / f"{y}_{m:02d}.nc"
            if not p.exists():
                missing.append(p.name)
                continue
            with xr.open_dataset(p) as ds:
                nv = len(ds.data_vars)
                # a variable that is entirely NaN at t0 usually means a missing raw file
                all_nan = [v for v in ds.data_vars if np.isnan(ds[v].isel(time=0).values).all()]
            if nv != N_VARS or all_nan:
                wrong.append((p.name, nv, all_nan[:5]))
    report("PASS" if not missing else "FAIL", f"monthly files missing in {args.y0}-{args.y1}: {len(missing)} {missing[:12]}")
    report("PASS" if not wrong else "FAIL", f"monthly files with wrong var count or all-NaN vars: {wrong[:12]}")


def cmd_normalized(args) -> None:
    nd = Path(args.norm_dir)
    with open(REF / "fingerprints" / "normalized_fingerprint.json") as f:
        fp = json.load(f)
    var_ref = ref_var_list()
    for split in ("train", "val", "test"):
        idx_p = nd / f"{split}_index.json"
        if not idx_p.exists():
            report("FAIL", f"{split}: {idx_p} missing")
            continue
        with open(idx_p) as f:
            idx = json.load(f)
        with open(REF / "index" / f"{split}_index.json") as f:
            ref = json.load(f)
        for k in ("start_date", "end_date", "n_timesteps", "n_pixels", "n_variables",
                  "dtype", "max_z_clip", "ordering", "nside"):
            report("PASS" if idx.get(k) == ref.get(k) else "FAIL",
                   f"{split}: {k}={idx.get(k)} (reference {ref.get(k)})")
        report("PASS" if idx["var_list"] == var_ref else "FAIL", f"{split}: var_list order identical to reference")
        arr = np.load(nd / f"{split}.npy", mmap_mode="r")
        report("PASS" if list(arr.shape) == [ref["n_timesteps"], NPIX, N_VARS] else "FAIL",
               f"{split}: memmap shape {arr.shape}")
        if split in fp:
            r = fp[split]
            x = np.asarray(arr[0], dtype=np.float64)
            dm = np.abs(x.mean(0) - np.array(r["mean_per_var_t0"]))
            dsd = np.abs(x.std(0) - np.array(r["std_per_var_t0"]))
            d = np.maximum(dm, dsd)
            worst = int(d.argmax())
            # float16 storage => ~1e-3 resolution; 0.02 z-units = clearly different data
            if d.max() > 0.02:
                bad = [(var_ref[i], round(float(d[i]), 4)) for i in np.argsort(-d)[:8]]
                report("FAIL", f"{split}: t0 normalized stats differ from reference; worst {bad}")
            elif d.max() > 2e-3:
                report("WARN", f"{split}: t0 max diff {d.max():.4f} ({var_ref[worst]})")
            else:
                report("PASS", f"{split}: t0 normalized stats match reference (max diff {d.max():.1e})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    pm = sub.add_parser("month")
    pm.add_argument("files", nargs="+")
    pms = sub.add_parser("months")
    pms.add_argument("out_root")
    pms.add_argument("y0", type=int)
    pms.add_argument("y1", type=int)
    pn = sub.add_parser("normalized")
    pn.add_argument("norm_dir")
    args = p.parse_args()
    {"stats": cmd_stats, "month": cmd_month, "months": cmd_months, "normalized": cmd_normalized}[args.cmd](args)
    print(f"\n{'ALL PASS' if _fail == 0 else f'{_fail} FAIL(s)'}")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
