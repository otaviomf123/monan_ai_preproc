"""On-disk cache for expensive geometry pre-computation.

Two artifacts that are pure functions of grid geometry and never change:

  1. **Regrid index**: BallTree NN result mapping each HEALPix nside_high
     pixel to its nearest source lat-lon point. ~6 MB int array, ~15 s to
     compute. Without cache, every parallel worker recomputes redundantly.

  2. **Hierarchical-degrade children map**: per-step (nside, nside/2) mapping
     from each parent to its 4 children. ~50 ms total to compute (RING
     ordering uses ang2pix lookup; NESTED is trivial). Cheap individually
     but worth caching for cleanliness + reuse from notebooks.

Cache layout (default `cache_dir` is `<out_root>/_cache/`):

    _cache/
    ├── regrid_ring_181x360_to_nside256_v3.npz
    ├── degrade_ring_nside256_to_64_v3.npz
    └── (more files for other configurations)

Files are versioned by `CACHE_VERSION` — bump when the on-disk format or
algorithm changes so old caches are invalidated.

Usage from pipeline code:

    from .cache import load_or_build_regrid_indices, load_or_build_children
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import healpy as hp
import numpy as np
from sklearn.neighbors import BallTree

CACHE_VERSION = "v3"


# ---------------------------------------------------------------------------
def _grid_signature(lat: np.ndarray, lon: np.ndarray) -> str:
    """Compact signature of an axis-aligned regular lat-lon grid."""
    return (
        f"{lat.size}x{lon.size}_"
        f"{lat[0]:+07.2f}_{lat[-1]:+07.2f}_{lon[0]:+07.2f}_{lon[-1]:+07.2f}"
    )


def regrid_cache_path(cache_dir: Path, lat: np.ndarray, lon: np.ndarray,
                       nside_high: int, ordering: str) -> Path:
    name = (f"regrid_{ordering}_{_grid_signature(lat, lon)}_"
            f"to_nside{nside_high}_{CACHE_VERSION}.npz")
    return cache_dir / name


def degrade_cache_path(cache_dir: Path, nside_high: int, nside_low: int,
                        ordering: str) -> Path:
    name = (f"degrade_{ordering}_nside{nside_high}_to_{nside_low}_"
            f"{CACHE_VERSION}.npz")
    return cache_dir / name


# ---------------------------------------------------------------------------
def load_or_build_regrid_indices(
    cache_dir: Optional[Path],
    lat: np.ndarray,
    lon: np.ndarray,
    nside_high: int,
    nest: bool,
    verbose: bool = True,
) -> np.ndarray:
    """Return the npix_high-length int64 array mapping each HEALPix pixel to
    its nearest lat-lon source index.

    If `cache_dir` is provided and the cache file exists, loads from disk
    (fast). Otherwise computes via BallTree(haversine), saves, returns.
    """
    ordering = "nested" if nest else "ring"

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = regrid_cache_path(cache_dir, lat, lon, nside_high, ordering)
        if path.exists():
            if verbose:
                print(f"[cache hit] regrid indices: {path.name}")
            with np.load(path) as data:
                return data["idx"]
        if verbose:
            print(f"[cache miss] computing BallTree NN → nside={nside_high} ({ordering}) ...")

    # Compute
    npix = hp.nside2npix(nside_high)
    theta, phi = hp.pix2ang(nside_high, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(theta)
    hp_lon = np.degrees(phi)
    hp_lon[hp_lon > 180.0] -= 360.0

    lon2d, lat2d = np.meshgrid(lon, lat)
    ll_rad = np.deg2rad(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    tree = BallTree(ll_rad, metric="haversine")
    hp_rad = np.deg2rad(np.column_stack([hp_lat, hp_lon]))
    _, idx = tree.query(hp_rad, k=1)
    idx = idx.ravel().astype(np.int64)

    if cache_dir is not None:
        np.savez_compressed(path, idx=idx,
                             lat=lat.astype(np.float32),
                             lon=lon.astype(np.float32),
                             nside_high=np.int32(nside_high))
        if verbose:
            print(f"[cache write] {path.name} ({path.stat().st_size/1e6:.2f} MB)")
    return idx


def load_or_build_children(
    cache_dir: Optional[Path],
    nside_high: int,
    nside_low: int,
    nest: bool,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Return list of (npix_step_low, 4) int64 arrays — one per degrade step
    (nside_high → nside_high/2 → ... → nside_low).

    For NESTED ordering: trivially `arange(N).reshape(M, 4)`.
    For RING ordering: parent index via ang2pix(nside/2, *pix2ang(nside)),
    then sort + reshape (one-time setup ~50 ms total at nside_high=256).
    """
    ordering = "nested" if nest else "ring"

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = degrade_cache_path(cache_dir, nside_high, nside_low, ordering)
        if path.exists():
            if verbose:
                print(f"[cache hit] degrade children: {path.name}")
            with np.load(path) as data:
                # mappings are stored under names step0, step1, ...
                return [data[k] for k in sorted(data.files) if k.startswith("step")]
        if verbose:
            print(f"[cache miss] computing children {ordering} {nside_high}→{nside_low} ...")

    # Compute
    mappings: list[np.ndarray] = []
    ns = nside_high
    while ns > nside_low:
        ns_next = ns // 2
        npix = hp.nside2npix(ns)
        if nest:
            children = np.arange(npix, dtype=np.int64).reshape(
                hp.nside2npix(ns_next), 4
            )
        else:  # RING — parent given by ang2pix lookup
            theta, phi = hp.pix2ang(ns, np.arange(npix), nest=False)
            parent = hp.ang2pix(ns_next, theta, phi, nest=False)
            order = np.argsort(parent, kind="stable").astype(np.int64)
            children = order.reshape(hp.nside2npix(ns_next), 4)
        mappings.append(children)
        ns = ns_next

    if cache_dir is not None:
        kwargs = {f"step{i}": m for i, m in enumerate(mappings)}
        np.savez_compressed(path, **kwargs)
        if verbose:
            print(f"[cache write] {path.name} ({path.stat().st_size/1e6:.2f} MB)")
    return mappings
