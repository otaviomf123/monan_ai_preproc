"""HEALPix regrid + hierarchical degradation utilities.

Pipeline: regular lat-lon → nside_high (BallTree NN) → nside_low (nanmean of
4 children at each step).

Default ordering is **RING** for compatibility with the existing MONAN-IA
model + DA infrastructure (BallTree-based convolutions are ordering-agnostic;
healpy/IFS default to RING). NESTED is also supported via `nest=True`.

Both the BallTree regrid index and the hierarchical-degrade children mapping
are cached on-disk (see `cache.py`). Subsequent calls / parallel workers just
reload — no recomputation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import healpy as hp
import numpy as np

from .cache import load_or_build_children, load_or_build_regrid_indices


class LatLonToHEALPix:
    """Nearest-neighbor regrid from regular lat-lon to HEALPix.

    Parameters
    ----------
    nside : int
        HEALPix nside (target high-resolution before degrade). Power of 2.
    nest : bool, default False
        Pixel ordering (False = RING, the project default).
    cache_dir : Path, optional
        Directory where the BallTree-derived index array is persisted.
    """

    def __init__(self, nside: int = 256, nest: bool = False,
                 cache_dir: Optional[Path] = None):
        if nside & (nside - 1):
            raise ValueError(f"nside must be a power of 2, got {nside}")
        self.nside = nside
        self.nest = nest
        self.npix = hp.nside2npix(nside)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Pixel centres (lat/lon) — used for downstream coords; cheap to compute
        theta, phi = hp.pix2ang(nside, np.arange(self.npix), nest=self.nest)
        self.lat = 90.0 - np.degrees(theta)
        self.lon = np.degrees(phi)
        self.lon[self.lon > 180.0] -= 360.0
        self._idx_for_grid: dict = {}

    def _indices_for(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return the npix-length index array mapping each HEALPix pixel to
        its nearest source-grid lat-lon point."""
        key = (lat.size, lon.size, float(lat[0]), float(lat[-1]),
               float(lon[0]), float(lon[-1]))
        if key in self._idx_for_grid:
            return self._idx_for_grid[key]
        idx = load_or_build_regrid_indices(
            cache_dir=self.cache_dir,
            lat=lat, lon=lon,
            nside_high=self.nside,
            nest=self.nest,
            verbose=False,
        )
        self._idx_for_grid[key] = idx
        return idx

    def regrid(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Regrid (..., n_lat, n_lon) → (..., n_pix). Preserves leading dims."""
        idx = self._indices_for(lat, lon)
        if data.shape[-2:] == (lat.size, lon.size):
            flat = data.reshape(*data.shape[:-2], -1)
        elif data.shape[-1] == lat.size * lon.size:
            flat = data
        else:
            raise ValueError(
                f"Data shape {data.shape} does not match lat={lat.size}, lon={lon.size}"
            )
        return flat[..., idx]


class HierarchicalDegrader:
    """Reduce nside_high → nside_low via nanmean of 4 children per step.

    NaN-aware (`np.nanmean`): limited-domain variables (sst, soil, sea ice)
    propagate masks correctly along coastlines.
    """

    def __init__(self, nside_high: int, nside_low: int, nest: bool = False,
                 cache_dir: Optional[Path] = None):
        if nside_high <= nside_low:
            raise ValueError(f"nside_high ({nside_high}) must exceed nside_low ({nside_low})")
        for n in (nside_high, nside_low):
            if n & (n - 1):
                raise ValueError(f"nside must be a power of 2, got {n}")
        self.nside_high = nside_high
        self.nside_low = nside_low
        self.nest = nest
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.mappings: list[np.ndarray] = load_or_build_children(
            cache_dir=self.cache_dir,
            nside_high=nside_high,
            nside_low=nside_low,
            nest=nest,
            verbose=False,
        )

    def __call__(self, x: np.ndarray, pix_axis: int = -1) -> np.ndarray:
        """Reduce array along its pixel axis from nside_high to nside_low.

        Returns float32. NaN-aware: if all 4 children are NaN, output is NaN.
        """
        out = np.moveaxis(x, pix_axis, -1) if pix_axis != -1 else x
        for children in self.mappings:
            gathered = out[..., children]  # (..., npix_low, 4)
            with np.errstate(invalid="ignore"):
                out = np.nanmean(gathered, axis=-1).astype(np.float32)
        if pix_axis != -1:
            out = np.moveaxis(out, -1, pix_axis)
        return out


def healpix_coords(nside: int, nest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays in degrees for every pixel of an nside grid.

    Default ordering is RING (`nest=False`) to match the project convention.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi)
    lon[lon > 180.0] -= 360.0
    return lat.astype(np.float32), lon.astype(np.float32)
