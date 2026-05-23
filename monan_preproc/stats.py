"""Welford accumulators (NaN-aware, vectorized) + power-transform discovery
for the v3 dataset stats computation.

Two accumulator classes:

  - `IncrementalStatistics` — per-pixel Welford for shape (npix,). Used in
    pass 1 to identify pixels with meaningful variation across timesteps
    (mask building).

  - `GlobalWelford` — scalar Welford. Used in pass 2 to compute global
    (mean, std) over masked pixels × timesteps, with power transform applied.

Both use Chan's parallel-batch update formula so a single (n_t, npix) array
update is one numpy op, not a Python loop over timesteps.

The signed-power transform (`signed_power`) preserves sign and is used to
tame heavy-tailed variables (precipitation, fluxes) before z-scoring.

NaN policy: NaN values are EXCLUDED from accumulators. Downstream, the
normalize step replaces NaN with 0 in z-scored output (the model adapts;
denormalization restores NaN where mask=0).
"""
from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
def signed_power(x: np.ndarray, power: float) -> np.ndarray:
    """Apply signed power transform: sign(x) * |x|^p, preserves sign + zeros.

    Identity if power == 1.0 (no copy).
    """
    if power == 1.0:
        return x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.sign(x) * np.abs(x) ** power


def find_optimal_power(
    data: np.ndarray,
    denominators=(1, 2, 3, 4, 5, 6, 7, 8),
    max_z: float = 5.0,
) -> tuple[float, float]:
    """Find the largest power 1/d (smallest 1/d ⇒ less compression) that brings
    max(|z|) of `data` to ≤ `max_z`.

    Tries 1.0 (identity) first; then 1/2, 1/3, ..., 1/8 in order. Returns the
    first one to meet the target. If none does, returns the best one observed.

    Returns (power, achieved_max_z).
    """
    def zmax(x: np.ndarray) -> float:
        f = x[np.isfinite(x)]
        if len(f) < 100:
            return np.inf
        std = max(float(f.std()), 1e-15)
        return float(np.max(np.abs((f - f.mean()) / std)))

    z0 = zmax(data)
    if z0 <= max_z:
        return 1.0, z0

    best_p, best_z = 1.0, z0
    for d in denominators:
        if d == 1:
            continue
        p = 1.0 / d
        z = zmax(signed_power(data, p))
        if z <= max_z:
            return p, z
        if z < best_z:
            best_p, best_z = p, z
    return best_p, best_z


# ---------------------------------------------------------------------------
# Asinh transform (for sparse / heavy-tailed variables where power fails)
# ---------------------------------------------------------------------------
def asinh_transform(x: np.ndarray, scale: float) -> np.ndarray:
    """y = arcsinh(x / scale).

    Smooth, sign-preserving, sends 0→0. Linear for |x|<<scale; logarithmic
    for |x|>>scale. Bounded growth tames extreme outliers. Inverse: sinh.
    """
    return np.arcsinh(x / scale)


def asinh_inverse(y: np.ndarray, scale: float) -> np.ndarray:
    """x = scale * sinh(y) — exact inverse of asinh_transform."""
    return scale * np.sinh(y)


def find_asinh_scale(samples: np.ndarray, percentile: float = 99.0,
                      min_scale: float = 1e-30) -> float:
    """Pick scale = N-th percentile of |non-zero values|.

    Default 99th percentile: asinh acts ≈ linear for 99% of typical non-zero
    values, only logarithmically compresses the top ~1% extreme outliers.
    Choosing the median (50th) would over-compress; the max would not
    compress at all. 99th is a good "transition point".

    Zeros are filtered out before computing the percentile, so structurally-
    sparse variables (most pixels = 0 most of the time) get a scale based on
    the magnitude of the rare non-zero events that are physically meaningful.
    """
    finite = samples[np.isfinite(samples)]
    nz = finite[np.abs(finite) > min_scale]
    if nz.size < 10:
        # Pathologically empty; fall back to overall max
        return max(float(np.abs(finite).max() if finite.size else 1.0), min_scale)
    return max(float(np.percentile(np.abs(nz), percentile)), min_scale)


def find_optimal_transform(
    samples: np.ndarray,
    max_z: float = 5.0,
    power_denominators=(1, 2, 3, 4, 5, 6, 7, 8),
    asinh_percentile: float = 99.0,
) -> dict:
    """Pick a transform for a variable based on its sample distribution.

    Strategy (deterministic, no comparison ambiguity):
      1. Try `signed_power` (1.0, 1/2, ..., 1/8). If the smallest power that
         brings max|z| ≤ `max_z` exists, return it — power preserves linearity
         for typical values.
      2. Else **always** fall back to `asinh` with scale = 99th percentile of
         |non-zero samples|. asinh growth is O(log x) for large x — bounded by
         construction, more graceful for sparse / heavy-tail data than power
         (which can amplify zero-vs-outlier ratios when applied to fractional
         exponents). Achieved max|z| is reported but not used as a gate; even
         if it exceeds the target, asinh is the better choice numerically.
      3. Pathological case (empty / <100 finite samples): return identity.

    Returns dict with `type`, `achieved_max_z`, and type-specific params:
        type='power' → p
        type='asinh' → scale
    """
    finite_count = int(np.isfinite(samples).sum())
    if finite_count < 100:
        return {"type": "identity", "achieved_max_z": float("nan")}

    # 1) Try signed_power
    p, z_pow = find_optimal_power(samples, power_denominators, max_z)
    if z_pow <= max_z:
        return {"type": "power", "p": float(p), "achieved_max_z": float(z_pow)}

    # 2) Always fall back to asinh — guaranteed-bounded for outliers.
    scale = find_asinh_scale(samples, percentile=asinh_percentile)
    finite = samples[np.isfinite(samples)]
    y = asinh_transform(finite, scale)
    std = max(float(y.std()), 1e-15)
    z_asinh = float(np.max(np.abs((y - y.mean()) / std)))
    return {
        "type": "asinh",
        "scale": float(scale),
        "achieved_max_z": float(z_asinh),
    }


def apply_transform(x: np.ndarray, t: dict) -> np.ndarray:
    """Forward transform per spec dict {type, ...}."""
    tt = t["type"]
    if tt == "power":
        return signed_power(x, t["p"])
    if tt == "asinh":
        return asinh_transform(x, t["scale"])
    return x  # identity


def invert_transform(y: np.ndarray, t: dict) -> np.ndarray:
    """Exact inverse for power / asinh / identity."""
    tt = t["type"]
    if tt == "power":
        p = t["p"]
        if p == 1.0:
            return y
        return np.sign(y) * np.abs(y) ** (1.0 / p)
    if tt == "asinh":
        return asinh_inverse(y, t["scale"])
    return y  # identity


# ---------------------------------------------------------------------------
class IncrementalStatistics:
    """Vectorized per-pixel Welford accumulator (NaN-aware).

    Each pixel keeps its own (count, mean, M2) state in float64 / int64.
    `update_batch` adds a (n_t, npix) array via Chan's parallel formula in
    a single numpy operation (no Python loop over timesteps).
    """

    __slots__ = ("count", "mean", "M2")

    def __init__(self, npix: int):
        self.count = np.zeros(npix, dtype=np.int64)
        self.mean = np.zeros(npix, dtype=np.float64)
        self.M2 = np.zeros(npix, dtype=np.float64)

    def update_batch(self, x: np.ndarray) -> None:
        """x: (n_t, npix) or (npix,). NaN cells are ignored per-pixel."""
        if x.ndim == 1:
            x = x[None, :]
        valid = np.isfinite(x)                                   # (n_t, npix) bool
        n_B = valid.sum(axis=0).astype(np.int64)                 # (npix,) per-pixel batch count

        # Batch mean per pixel (avoid div-by-0 with safe denom)
        sum_B = np.where(valid, x, 0.0).sum(axis=0)              # (npix,)
        n_B_safe = np.where(n_B > 0, n_B, 1)
        mean_B = np.where(n_B > 0, sum_B / n_B_safe, 0.0)        # (npix,)

        # Batch M2 (sum of squared deviations from batch mean, NaN cells → 0)
        diff = np.where(valid, x - mean_B[None, :], 0.0)
        M2_B = (diff * diff).sum(axis=0)                         # (npix,)

        # Chan's parallel combine
        n_A = self.count
        n_AB = n_A + n_B
        n_AB_safe = np.where(n_AB > 0, n_AB, 1)
        delta = mean_B - self.mean
        with np.errstate(invalid="ignore"):
            new_mean = (n_A * self.mean + n_B * mean_B) / n_AB_safe
            new_M2 = self.M2 + M2_B + delta * delta * n_A * n_B / n_AB_safe
        # Update only where this batch contributed
        upd = n_B > 0
        self.mean = np.where(upd, new_mean, self.mean)
        self.M2 = np.where(upd, new_M2, self.M2)
        self.count = n_AB

    def merge(self, other: "IncrementalStatistics") -> None:
        """Merge another IncrementalStatistics into self (Chan's formula)."""
        n_A = self.count
        n_B = other.count
        n_AB = n_A + n_B
        n_AB_safe = np.where(n_AB > 0, n_AB, 1)
        delta = other.mean - self.mean
        new_mean = (n_A * self.mean + n_B * other.mean) / n_AB_safe
        new_M2 = self.M2 + other.M2 + delta * delta * n_A * n_B / n_AB_safe
        upd = n_B > 0
        self.mean = np.where(upd, new_mean, self.mean)
        self.M2 = np.where(upd, new_M2, self.M2)
        self.count = n_AB

    def get_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) per pixel; NaN where count < 2."""
        mean = np.full_like(self.mean, np.nan)
        std = np.full_like(self.M2, np.nan)
        ok = self.count >= 2
        if ok.any():
            mean[ok] = self.mean[ok]
            std[ok] = np.sqrt(np.maximum(self.M2[ok] / (self.count[ok] - 1), 0.0))
        return mean, std


# ---------------------------------------------------------------------------
class GlobalWelford:
    """Scalar Welford for global (mean, std) over a flat selection of pixels.

    Use `update_masked(x, mask)` to fold in a (n_t, npix) batch keeping only
    pixels where `mask` is True AND value is finite.
    """

    __slots__ = ("count", "mean", "M2", "vmin", "vmax")

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.vmin = float("inf")
        self.vmax = float("-inf")

    def update_masked(self, x: np.ndarray, mask: np.ndarray) -> None:
        """x: (n_t, npix) or (npix,). mask: (npix,) bool."""
        if x.ndim == 1:
            x = x[None, :]
        valid = mask[None, :] & np.isfinite(x)                   # (n_t, npix)
        v = x[valid]
        if v.size == 0:
            return
        n_new = int(v.size)
        new_mean = float(v.mean())
        new_M2 = float(((v - new_mean) ** 2).sum())
        self.vmin = min(self.vmin, float(v.min()))
        self.vmax = max(self.vmax, float(v.max()))
        if self.count == 0:
            self.count, self.mean, self.M2 = n_new, new_mean, new_M2
            return
        total = self.count + n_new
        delta = new_mean - self.mean
        self.mean = (self.count * self.mean + n_new * new_mean) / total
        self.M2 += new_M2 + delta * delta * self.count * n_new / total
        self.count = total

    def merge(self, other: "GlobalWelford") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.M2 = other.count, other.mean, other.M2
            self.vmin, self.vmax = other.vmin, other.vmax
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean = (self.count * self.mean + other.count * other.mean) / total
        self.M2 += other.M2 + delta * delta * self.count * other.count / total
        self.count = total
        self.vmin = min(self.vmin, other.vmin)
        self.vmax = max(self.vmax, other.vmax)

    def get_mean_std(self) -> tuple[float, float]:
        if self.count < 2:
            return 0.0, 0.0
        return float(self.mean), float(np.sqrt(max(self.M2 / (self.count - 1), 0.0)))


# ---------------------------------------------------------------------------
def build_mask_from_per_pixel_stats(
    mean: np.ndarray,
    std: np.ndarray,
    abs_std_threshold: float = 1e-3,
    rel_std_threshold: float = 0.01,
) -> np.ndarray:
    """Build a per-pixel validity mask from per-pixel mean/std.

    A pixel is `1.0` (valid) iff:
        std is finite, AND
        (std > abs_std_threshold OR cv > rel_std_threshold)

    where cv = std / |mean| (coefficient of variation). Zero or all-NaN pixels
    fail both thresholds and become 0.0 (excluded). Returns float32 (npix,).
    """
    mask = np.zeros(mean.shape, dtype=np.float32)
    has_data = ~np.isnan(std)
    has_var_abs = std > abs_std_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 1e-15, std / np.abs(mean), 0.0)
    has_var_rel = cv > rel_std_threshold
    mask[has_data & (has_var_abs | has_var_rel)] = 1.0
    return mask
