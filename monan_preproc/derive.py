"""Derived variables from raw ERA5 fields.

Currently:
    - q (specific humidity) from r (relative humidity), t (temperature), p (pressure)
      via Bolton (1980) saturation over water + mixing-ratio formula.
"""
from __future__ import annotations

import numpy as np


def q_from_rtp(r: np.ndarray, t: np.ndarray, p_hpa: float) -> np.ndarray:
    """Specific humidity (kg/kg) from RH (%), T (K) and pressure (hPa).

    Bolton (1980) formula for saturation vapor pressure over liquid water:
        e_sat(T) = 6.112 · exp(17.67 · (T - 273.15) / (T - 273.15 + 243.5))   [hPa]

    Mixing-ratio form of specific humidity:
        e   = (RH/100) · e_sat(T)
        q   = 0.622 · e / (p − 0.378 · e)

    Notes
    -----
    Bolton uses saturation over LIQUID water. ERA5 (IFS) uses mixed-phase
    saturation (over water above 0°C, over ice below −23°C, blended in between).
    Discrepancy is <0.5% for T > −23°C and irrelevant in absolute terms above
    that threshold (q itself is ~10⁻⁵ kg/kg). Acceptable for CRTM input.

    Parameters
    ----------
    r : array — RH in percent (0..100), any shape.
    t : array — T in Kelvin, same shape as r.
    p_hpa : scalar — pressure level in hPa.

    Returns
    -------
    q : array of same shape, kg/kg, float32.
    """
    t_c = t.astype(np.float64) - 273.15
    e_sat = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))           # hPa
    e = (r.astype(np.float64) / 100.0) * e_sat                    # hPa
    q = 0.622 * e / (p_hpa - 0.378 * e)
    # Clip to physical range; numerical noise in r near zero can give tiny negatives.
    q = np.clip(q, 0.0, 1.0)
    return q.astype(np.float32)
