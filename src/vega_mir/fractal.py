"""Higuchi fractal dimension of a 1-D time series.

Implements the algorithm from Higuchi (1988), "Approach to an irregular
time series on the basis of the fractal theory". The fractal dimension
``D`` is recovered from the slope of the log-log regression
``log <L(k)> versus log k`` over scales ``k = 1, ..., k_max``.

Interpretation:

* ``D ≈ 1`` for smooth, low-complexity signals (e.g., sine, ramp).
* ``D ≈ 1.5`` for fractional Brownian motion with Hurst exponent 0.5.
* ``D ≈ 2`` for white noise (maximally rough 1-D signals).

Used in the Cygnus methodology (Jalbert-Desforges, 2026) on per-piece
energy curves to summarise textural complexity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class FractalDimension(NamedTuple):
    """Higuchi fractal dimension of a time series.

    Attributes
    ----------
    dimension : float
        Fractal dimension ``D``, computed as ``-slope`` of the log-log
        regression. For natural signals this is bounded approximately
        in ``[1, 2]``.
    r_squared : float
        Coefficient of determination of the log-log regression. Values
        close to ``1`` indicate a clean power-law scaling. Returns
        ``0.0`` together with ``dimension = 0.0`` when fewer than three
        valid scales remain (e.g., a constant or near-constant series).
    n_points : int
        Number of ``(k, L(k))`` points retained in the regression after
        dropping scales with ``L(k) = 0``.
    k_max : int
        Maximum scale parameter used.
    """

    dimension: float
    r_squared: float
    n_points: int
    k_max: int


def higuchi_fractal_dimension(
    time_series: Sequence[float] | NDArray[np.floating],
    k_max: int = 10,
) -> FractalDimension:
    """Higuchi fractal dimension of a 1-D time series.

    For each scale ``k = 1, ..., k_max``, the algorithm builds ``k``
    interleaved subsequences (one per starting offset ``m = 1, ..., k``),
    measures the normalized "curve length" of each, and averages over
    ``m``. The fractal dimension is the negative slope of the OLS fit
    of ``log <L(k)>`` against ``log k``.

    Parameters
    ----------
    time_series : array_like of float
        1-D real-valued time series. Must contain at least
        ``2 * k_max`` samples.
    k_max : int, default 10
        Maximum scale to use.

    Returns
    -------
    FractalDimension

    Raises
    ------
    ValueError
        If ``k_max < 2`` or the series is too short
        (``len(time_series) < 2 * k_max``).
    """
    if k_max < 2:
        raise ValueError(f"k_max must be at least 2, got {k_max}")
    x = np.asarray(time_series, dtype=np.float64)
    n = x.size
    if n < 2 * k_max:
        raise ValueError(
            f"time_series must have at least 2 * k_max = {2 * k_max} samples, got {n}"
        )

    l_values: list[float] = []
    k_values: list[int] = []

    for k in range(1, k_max + 1):
        l_k = 0.0
        count = 0
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if indices.size < 2:
                continue
            sum_diffs = float(np.sum(np.abs(np.diff(x[indices]))))
            n_seg = indices.size - 1
            if n_seg > 0:
                l_mk = sum_diffs * (n - 1) / (k * n_seg * k)
                l_k += l_mk
                count += 1
        if count > 0:
            l_k /= count
            if l_k > 0:
                l_values.append(l_k)
                k_values.append(k)

    if len(l_values) < 3:
        return FractalDimension(
            dimension=0.0,
            r_squared=0.0,
            n_points=len(l_values),
            k_max=k_max,
        )

    log_k = np.log(np.array(k_values, dtype=np.float64))
    log_l = np.log(np.array(l_values, dtype=np.float64))
    slope, _intercept, r_value, _p, _stderr = stats.linregress(log_k, log_l)

    return FractalDimension(
        dimension=float(-slope),
        r_squared=float(r_value ** 2),
        n_points=len(l_values),
        k_max=k_max,
    )
