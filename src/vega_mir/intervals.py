"""Statistical fits and summary statistics for interval distributions.

Given a sample of melodic intervals (in semitones), this module fits
the Exponential and Laplace families via maximum likelihood, scores
each fit with the Kolmogorov-Smirnov statistic, and returns the
best-fitting model along with summary statistics.

Used in the Cygnus methodology (Jalbert-Desforges, 2026) to summarise
melodic motion. Conjunct motion (small intervals) typically dominates
piano repertoire, leading to a steeply decreasing distribution that is
well captured by either Exponential or Laplace models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class IntervalAnalysis(NamedTuple):
    """Result of an interval distribution fit and summary.

    Attributes
    ----------
    best_fit : str
        Name of the best-fitting distribution (``"exponential"`` or
        ``"laplace"``), chosen by lowest KS statistic.
    best_ks : float
        KS statistic of the winning fit.
    exponential_lambda : float
        Rate parameter ``lambda`` of the Exponential fit (``1 / scale``).
    exponential_ks : float
        KS statistic of the Exponential fit.
    laplace_mu : float
        Location parameter ``mu`` of the Laplace fit.
    laplace_b : float
        Scale parameter ``b`` of the Laplace fit.
    laplace_ks : float
        KS statistic of the Laplace fit.
    mean : float
        Sample mean of the absolute intervals.
    std : float
        Sample standard deviation of the absolute intervals.
    pct_conjunct : float
        Percentage of intervals with ``|interval| <= 2`` (conjunct motion).
    pct_leaps : float
        Percentage of intervals with ``|interval| > 4`` (leaps).
    n : int
        Number of intervals in the sample.
    """

    best_fit: str
    best_ks: float
    exponential_lambda: float
    exponential_ks: float
    laplace_mu: float
    laplace_b: float
    laplace_ks: float
    mean: float
    std: float
    pct_conjunct: float
    pct_leaps: float
    n: int


def fit_intervals(intervals: Sequence[float] | NDArray[np.floating]) -> IntervalAnalysis:
    """Fit Exponential and Laplace distributions to an interval sample.

    Intervals are converted to absolute values internally. Both fits use
    maximum likelihood (via :func:`scipy.stats.expon.fit` and
    :func:`scipy.stats.laplace.fit`) and are scored by the
    Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    intervals : array_like of float
        Melodic interval sample in semitones (signed or absolute). Must
        contain at least 10 entries.

    Returns
    -------
    IntervalAnalysis

    Raises
    ------
    ValueError
        If fewer than 10 intervals are supplied.
    """
    arr = np.abs(np.asarray(intervals, dtype=np.float64))
    if arr.size < 10:
        raise ValueError(f"need at least 10 intervals, got {arr.size}")

    loc_exp, scale_exp = stats.expon.fit(arr)
    ks_exp, _p_exp = stats.kstest(arr, "expon", args=(loc_exp, scale_exp))
    lam = 1.0 / max(scale_exp, 1e-10)

    loc_lap, scale_lap = stats.laplace.fit(arr)
    ks_lap, _p_lap = stats.kstest(arr, "laplace", args=(loc_lap, scale_lap))

    if ks_exp <= ks_lap:
        best_fit, best_ks = "exponential", float(ks_exp)
    else:
        best_fit, best_ks = "laplace", float(ks_lap)

    return IntervalAnalysis(
        best_fit=best_fit,
        best_ks=best_ks,
        exponential_lambda=float(lam),
        exponential_ks=float(ks_exp),
        laplace_mu=float(loc_lap),
        laplace_b=float(scale_lap),
        laplace_ks=float(ks_lap),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        pct_conjunct=float(np.mean(arr <= 2) * 100),
        pct_leaps=float(np.mean(arr > 4) * 100),
        n=int(arr.size),
    )


def reconstruct_sample(
    distribution: Mapping[int | str, float],
    sample_size: int = 10000,
) -> NDArray[np.float64]:
    """Reconstruct an absolute-interval sample from a distribution dict.

    Each entry ``(interval, weight)`` contributes
    ``max(1, int(weight * sample_size))`` copies of ``abs(int(interval))``
    to the output array. Useful when the source data is stored as a
    proportion dict (e.g., the ``melodic.interval_distribution`` block of
    a Cygnus profile) rather than a raw interval sample.

    Parameters
    ----------
    distribution : mapping of int or str to float
        Maps signed or unsigned intervals (semitones) to their weight.
        Weights are typically proportions summing to 1, but any
        non-negative weight is accepted.
    sample_size : int, default 10000
        Target sample size used to convert weights to integer counts.
        Each entry contributes at least one copy regardless of weight.

    Returns
    -------
    ndarray of float
        Absolute-interval sample.
    """
    out: list[int] = []
    for k, w in distribution.items():
        iv = abs(int(k))
        count = max(1, int(w * sample_size))
        out.extend([iv] * count)
    return np.array(out, dtype=np.float64)
