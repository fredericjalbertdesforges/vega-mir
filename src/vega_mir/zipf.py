"""Zipf's law analysis on symbolic music distributions.

This module fits the model ``log2(P) = C - alpha * log2(rank)`` to a
probability distribution and returns the exponent ``alpha``, the
coefficient of determination ``R^2``, the intercept, and the number of
non-zero points used. Defaults follow the Cygnus methodology v2
(Jalbert-Desforges, 2026): 15-symbol scale-degree alphabet, Jeffreys-
Laplace smoothing (alpha = 0.5), consecutive duplicates collapsed.

Two convenience entry points operate on a scale-degree sequence:

* :func:`zipf_fit_marginal` — fits the 15-point marginal distribution.
* :func:`zipf_fit_transitions` — fits the 225-point joint (bigram)
  distribution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from vega_mir.shannon import (
    CYGNUS_15_ALPHABET,
    collapse_repetitions,
    smoothed_probabilities,
)


class ZipfFit(NamedTuple):
    """Result of a Zipf OLS fit ``log2(P) = C - alpha * log2(rank)``.

    Attributes
    ----------
    alpha : float
        Negative slope of the log-log regression. Positive for a
        decreasing distribution; ``alpha == 1`` is the canonical Zipf law.
    r_squared : float
        Coefficient of determination. Values near 1 indicate a good
        power-law fit.
    intercept : float
        Intercept of the log-log regression (depends on log base).
    n_points : int
        Number of non-zero probability entries used in the fit.
    """

    alpha: float
    r_squared: float
    intercept: float
    n_points: int


def zipf_fit(probs: NDArray[np.floating] | Sequence[float]) -> ZipfFit:
    """OLS fit of ``log2(P) = C - alpha * log2(rank)`` on a probability vector.

    Zero entries are dropped (with smoothing applied beforehand, fixed-
    alphabet distributions normally have no zeros). When fewer than three
    non-zero entries are available the function returns a degenerate fit
    with all numeric fields set to zero and ``n_points`` set to the
    number of non-zero entries.

    Parameters
    ----------
    probs : array_like of float
        Probability or frequency vector. Need not be normalized.

    Returns
    -------
    ZipfFit
        Named tuple with ``alpha``, ``r_squared``, ``intercept``, ``n_points``.
    """
    p = np.asarray(probs, dtype=np.float64)
    sorted_p = np.sort(p)[::-1]
    mask = sorted_p > 0
    n_pos = int(mask.sum())
    if n_pos < 3:
        return ZipfFit(alpha=0.0, r_squared=0.0, intercept=0.0, n_points=n_pos)
    ranks = np.arange(1, n_pos + 1)
    log_r = np.log2(ranks)
    log_p = np.log2(sorted_p[mask])
    slope, intercept, r_value, _p_value, _stderr = stats.linregress(log_r, log_p)
    return ZipfFit(
        alpha=float(-slope),
        r_squared=float(r_value ** 2),
        intercept=float(intercept),
        n_points=n_pos,
    )


def zipf_fit_marginal(
    sequence: Sequence[str],
    alphabet: Sequence[str] = CYGNUS_15_ALPHABET,
    alpha: float = 0.5,
    collapse: bool = True,
) -> ZipfFit:
    """Zipf fit on the marginal distribution of a scale-degree sequence.

    Defaults match the Cygnus methodology v2: 15-symbol alphabet,
    Jeffreys-Laplace smoothing (``alpha = 0.5``), consecutive duplicates
    collapsed.

    Parameters
    ----------
    sequence : sequence of str
        Scale-degree sequence (e.g., ``["I", "V", "I", "IV", ...]``).
    alphabet : sequence of str, default :data:`CYGNUS_15_ALPHABET`
        Reference alphabet. Symbols outside it are dropped.
    alpha : float, default 0.5
        Smoothing parameter (Jeffreys prior).
    collapse : bool, default True
        If True, collapse consecutive duplicates before counting.

    Returns
    -------
    ZipfFit
    """
    seq = collapse_repetitions(sequence) if collapse else list(sequence)
    counts: dict[str, float] = dict(Counter(seq))
    probs = smoothed_probabilities(counts, alphabet, alpha=alpha)
    return zipf_fit(probs)


def zipf_fit_transitions(
    sequence: Sequence[str],
    alphabet: Sequence[str] = CYGNUS_15_ALPHABET,
    alpha: float = 0.5,
    collapse: bool = True,
) -> ZipfFit:
    """Zipf fit on the joint (bigram) distribution of a scale-degree sequence.

    The joint distribution has ``|alphabet|^2`` entries (225 for the
    15-symbol Cygnus alphabet). When ``collapse = True`` self-loops
    ``(d_i, d_i)`` are excluded by construction (consecutive duplicates
    are merged before bigrams are built).

    Parameters
    ----------
    sequence : sequence of str
        Scale-degree sequence.
    alphabet : sequence of str, default :data:`CYGNUS_15_ALPHABET`
    alpha : float, default 0.5
    collapse : bool, default True

    Returns
    -------
    ZipfFit
    """
    seq = collapse_repetitions(sequence) if collapse else list(sequence)
    n = len(alphabet)
    idx = {sym: i for i, sym in enumerate(alphabet)}
    counts = np.zeros((n, n), dtype=np.float64)
    for src, tgt in zip(seq[:-1], seq[1:], strict=False):
        if src in idx and tgt in idx:
            counts[idx[src], idx[tgt]] += 1.0
    smoothed: NDArray[np.float64] = counts + alpha
    total = smoothed.sum()
    if total <= 0:
        return ZipfFit(alpha=0.0, r_squared=0.0, intercept=0.0, n_points=0)
    probs = (smoothed / total).flatten()
    return zipf_fit(probs)
