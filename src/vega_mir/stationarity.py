"""Chi-squared contingency test for stationarity of a symbolic sequence.

Splits a sequence into ``n_segments`` equal parts and tests whether the
symbol distribution is stable across segments via Pearson's chi-squared
test of independence. The effect size is reported as Cramer's V, a
divergence-like statistic in ``[0, 1]`` (0 = perfectly stationary,
1 = each segment uses a disjoint vocabulary).

Used in the Cygnus methodology (Jalbert-Desforges, 2026) to summarise
within-piece harmonic stability. Conceptually this is a *temporal
divergence* — it asks whether the empirical distribution of the early
section diverges from that of the late section — which places it in the
information-theoretic family alongside KL and JS.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy import stats


class StationarityResult(NamedTuple):
    """Result of a chi-squared stationarity test on a symbolic sequence.

    Attributes
    ----------
    chi2 : float
        Pearson chi-squared statistic.
    p_value : float
        Two-sided p-value of the chi-squared test.
    dof : int
        Degrees of freedom (``(n_rows - 1) * (n_cols - 1)``).
    cramers_v : float
        Cramer's V effect size in ``[0, 1]``. ``0`` means the contingency
        table reveals no association between segment and symbol (the
        sequence is stationary); higher values indicate stronger
        non-stationarity.
    n_segments : int
        Number of segments the sequence was split into.
    n_observations : int
        Total number of observations in the contingency table after
        dropping zero-sum columns.
    is_stationary : bool
        ``True`` if ``p_value > significance``, the conventional
        non-rejection of the stationarity null hypothesis.
    """

    chi2: float
    p_value: float
    dof: int
    cramers_v: float
    n_segments: int
    n_observations: int
    is_stationary: bool


def stationarity_test(
    sequence: Sequence[str],
    n_segments: int = 4,
    significance: float = 0.05,
) -> StationarityResult:
    """Chi-squared contingency test for stationarity of a symbolic sequence.

    The sequence is split into ``n_segments`` non-overlapping equal-size
    segments (the trailing remainder, if any, is dropped). A contingency
    table of shape ``(n_segments, n_distinct_symbols)`` is built, columns
    that are entirely zero are dropped, and Pearson's chi-squared test of
    independence is run. The effect size is reported as Cramer's V.

    Parameters
    ----------
    sequence : sequence of str
        Symbolic sequence (typically a chord progression).
    n_segments : int, default 4
        Number of equal segments. Must be at least 2.
    significance : float, default 0.05
        Threshold for the ``is_stationary`` flag.

    Returns
    -------
    StationarityResult

    Raises
    ------
    ValueError
        If ``n_segments < 2``, the sequence is too short to support
        ``n_segments`` segments of at least five observations each, or
        the sequence contains fewer than two distinct symbols (no
        variability to test).
    """
    if n_segments < 2:
        raise ValueError(f"n_segments must be at least 2, got {n_segments}")
    seq = list(sequence)
    if len(seq) < n_segments * 5:
        raise ValueError(
            f"sequence too short for {n_segments} segments of >= 5 observations "
            f"(got len={len(seq)}, need >= {n_segments * 5})"
        )

    segment_size = len(seq) // n_segments
    segments = [seq[i * segment_size:(i + 1) * segment_size] for i in range(n_segments)]

    symbols = sorted(set(seq))
    if len(symbols) < 2:
        raise ValueError("sequence must contain at least two distinct symbols")

    contingency = np.array(
        [[Counter(segment).get(s, 0) for s in symbols] for segment in segments],
        dtype=np.int64,
    )

    col_sums = contingency.sum(axis=0)
    contingency = contingency[:, col_sums > 0]
    if contingency.shape[1] < 2:
        raise ValueError("sequence must contain at least two symbols active in segments")

    chi2, p_value, dof, _expected = stats.chi2_contingency(contingency)
    n = int(contingency.sum())
    k = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0

    return StationarityResult(
        chi2=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        cramers_v=cramers_v,
        n_segments=n_segments,
        n_observations=n,
        is_stationary=bool(p_value > significance),
    )
