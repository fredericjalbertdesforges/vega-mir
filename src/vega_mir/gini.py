"""Gini coefficient on symbolic music distributions.

The Gini coefficient measures inequality of a non-negative sample.
For a distribution it answers: "how concentrated is the mass?"
Returns ``0`` for perfectly uniform values and approaches
``(n - 1) / n`` (the maximum on ``n`` non-negative entries) for the
most concentrated possible distribution.

The Gini coefficient complements Shannon entropy: both are zero for a
deterministic distribution and reach their respective extrema for a
uniform one, but Gini measures *inequality of mass* whereas entropy
measures *uncertainty*.

Used in the Cygnus methodology (Jalbert-Desforges, 2026) to summarise
harmonic concentration, dynamic range, articulation balance, and other
profile dimensions in a single scalar bounded in ``[0, 1)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def gini(values: NDArray[np.floating] | Sequence[float]) -> float:
    """Gini coefficient of a non-negative sample.

    Uses the standard sorted formula
    ``G = (2 * sum_i i * x_(i)) / (n * sum_i x_i) - (n + 1) / n``,
    where ``x_(i)`` is the i-th smallest value (1-indexed).

    Parameters
    ----------
    values : array_like of float
        Non-negative values.

    Returns
    -------
    float
        Gini coefficient in ``[0, (n - 1) / n]``. Returns ``0.0`` if the
        array is empty or all values are zero.

    Raises
    ------
    ValueError
        If any value is negative.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini is only defined for non-negative values")
    total = arr.sum()
    if total == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) - (n + 1) * total) / (n * total))


def gini_from_counts(counts: Mapping[str, float]) -> float:
    """Gini coefficient on the values of a count dict (keys are ignored).

    Convenience for the common case of a frequency dictionary indexed by
    symbol.
    """
    return gini(list(counts.values()))


def gini_multi(
    distributions: Mapping[str, Sequence[float] | NDArray[np.floating]],
) -> dict[str, float]:
    """Gini coefficient on each of several named distributions.

    Returns a dict ``{name: gini_value}``. Useful for summarising
    concentration across multiple profile dimensions in a single call.
    """
    return {name: gini(values) for name, values in distributions.items()}
