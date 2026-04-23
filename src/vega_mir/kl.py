"""Kullback-Leibler and Jensen-Shannon divergences on symbolic distributions.

This module implements the asymmetric Kullback-Leibler divergence
``D(P || Q)`` and the symmetric Jensen-Shannon divergence used in the
Cygnus methodology v2 (Jalbert-Desforges, 2026).

Distributions passed to :func:`kl_divergence` and :func:`js_divergence`
must be non-negative; they are normalized internally if their sum
differs from 1. Distributions with zero entries should normally be
smoothed first via :func:`vega_mir.shannon.smoothed_probabilities`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from vega_mir.shannon import smoothed_probabilities


def _validate_pair(
    p: NDArray[np.floating] | Sequence[float],
    q: NDArray[np.floating] | Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p_arr.shape} and {q_arr.shape}"
        )
    if np.any(p_arr < 0) or np.any(q_arr < 0):
        raise ValueError("p and q must be non-negative")
    return p_arr, q_arr


def kl_divergence(
    p: NDArray[np.floating] | Sequence[float],
    q: NDArray[np.floating] | Sequence[float],
    base: float = 2.0,
) -> float:
    """Asymmetric Kullback-Leibler divergence ``D(P || Q)``.

    Computed as ``sum_i p_i * log_b(p_i / q_i)``, with the convention
    ``0 * log 0 = 0``. Returns ``inf`` if there is any index ``i`` such
    that ``q_i = 0`` and ``p_i > 0`` (the divergence is undefined in
    that case). Both vectors are normalized internally if their sums
    differ from 1.

    Parameters
    ----------
    p, q : array_like of float
        Non-negative probability or frequency vectors of the same length.
    base : float, default 2.0
        Logarithm base. ``2`` gives bits, ``e`` gives nats.

    Returns
    -------
    float
        ``D(P || Q)`` in the chosen base.

    Raises
    ------
    ValueError
        If ``p`` and ``q`` have different shapes or contain negative entries.
    """
    p_arr, q_arr = _validate_pair(p, q)
    return float(stats.entropy(p_arr, q_arr, base=base))


def js_divergence(
    p: NDArray[np.floating] | Sequence[float],
    q: NDArray[np.floating] | Sequence[float],
    base: float = 2.0,
) -> float:
    """Jensen-Shannon divergence between two distributions.

    Defined as ``0.5 * D(P || M) + 0.5 * D(Q || M)`` with
    ``M = 0.5 * (P + Q)``. Symmetric in ``P`` and ``Q``; bounded in
    ``[0, log_b 2]`` (i.e. ``[0, 1]`` for ``base = 2``).

    Parameters
    ----------
    p, q : array_like of float
        Non-negative probability or frequency vectors of the same length.
    base : float, default 2.0
        Logarithm base.

    Returns
    -------
    float
        ``JS(P, Q)`` in the chosen base.
    """
    p_arr, q_arr = _validate_pair(p, q)
    if not np.isclose(p_arr.sum(), 1.0):
        p_arr = p_arr / p_arr.sum()
    if not np.isclose(q_arr.sum(), 1.0):
        q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, m, base=base) + 0.5 * kl_divergence(q_arr, m, base=base)


def kl_divergence_from_counts(
    p_counts: dict[str, float],
    q_counts: dict[str, float],
    alphabet: Sequence[str],
    alpha: float = 0.5,
    base: float = 2.0,
) -> float:
    """KL divergence ``D(P || Q)`` from raw counts with shared smoothing.

    Both distributions are smoothed using add-alpha smoothing on the
    same alphabet before the divergence is computed. This guarantees
    that ``q`` has no zeros where ``p`` has support, so the result is
    always finite.

    Parameters
    ----------
    p_counts, q_counts : dict[str, float]
        Counts for each symbol.
    alphabet : sequence of str
        Reference alphabet shared by both distributions.
    alpha : float, default 0.5
        Smoothing parameter (Jeffreys prior, matching Cygnus methodology).
    base : float, default 2.0
        Logarithm base.

    Returns
    -------
    float
    """
    p = smoothed_probabilities(p_counts, alphabet, alpha=alpha)
    q = smoothed_probabilities(q_counts, alphabet, alpha=alpha)
    return kl_divergence(p, q, base=base)


def kl_matrix(
    distributions: Mapping[str, NDArray[np.floating] | Sequence[float]],
    base: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Pairwise asymmetric KL matrix between named probability distributions.

    Returns a nested dict ``matrix[source][target] = D(source || target)``.
    Diagonal entries are ``0.0`` by definition. The matrix is asymmetric
    in general.

    Parameters
    ----------
    distributions : mapping of str to array_like
        Named probability vectors. All must have the same length and
        be non-negative.
    base : float, default 2.0
        Logarithm base.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict; outer key is the source distribution name, inner
        key is the target.
    """
    names = list(distributions.keys())
    matrix: dict[str, dict[str, float]] = {}
    for src in names:
        row: dict[str, float] = {}
        for tgt in names:
            if src == tgt:
                row[tgt] = 0.0
            else:
                row[tgt] = kl_divergence(distributions[src], distributions[tgt], base=base)
        matrix[src] = row
    return matrix
