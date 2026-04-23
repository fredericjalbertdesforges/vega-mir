"""Shannon entropy on symbolic music sequences.

This module implements the marginal Shannon entropy used in the Cygnus
methodology (Jalbert-Desforges, 2026): a 15-symbol scale-degree alphabet,
Jeffreys-Laplace smoothing (alpha = 0.5), log base 2 (bits), and an
optional collapse of consecutive identical degrees ("sans self-repetitions").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

CYGNUS_15_ALPHABET: Final[tuple[str, ...]] = (
    "I", "i", "bII", "II", "bIII", "III", "IV", "iv",
    "#IV", "V", "v", "bVI", "VI", "bVII", "VII",
)


def shannon_entropy(
    probs: NDArray[np.floating] | Sequence[float],
    base: float = 2.0,
) -> float:
    """Shannon entropy of a probability vector.

    Zero-probability entries are dropped (using the convention 0 log 0 = 0).
    The vector is normalized internally if its sum differs from 1.

    Parameters
    ----------
    probs : array_like of float
        Probability or frequency vector.
    base : float, default 2.0
        Logarithm base. ``2`` gives bits, ``e`` gives nats, ``10`` gives hartleys.

    Returns
    -------
    float
        Shannon entropy in the chosen base. Returns ``0.0`` for an empty
        vector or one whose entries sum to zero.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    total = p.sum()
    if total != 1.0:
        p = p / total
    return float(-np.sum(p * np.log(p) / np.log(base)))


def smoothed_probabilities(
    counts: dict[str, float],
    alphabet: Sequence[str],
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """Add-alpha smoothed probability vector over a fixed alphabet.

    Uses ``p_i = (count_i + alpha) / (sum_j count_j + alpha * |alphabet|)``.
    With ``alpha = 0.5`` this is the Jeffreys prior; ``alpha = 1`` is
    Laplace's rule of succession.

    Parameters
    ----------
    counts : dict[str, float]
        Counts (possibly non-integer) for each observed symbol.
    alphabet : sequence of str
        Reference alphabet. Symbols outside ``alphabet`` are ignored.
    alpha : float, default 0.5
        Smoothing parameter (Jeffreys prior).

    Returns
    -------
    ndarray of float, shape (len(alphabet),)
        Normalized probability vector. Zero-length if ``alphabet`` is empty.
    """
    vec = np.array(
        [float(counts.get(symbol, 0)) for symbol in alphabet],
        dtype=np.float64,
    )
    vec = vec + alpha
    total = vec.sum()
    if total <= 0:
        return np.zeros_like(vec)
    return vec / total


def collapse_repetitions(sequence: Iterable[str]) -> list[str]:
    """Collapse consecutive identical symbols into a single occurrence.

    Mirrors the v2 Cygnus methodology where a held chord contributes one
    occurrence to the marginal count.

    Examples
    --------
    >>> collapse_repetitions(["I", "I", "V", "V", "V", "I"])
    ['I', 'V', 'I']
    """
    out: list[str] = []
    for sym in sequence:
        if not out or sym != out[-1]:
            out.append(sym)
    return out


def shannon_scale_degrees(
    sequence: Sequence[str],
    alphabet: Sequence[str] = CYGNUS_15_ALPHABET,
    alpha: float = 0.5,
    base: float = 2.0,
    collapse: bool = True,
) -> float:
    """Marginal Shannon entropy on a scale-degree sequence.

    Defaults match the Cygnus methodology (Jalbert-Desforges, 2026):
    15-symbol alphabet, Jeffreys-Laplace smoothing (``alpha = 0.5``),
    log base 2 (bits), and consecutive-duplicate collapsing
    ("sans self-repetitions"). Symbols outside ``alphabet`` are
    silently dropped.

    Parameters
    ----------
    sequence : sequence of str
        Scale-degree sequence (e.g., ``["I", "V", "I", "IV", ...]``).
    alphabet : sequence of str, default :data:`CYGNUS_15_ALPHABET`
        Reference alphabet. Defaults to the 15-symbol Cygnus alphabet.
    alpha : float, default 0.5
        Smoothing parameter.
    base : float, default 2.0
        Logarithm base (``2`` for bits).
    collapse : bool, default True
        If True, collapse consecutive duplicates before counting.

    Returns
    -------
    float
        Marginal Shannon entropy in the chosen base.

    Examples
    --------
    >>> H = shannon_scale_degrees(["I", "V", "I", "IV", "V", "I"])
    >>> 0 < H < 4  # bounded by log2(15) ~ 3.91
    True
    """
    seq = collapse_repetitions(sequence) if collapse else list(sequence)
    counts: dict[str, float] = dict(Counter(seq))
    probs = smoothed_probabilities(counts, alphabet, alpha=alpha)
    return shannon_entropy(probs, base=base)
