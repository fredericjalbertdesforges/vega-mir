"""Spectral analysis of a tempo curve to classify rubato style.

A tempo curve (BPM-over-time) carries the rhythmic flexibility of a
performance. This module decomposes the centered curve via the
real-valued FFT, locates peaks in the power spectrum, and reports:

* the dominant periods (in sample units) and their normalized power,
* the *periodicity ratio* (sum of peak power over total power),
* a categorical *rubato type* in
  ``{"periodic", "quasi_periodic", "free", "metronomic"}``.

The classification thresholds match the Cygnus methodology
(Jalbert-Desforges, 2026):

* ``periodicity_ratio > 0.5`` → ``"periodic"``
* ``periodicity_ratio > 0.3`` → ``"quasi_periodic"``
* otherwise, ``std(tempo) > 3`` BPM → ``"free"``
* otherwise → ``"metronomic"``

A near-constant tempo (``std < 0.5`` BPM) short-circuits to
``"metronomic"`` regardless of the spectrum.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


class DominantPeriod(NamedTuple):
    """A peak in the tempo power spectrum.

    Attributes
    ----------
    period : float
        Period in sample units (``1 / frequency``). Multiply by your
        sampling step to convert to seconds or beats.
    normalized_power : float
        Power at this peak divided by the maximum power in the spectrum.
    """

    period: float
    normalized_power: float


class RubatoAnalysis(NamedTuple):
    """Spectral analysis of a tempo curve.

    Attributes
    ----------
    rubato_type : str
        One of ``"periodic"``, ``"quasi_periodic"``, ``"free"``,
        ``"metronomic"``.
    periodicity_ratio : float
        Sum of peak power divided by total power (excluding DC).
        Bounded in ``[0, 1]``.
    dominant_periods : tuple of DominantPeriod
        Up to three highest peaks, sorted by spectral order.
    n_samples : int
        Number of valid BPM samples used.
    tempo_mean : float
        Mean BPM across the curve.
    tempo_std : float
        Standard deviation of the BPM curve.
    """

    rubato_type: str
    periodicity_ratio: float
    dominant_periods: tuple[DominantPeriod, ...]
    n_samples: int
    tempo_mean: float
    tempo_std: float


def rubato_spectral(
    bpm_curve: Sequence[float] | NDArray[np.floating],
    min_samples: int = 32,
) -> RubatoAnalysis:
    """Spectral analysis and rubato-type classification of a BPM curve.

    Parameters
    ----------
    bpm_curve : array_like of float
        Sequence of BPM values. Must contain at least ``min_samples``
        positive entries; non-positive entries are silently dropped
        before analysis.
    min_samples : int, default 32
        Minimum number of valid samples required for a meaningful FFT.

    Returns
    -------
    RubatoAnalysis

    Raises
    ------
    ValueError
        If fewer than ``min_samples`` positive BPM values are supplied.
    """
    arr = np.asarray(bpm_curve, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size < min_samples:
        raise ValueError(
            f"need at least {min_samples} positive BPM samples, got {arr.size}"
        )

    tempo_mean = float(np.mean(arr))
    tempo_std = float(np.std(arr))

    if tempo_std < 0.5:
        return RubatoAnalysis(
            rubato_type="metronomic",
            periodicity_ratio=0.0,
            dominant_periods=(),
            n_samples=int(arr.size),
            tempo_mean=tempo_mean,
            tempo_std=tempo_std,
        )

    centered = arr - tempo_mean
    fft = np.fft.rfft(centered)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(centered.size)

    if power.size < 3:
        return RubatoAnalysis(
            rubato_type="free" if tempo_std > 3 else "metronomic",
            periodicity_ratio=0.0,
            dominant_periods=(),
            n_samples=int(arr.size),
            tempo_mean=tempo_mean,
            tempo_std=tempo_std,
        )

    max_power_excl_dc = float(np.max(power[1:])) if power.size > 1 else 0.0
    threshold = 0.1 * max_power_excl_dc
    peaks_no_dc, _ = find_peaks(power[1:], height=threshold)
    peaks = peaks_no_dc + 1  # restore index into the full power array

    dominant: list[DominantPeriod] = []
    if max_power_excl_dc > 0:
        for p in peaks[:3]:
            if p < freqs.size and freqs[p] > 0:
                period = 1.0 / float(freqs[p])
                dominant.append(
                    DominantPeriod(
                        period=period,
                        normalized_power=float(power[p] / max(np.max(power), 1e-10)),
                    )
                )

    if peaks.size > 0:
        valid_peaks = peaks[peaks < power.size]
        periodic_energy = float(np.sum(power[valid_peaks]))
        total_energy = float(np.sum(power[1:]))
        periodicity_ratio = periodic_energy / max(total_energy, 1e-10)
    else:
        periodicity_ratio = 0.0

    if periodicity_ratio > 0.5:
        rubato_type = "periodic"
    elif periodicity_ratio > 0.3:
        rubato_type = "quasi_periodic"
    elif tempo_std > 3:
        rubato_type = "free"
    else:
        rubato_type = "metronomic"

    return RubatoAnalysis(
        rubato_type=rubato_type,
        periodicity_ratio=periodicity_ratio,
        dominant_periods=tuple(dominant),
        n_samples=int(arr.size),
        tempo_mean=tempo_mean,
        tempo_std=tempo_std,
    )
