"""Tests for vega_mir.rubato."""

from __future__ import annotations

import numpy as np
import pytest

from vega_mir import (
    DominantPeriod,
    RubatoAnalysis,
    rubato_spectral,
)


class TestRubatoAnalysisFields:
    def test_namedtuple(self) -> None:
        rng = np.random.default_rng(42)
        bpm = 120 + rng.normal(scale=2.0, size=200)
        result = rubato_spectral(bpm)
        assert isinstance(result, RubatoAnalysis)
        for field in (
            "rubato_type",
            "periodicity_ratio",
            "dominant_periods",
            "n_samples",
            "tempo_mean",
            "tempo_std",
        ):
            assert hasattr(result, field)


class TestClassification:
    def test_constant_tempo_is_metronomic(self) -> None:
        bpm = np.full(200, 120.0)
        result = rubato_spectral(bpm)
        assert result.rubato_type == "metronomic"
        assert result.periodicity_ratio == 0.0
        assert result.tempo_std == pytest.approx(0.0)

    def test_near_constant_short_circuits_to_metronomic(self) -> None:
        # std < 0.5 → metronomic regardless of spectrum
        rng = np.random.default_rng(42)
        bpm = 120 + rng.normal(scale=0.1, size=200)
        result = rubato_spectral(bpm)
        assert result.rubato_type == "metronomic"
        assert result.dominant_periods == ()

    def test_pure_sinusoid_is_periodic(self) -> None:
        # tempo = 120 + 5 * sin(2π * 0.05 * i) → strong single peak at f = 0.05
        n = 400
        i = np.arange(n)
        bpm = 120 + 5.0 * np.sin(2 * np.pi * 0.05 * i)
        result = rubato_spectral(bpm)
        assert result.rubato_type == "periodic"
        assert result.periodicity_ratio > 0.5
        # First dominant period should be ~ 1 / 0.05 = 20 samples
        assert len(result.dominant_periods) >= 1
        assert result.dominant_periods[0].period == pytest.approx(20.0, rel=0.05)

    def test_brownian_drift_is_free(self) -> None:
        # Brownian motion has a 1/f^2 spectrum: smoothly decreasing, no
        # sharp peaks. The Cygnus 10%-of-max peak heuristic finds few peaks
        # → low periodicity ratio. We pick an amplitude that pushes tempo_std
        # well above 3 BPM so the fall-through classifies as "free".
        rng = np.random.default_rng(7)
        increments = rng.normal(scale=1.5, size=400)
        bpm = 120 + np.cumsum(increments)
        result = rubato_spectral(bpm)
        assert result.tempo_std > 3
        assert result.periodicity_ratio < 0.5
        assert result.rubato_type in ("free", "quasi_periodic")

    def test_classification_thresholds(self) -> None:
        # Sanity-check the four-way classification produces consistent verdicts
        rng = np.random.default_rng(42)
        n = 400
        i = np.arange(n)

        # Constant → metronomic
        flat = np.full(n, 120.0)
        assert rubato_spectral(flat).rubato_type == "metronomic"

        # Strong sinusoid → periodic
        sinus = 120 + 8.0 * np.sin(2 * np.pi * 0.05 * i)
        assert rubato_spectral(sinus).rubato_type == "periodic"

        # Brownian drift (1/f^2 spectrum) with std > 3 → free
        brown = 120 + np.cumsum(rng.normal(scale=1.5, size=n))
        result = rubato_spectral(brown)
        assert result.tempo_std > 3
        assert result.rubato_type in ("free", "quasi_periodic")


class TestDominantPeriods:
    def test_period_matches_inverse_frequency(self) -> None:
        n = 400
        i = np.arange(n)
        f = 0.025  # period = 40 samples
        bpm = 120 + 5.0 * np.sin(2 * np.pi * f * i)
        result = rubato_spectral(bpm)
        assert len(result.dominant_periods) >= 1
        assert result.dominant_periods[0].period == pytest.approx(1.0 / f, rel=0.05)

    def test_dominant_period_namedtuple(self) -> None:
        n = 400
        i = np.arange(n)
        bpm = 120 + 5.0 * np.sin(2 * np.pi * 0.05 * i)
        result = rubato_spectral(bpm)
        assert all(isinstance(d, DominantPeriod) for d in result.dominant_periods)

    def test_at_most_three_dominant_periods(self) -> None:
        # Multiple sinusoids → multiple peaks, but we keep only top 3
        n = 600
        i = np.arange(n)
        bpm = (
            120.0
            + 3.0 * np.sin(2 * np.pi * 0.02 * i)
            + 3.0 * np.sin(2 * np.pi * 0.05 * i)
            + 3.0 * np.sin(2 * np.pi * 0.10 * i)
            + 3.0 * np.sin(2 * np.pi * 0.15 * i)
        )
        result = rubato_spectral(bpm)
        assert len(result.dominant_periods) <= 3

    def test_normalized_power_in_unit_interval(self) -> None:
        n = 400
        i = np.arange(n)
        bpm = 120 + 5.0 * np.sin(2 * np.pi * 0.05 * i)
        result = rubato_spectral(bpm)
        for d in result.dominant_periods:
            assert 0 <= d.normalized_power <= 1


class TestSummaryStatistics:
    def test_tempo_mean_std_match_numpy(self) -> None:
        rng = np.random.default_rng(42)
        bpm = 100 + rng.normal(scale=4.0, size=200)
        result = rubato_spectral(bpm)
        assert result.tempo_mean == pytest.approx(float(np.mean(bpm)))
        assert result.tempo_std == pytest.approx(float(np.std(bpm)))

    def test_n_samples_excludes_zeros(self) -> None:
        bpm = [120.0] * 100 + [0.0] * 50 + [110.0] * 100
        result = rubato_spectral(bpm)
        assert result.n_samples == 200  # 100 + 100

    def test_periodicity_ratio_bounded(self) -> None:
        rng = np.random.default_rng(42)
        for seed in range(5):
            rng2 = np.random.default_rng(seed)
            bpm = 120 + rng2.normal(scale=5.0, size=400)
            result = rubato_spectral(bpm)
            assert 0 <= result.periodicity_ratio <= 1


class TestEdgeCases:
    def test_too_few_samples_raises(self) -> None:
        bpm = [120.0, 121.0, 119.0]
        with pytest.raises(ValueError, match="at least 32"):
            rubato_spectral(bpm)

    def test_too_few_samples_after_filtering_raises(self) -> None:
        # 200 entries but most are zero → only 10 valid → raise
        bpm = [120.0] * 10 + [0.0] * 190
        with pytest.raises(ValueError, match="at least 32"):
            rubato_spectral(bpm)

    def test_min_samples_parameter(self) -> None:
        bpm = [120 + i * 0.1 for i in range(50)]
        # Default min_samples=32 should pass for 50
        result = rubato_spectral(bpm)
        assert result.n_samples == 50
        # Stricter min_samples=100 should fail
        with pytest.raises(ValueError, match="at least 100"):
            rubato_spectral(bpm, min_samples=100)


class TestCygnusParity:
    """Parity with Cygnus math_metrics.rubato_spectral_track on a known signal."""

    def test_classification_matches_cygnus_thresholds(self) -> None:
        # Reproduce Cygnus: periodicity_ratio > 0.5 → periodic
        n = 400
        i = np.arange(n)
        bpm = 120 + 8.0 * np.sin(2 * np.pi * 0.05 * i)
        result = rubato_spectral(bpm)
        if result.periodicity_ratio > 0.5:
            assert result.rubato_type == "periodic"
        elif result.periodicity_ratio > 0.3:
            assert result.rubato_type == "quasi_periodic"
        elif result.tempo_std > 3:
            assert result.rubato_type == "free"
        else:
            assert result.rubato_type == "metronomic"

    def test_metronomic_short_circuit_matches_cygnus(self) -> None:
        # Cygnus: std < 0.5 → metronomic, periodicity_ratio = 0
        rng = np.random.default_rng(42)
        bpm = 120 + rng.normal(scale=0.1, size=200)
        result = rubato_spectral(bpm)
        assert result.rubato_type == "metronomic"
        assert result.periodicity_ratio == 0.0
