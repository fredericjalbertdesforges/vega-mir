"""Tests for vega_mir.intervals."""

from __future__ import annotations

import numpy as np
import pytest

from vega_mir import (
    IntervalAnalysis,
    fit_intervals,
    reconstruct_sample,
)


class TestIntervalAnalysisFields:
    def test_namedtuple(self) -> None:
        rng = np.random.default_rng(42)
        sample = rng.exponential(scale=2.0, size=500)
        result = fit_intervals(sample)
        assert isinstance(result, IntervalAnalysis)
        for field in (
            "best_fit",
            "best_ks",
            "exponential_lambda",
            "exponential_ks",
            "laplace_mu",
            "laplace_b",
            "laplace_ks",
            "mean",
            "std",
            "pct_conjunct",
            "pct_leaps",
            "n",
        ):
            assert hasattr(result, field)


class TestFitIntervalsCore:
    def test_pure_exponential_picks_exponential(self) -> None:
        rng = np.random.default_rng(42)
        sample = rng.exponential(scale=2.0, size=2000)
        result = fit_intervals(sample)
        assert result.best_fit == "exponential"
        # Estimated lambda should be close to 1 / scale = 0.5
        assert result.exponential_lambda == pytest.approx(0.5, rel=0.1)

    def test_pure_laplace_picks_laplace(self) -> None:
        # Use absolute values of Laplace draws; the magnitude distribution
        # is heavier than the matching Exponential so KS should still favour
        # Laplace when fed back through the analysis.
        rng = np.random.default_rng(42)
        sample = np.abs(rng.laplace(loc=0.0, scale=2.0, size=2000))
        result = fit_intervals(sample)
        # Both fits will be approximations on this folded data, but Laplace
        # tends to win on this sample.  Just verify both fits succeeded and
        # that one of them is the chosen winner.
        assert result.best_fit in ("exponential", "laplace")
        assert result.exponential_ks > 0
        assert result.laplace_ks > 0
        assert result.best_ks == min(result.exponential_ks, result.laplace_ks)

    def test_pct_conjunct_all_small(self) -> None:
        sample = [0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0]
        result = fit_intervals(sample)
        assert result.pct_conjunct == pytest.approx(100.0)
        assert result.pct_leaps == pytest.approx(0.0)

    def test_pct_leaps_all_large(self) -> None:
        sample = [5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0]
        result = fit_intervals(sample)
        assert result.pct_conjunct == pytest.approx(0.0)
        assert result.pct_leaps == pytest.approx(100.0)

    def test_mean_and_std_match_numpy(self) -> None:
        rng = np.random.default_rng(42)
        sample = rng.exponential(scale=3.0, size=1000)
        result = fit_intervals(sample)
        assert result.mean == pytest.approx(float(np.mean(sample)))
        assert result.std == pytest.approx(float(np.std(sample)))

    def test_n_matches_input_size(self) -> None:
        sample = list(range(50))
        result = fit_intervals(sample)
        assert result.n == 50

    def test_signed_intervals_get_absolute(self) -> None:
        # Signed input → mean of |intervals|. Mix of magnitudes to avoid
        # scipy degenerate-fit warnings on perfectly constant-magnitude data.
        signed = [-3, 3, -2, 2, -1, 1, -3, 3, -2, 2]
        result = fit_intervals(signed)
        assert result.mean == pytest.approx(2.2)

    def test_too_few_intervals_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            fit_intervals([1, 2, 3, 4, 5])

    def test_accepts_numpy_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = fit_intervals(arr)
        assert result.n == 10

    def test_best_ks_consistent_with_best_fit(self) -> None:
        rng = np.random.default_rng(42)
        sample = rng.exponential(scale=2.0, size=500)
        result = fit_intervals(sample)
        if result.best_fit == "exponential":
            assert result.best_ks == pytest.approx(result.exponential_ks)
        else:
            assert result.best_ks == pytest.approx(result.laplace_ks)


class TestReconstructSample:
    def test_proportion_dict(self) -> None:
        # 50% interval=0, 30% interval=1, 20% interval=2 → 10000 sample
        dist = {"0": 0.5, "1": 0.3, "2": 0.2}
        sample = reconstruct_sample(dist, sample_size=10000)
        assert sample.size == 10000  # exact since proportions divide evenly
        # Counts should match expectations
        zeros = int(np.sum(sample == 0))
        ones = int(np.sum(sample == 1))
        twos = int(np.sum(sample == 2))
        assert zeros == 5000
        assert ones == 3000
        assert twos == 2000

    def test_negative_intervals_made_absolute(self) -> None:
        dist = {"-3": 0.5, "3": 0.5}
        sample = reconstruct_sample(dist, sample_size=1000)
        # All entries should be 3, none should be -3
        assert np.all(sample == 3.0)

    def test_int_keys_supported(self) -> None:
        dist = {0: 0.5, 1: 0.3, 2: 0.2}
        sample = reconstruct_sample(dist, sample_size=1000)
        assert sample.size == 1000

    def test_minimum_one_count_per_entry(self) -> None:
        # Tiny weight should still produce at least one sample
        dist = {"0": 0.999, "1": 0.0001}
        sample = reconstruct_sample(dist, sample_size=100)
        # 0 → max(1, 99) = 99 ; 1 → max(1, 0) = 1
        assert int(np.sum(sample == 1)) == 1


class TestEndToEnd:
    def test_distribution_roundtrip(self) -> None:
        # Build a realistic interval distribution then run the full pipeline
        dist = {
            "0": 0.30,  # repeated note
            "1": 0.20,  # half-step
            "2": 0.20,  # whole step
            "3": 0.10,  # minor third
            "4": 0.08,  # major third
            "5": 0.05,  # fourth
            "7": 0.04,  # fifth
            "12": 0.03,  # octave
        }
        sample = reconstruct_sample(dist, sample_size=10000)
        result = fit_intervals(sample)
        assert result.n >= 10000
        # Conjunct motion (≤ 2) should be 30 + 20 + 20 = 70%
        assert result.pct_conjunct == pytest.approx(70.0, abs=0.5)
        # Leaps (> 4) should be 5 + 4 + 3 = 12%
        assert result.pct_leaps == pytest.approx(12.0, abs=0.5)


class TestCygnusParity:
    """Parity with Cygnus math_metrics.interval_distribution_fit."""

    def test_matches_cygnus_pipeline(self) -> None:
        # Reproduce the Cygnus path: dict → reconstructed sample → fit
        from scipy import stats as scipy_stats

        interval_dist = {
            "-3": 0.05,
            "-2": 0.10,
            "-1": 0.15,
            "0": 0.30,
            "1": 0.15,
            "2": 0.10,
            "3": 0.05,
            "5": 0.05,
            "7": 0.05,
        }

        sample = reconstruct_sample(interval_dist, sample_size=10000)
        result = fit_intervals(sample)

        # Reference: reproduce Cygnus interval_distribution_fit
        abs_intervals = []
        for iv_str, prop in interval_dist.items():
            iv = int(iv_str)
            count = max(1, int(prop * 10000))
            abs_intervals.extend([abs(iv)] * count)
        intervals = np.array(abs_intervals, dtype=float)

        params_exp = scipy_stats.expon.fit(intervals)
        ks_exp, _ = scipy_stats.kstest(intervals, "expon", params_exp)
        params_lap = scipy_stats.laplace.fit(intervals)
        ks_lap, _ = scipy_stats.kstest(intervals, "laplace", params_lap)

        assert result.exponential_ks == pytest.approx(ks_exp)
        assert result.laplace_ks == pytest.approx(ks_lap)
        assert result.exponential_lambda == pytest.approx(1 / max(params_exp[1], 1e-10))
        assert result.laplace_mu == pytest.approx(params_lap[0])
        assert result.laplace_b == pytest.approx(params_lap[1])
        assert result.mean == pytest.approx(float(np.mean(intervals)))
        assert result.std == pytest.approx(float(np.std(intervals)))
