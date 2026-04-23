"""Tests for vega_mir.fractal."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from vega_mir import (
    FractalDimension,
    higuchi_fractal_dimension,
)


class TestFractalDimensionFields:
    def test_namedtuple(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(size=1000)
        result = higuchi_fractal_dimension(series)
        assert isinstance(result, FractalDimension)
        for field in ("dimension", "r_squared", "n_points", "k_max"):
            assert hasattr(result, field)


class TestKnownSignals:
    def test_linear_ramp_close_to_one(self) -> None:
        # A linear ramp is maximally smooth → D close to 1
        series = np.arange(500, dtype=np.float64)
        result = higuchi_fractal_dimension(series)
        assert 0.9 < result.dimension < 1.2

    def test_white_noise_close_to_two(self) -> None:
        # White Gaussian noise → D close to 2
        rng = np.random.default_rng(42)
        series = rng.normal(size=2000)
        result = higuchi_fractal_dimension(series)
        assert 1.7 < result.dimension < 2.05

    def test_brownian_motion_close_to_one_point_five(self) -> None:
        # Brownian motion (random walk) → D close to 1.5
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(size=2000))
        result = higuchi_fractal_dimension(series)
        assert 1.3 < result.dimension < 1.7

    def test_sine_wave_close_to_one(self) -> None:
        # Smooth sine wave → D close to 1
        x = np.linspace(0, 4 * np.pi, 500)
        series = np.sin(x)
        result = higuchi_fractal_dimension(series)
        assert 0.9 < result.dimension < 1.3

    def test_high_r_squared_on_clean_signals(self) -> None:
        # Linear and noise both have clean power-law scaling
        rng = np.random.default_rng(42)
        ramp = np.arange(500, dtype=np.float64)
        noise = rng.normal(size=2000)
        assert higuchi_fractal_dimension(ramp).r_squared > 0.8
        assert higuchi_fractal_dimension(noise).r_squared > 0.8


class TestDegenerateCases:
    def test_constant_series_returns_degenerate(self) -> None:
        series = np.ones(100)
        result = higuchi_fractal_dimension(series)
        # All differences are 0 → all L(k) = 0 → no valid points
        assert result.dimension == 0.0
        assert result.r_squared == 0.0
        assert result.n_points == 0

    def test_too_short_raises(self) -> None:
        series = np.zeros(10)
        with pytest.raises(ValueError, match="at least"):
            higuchi_fractal_dimension(series, k_max=10)

    def test_kmax_below_two_raises(self) -> None:
        series = np.arange(100, dtype=np.float64)
        with pytest.raises(ValueError, match="k_max must be at least 2"):
            higuchi_fractal_dimension(series, k_max=1)


class TestKmaxParameter:
    def test_kmax_changes_n_points(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(size=1000)
        r5 = higuchi_fractal_dimension(series, k_max=5)
        r15 = higuchi_fractal_dimension(series, k_max=15)
        assert r5.n_points <= 5
        assert r15.n_points <= 15
        assert r15.n_points > r5.n_points

    def test_kmax_default_is_ten(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(size=200)
        r_default = higuchi_fractal_dimension(series)
        r_explicit = higuchi_fractal_dimension(series, k_max=10)
        assert r_default == r_explicit

    def test_dimension_stable_across_kmax(self) -> None:
        # Estimated dimension should be roughly stable across k_max choices
        rng = np.random.default_rng(42)
        series = rng.normal(size=2000)
        d_5 = higuchi_fractal_dimension(series, k_max=5).dimension
        d_10 = higuchi_fractal_dimension(series, k_max=10).dimension
        d_20 = higuchi_fractal_dimension(series, k_max=20).dimension
        # Expect all in the noise regime (close to 2)
        for d in (d_5, d_10, d_20):
            assert 1.5 < d < 2.1


class TestCygnusParity:
    """Parity with Cygnus math_metrics.higuchi_fractal_dimension."""

    def test_matches_cygnus_formula(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(size=500).astype(np.float64)
        k_max = 10
        result = higuchi_fractal_dimension(series, k_max=k_max)

        # Reference: reproduce Cygnus higuchi_fractal_dimension verbatim
        x = np.array(series, dtype=float)
        n = len(x)
        l_values: list[float] = []
        k_values: list[int] = []
        for k in range(1, k_max + 1):
            l_k = 0.0
            count = 0
            for m in range(1, k + 1):
                indices = np.arange(m - 1, n, k)
                if len(indices) < 2:
                    continue
                l_mk = float(np.sum(np.abs(np.diff(x[indices]))))
                n_seg = len(indices) - 1
                if n_seg > 0:
                    l_mk = l_mk * (n - 1) / (k * n_seg * k)
                    l_k += l_mk
                    count += 1
            if count > 0:
                l_k /= count
                if l_k > 0:
                    l_values.append(l_k)
                    k_values.append(k)
        log_k = np.log(np.array(k_values))
        log_l = np.log(np.array(l_values))
        slope, _, r_value, _, _ = stats.linregress(log_k, log_l)

        assert result.dimension == pytest.approx(-slope)
        assert result.r_squared == pytest.approx(r_value ** 2)
        assert result.n_points == len(l_values)
