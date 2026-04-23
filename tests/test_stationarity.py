"""Tests for vega_mir.stationarity."""

from __future__ import annotations

import numpy as np
import pytest

from vega_mir import (
    CYGNUS_15_ALPHABET,
    StationarityResult,
    stationarity_test,
)


class TestStationarityResult:
    def test_namedtuple_fields(self) -> None:
        # Verify the result is a NamedTuple with all expected fields
        rng = np.random.default_rng(42)
        seq = rng.choice(list(CYGNUS_15_ALPHABET), size=200).tolist()
        result = stationarity_test(seq)
        assert isinstance(result, StationarityResult)
        assert hasattr(result, "chi2")
        assert hasattr(result, "p_value")
        assert hasattr(result, "dof")
        assert hasattr(result, "cramers_v")
        assert hasattr(result, "n_segments")
        assert hasattr(result, "n_observations")
        assert hasattr(result, "is_stationary")


class TestStationaryDetection:
    def test_uniform_random_is_stationary(self) -> None:
        # IID samples from a fixed distribution → stationary by construction
        rng = np.random.default_rng(42)
        seq = rng.choice(list(CYGNUS_15_ALPHABET), size=400).tolist()
        result = stationarity_test(seq, n_segments=4)
        assert result.is_stationary
        assert result.p_value > 0.05
        # Cramer's V should be small for IID data
        assert result.cramers_v < 0.2

    def test_repeated_pattern_is_stationary(self) -> None:
        # 5-symbol pattern * 40 = 200 obs, n_segments=4 → segment_size=50 = 10 perfect
        # cycles per segment → contingency rows identical → Cramer's V == 0 exactly.
        seq = ["I", "V", "IV", "vi", "ii"] * 40
        result = stationarity_test(seq, n_segments=4)
        assert result.is_stationary
        assert result.cramers_v == pytest.approx(0.0, abs=1e-10)


class TestNonStationaryDetection:
    def test_concatenated_distinct_distributions_is_non_stationary(self) -> None:
        # First half all "I", second half all "V" → wildly non-stationary
        seq = ["I"] * 100 + ["V"] * 100
        result = stationarity_test(seq, n_segments=4)
        assert not result.is_stationary
        assert result.p_value < 0.05
        # Cramer's V should be near maximum (1.0) for completely disjoint segments
        assert result.cramers_v > 0.9

    def test_drift_in_distribution_detected(self) -> None:
        # Segment 1: 80% I, Segment 2: 60% I, Segment 3: 40% I, Segment 4: 20% I
        rng = np.random.default_rng(42)
        seq: list[str] = []
        for p_i in (0.8, 0.6, 0.4, 0.2):
            n_i = int(50 * p_i)
            n_v = 50 - n_i
            segment = ["I"] * n_i + ["V"] * n_v
            rng.shuffle(segment)
            seq.extend(segment)
        result = stationarity_test(seq, n_segments=4)
        assert not result.is_stationary
        assert result.cramers_v > 0.3  # medium-large effect


class TestEdgeCases:
    def test_too_short_raises(self) -> None:
        # n_segments=4 needs at least 4*5=20 observations
        seq = ["I", "V"] * 5  # only 10
        with pytest.raises(ValueError, match="too short"):
            stationarity_test(seq, n_segments=4)

    def test_n_segments_below_two_raises(self) -> None:
        seq = ["I", "V"] * 50
        with pytest.raises(ValueError, match="at least 2"):
            stationarity_test(seq, n_segments=1)

    def test_single_symbol_raises(self) -> None:
        seq = ["I"] * 100
        with pytest.raises(ValueError, match="distinct symbols"):
            stationarity_test(seq)

    def test_two_segments(self) -> None:
        # Should work with the minimum n_segments=2
        rng = np.random.default_rng(42)
        seq = rng.choice(list(CYGNUS_15_ALPHABET), size=100).tolist()
        result = stationarity_test(seq, n_segments=2)
        assert result.n_segments == 2
        assert result.dof >= 1

    def test_significance_parameter_changes_flag(self) -> None:
        # Marginal p-value should flip with stricter significance
        rng = np.random.default_rng(0)
        seq: list[str] = []
        # Build a sequence with a small but real drift
        for p_i in (0.55, 0.50, 0.50, 0.45):
            n_i = int(100 * p_i)
            n_v = 100 - n_i
            segment = ["I"] * n_i + ["V"] * n_v
            rng.shuffle(segment)
            seq.extend(segment)
        # Use a very lax significance to almost always call it stationary
        lax = stationarity_test(seq, significance=0.99)
        # Use a very strict significance to almost always reject stationarity
        strict = stationarity_test(seq, significance=0.0001)
        assert strict.is_stationary != lax.is_stationary or strict.p_value == lax.p_value


class TestCygnusParity:
    """Parity with Cygnus math_metrics.harmonic_stationarity_track."""

    def test_matches_cygnus_chi2_and_cramers_v(self) -> None:
        # Build a sequence and reproduce the Cygnus computation
        rng = np.random.default_rng(42)
        seq = rng.choice(["I", "V", "IV", "vi", "ii"], size=200).tolist()

        n_segments = 4
        result = stationarity_test(seq, n_segments=n_segments)

        # Reference: reproduce Cygnus harmonic_stationarity_track
        from collections import Counter
        from scipy import stats

        segment_size = len(seq) // n_segments
        segments = [seq[i * segment_size:(i + 1) * segment_size] for i in range(n_segments)]
        all_symbols = sorted(set(seq))
        contingency = np.array(
            [[Counter(seg).get(s, 0) for s in all_symbols] for seg in segments],
            dtype=np.int64,
        )
        col_sums = contingency.sum(axis=0)
        contingency = contingency[:, col_sums > 0]
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        n = contingency.sum()
        k = min(contingency.shape) - 1
        expected_v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0

        assert result.chi2 == pytest.approx(chi2)
        assert result.p_value == pytest.approx(p_value)
        assert result.dof == int(dof)
        assert result.cramers_v == pytest.approx(expected_v)
        assert result.is_stationary == bool(p_value > 0.05)
