"""Tests for vega_mir.gini."""

from __future__ import annotations

import numpy as np
import pytest

from vega_mir import gini, gini_from_counts, gini_multi


class TestGiniCore:
    def test_uniform_zero(self) -> None:
        # All equal values → no inequality → Gini = 0
        assert gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_uniform_zero_large(self) -> None:
        assert gini(np.full(15, 1 / 15)) == pytest.approx(0.0)

    def test_max_inequality(self) -> None:
        # All mass on one entry → Gini = (n-1)/n
        for n in (2, 4, 10, 100):
            arr = np.zeros(n)
            arr[0] = 1.0
            assert gini(arr) == pytest.approx((n - 1) / n)

    def test_known_analytic_value(self) -> None:
        # values = [1, 2, 3, 4]
        # G = (2*30 - 5*10) / (4*10) = 10/40 = 0.25
        assert gini([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)

    def test_invariant_to_order(self) -> None:
        # Gini is sort-invariant by construction
        v1 = [1.0, 2.0, 3.0, 4.0]
        v2 = [4.0, 1.0, 3.0, 2.0]
        assert gini(v1) == pytest.approx(gini(v2))

    def test_invariant_to_scale(self) -> None:
        # Gini is scale-invariant: gini(c * x) = gini(x) for c > 0
        v = [1.0, 2.0, 3.0, 4.0]
        v_scaled = [10.0, 20.0, 30.0, 40.0]
        assert gini(v) == pytest.approx(gini(v_scaled))

    def test_all_zeros_returns_zero(self) -> None:
        assert gini([0.0, 0.0, 0.0]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert gini([]) == 0.0

    def test_single_value_returns_zero(self) -> None:
        # n=1, max possible Gini = (1-1)/1 = 0
        assert gini([5.0]) == 0.0

    def test_two_values(self) -> None:
        # values = [1, 3], sorted = [1, 3], sum = 4, n = 2
        # G = (2*(1*1 + 2*3) - 3*4) / (2*4) = (14 - 12) / 8 = 0.25
        assert gini([1.0, 3.0]) == pytest.approx(0.25)

    def test_negative_input_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            gini([1.0, -1.0, 2.0])

    def test_accepts_numpy_array(self) -> None:
        assert gini(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.25)

    def test_accepts_python_list(self) -> None:
        assert gini([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)

    def test_complements_entropy_at_extremes(self) -> None:
        # Deterministic distribution: gini = (n-1)/n, entropy = 0
        # Uniform distribution: gini = 0, entropy = log2(n)
        from vega_mir import shannon_entropy
        n = 10
        uniform = np.full(n, 1 / n)
        deterministic = np.zeros(n)
        deterministic[0] = 1.0

        assert gini(uniform) == pytest.approx(0.0)
        assert shannon_entropy(uniform) == pytest.approx(np.log2(n))
        assert gini(deterministic) == pytest.approx((n - 1) / n)
        assert shannon_entropy(deterministic) == pytest.approx(0.0)


class TestGiniFromCounts:
    def test_extracts_values(self) -> None:
        counts = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
        assert gini_from_counts(counts) == pytest.approx(0.25)

    def test_keys_ignored(self) -> None:
        counts1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        counts2 = {"x": 1.0, "y": 2.0, "z": 3.0}
        assert gini_from_counts(counts1) == pytest.approx(gini_from_counts(counts2))

    def test_empty_dict(self) -> None:
        assert gini_from_counts({}) == 0.0


class TestGiniMulti:
    def test_returns_dict(self) -> None:
        distributions = {
            "harmonic": [1.0, 2.0, 3.0, 4.0],
            "dynamic": [1.0, 1.0, 1.0, 1.0],
            "register": [1.0, 0.0, 0.0, 0.0],
        }
        result = gini_multi(distributions)
        assert set(result.keys()) == {"harmonic", "dynamic", "register"}

    def test_each_value_correct(self) -> None:
        distributions = {
            "uniform": [1.0, 1.0, 1.0, 1.0],
            "skewed": [1.0, 0.0, 0.0, 0.0],
            "linear": [1.0, 2.0, 3.0, 4.0],
        }
        result = gini_multi(distributions)
        assert result["uniform"] == pytest.approx(0.0)
        assert result["skewed"] == pytest.approx(0.75)
        assert result["linear"] == pytest.approx(0.25)

    def test_empty_dict(self) -> None:
        assert gini_multi({}) == {}


class TestCygnusParity:
    """Parity with Cygnus math_metrics._gini."""

    def test_matches_cygnus_formula(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(2, 50)
            values = rng.exponential(scale=1.0, size=int(n))

            # Cygnus formula reproduced exactly
            sorted_vals = np.sort(np.array(values, dtype=float))
            ns = len(sorted_vals)
            total = np.sum(sorted_vals)
            index = np.arange(1, ns + 1)
            expected = float(
                (2 * np.sum(index * sorted_vals) - (ns + 1) * total) / (ns * total)
            )

            actual = gini(values)
            assert actual == pytest.approx(expected, rel=1e-12)
