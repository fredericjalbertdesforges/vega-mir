"""Tests for vega_mir.kl."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vega_mir import (
    CYGNUS_15_ALPHABET,
    js_divergence,
    kl_divergence,
    kl_divergence_from_counts,
    kl_matrix,
)


class TestKLDivergence:
    def test_self_divergence_is_zero(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        assert kl_divergence(p, p) == pytest.approx(0.0)

    def test_uniform_self_is_zero(self) -> None:
        p = np.full(15, 1 / 15)
        assert kl_divergence(p, p) == pytest.approx(0.0)

    def test_known_analytic_bernoulli(self) -> None:
        # D(Bernoulli(0.5) || Bernoulli(0.25))
        # = 0.5 * log2(0.5/0.25) + 0.5 * log2(0.5/0.75)
        # = 0.5 + 0.5 * (1 - log2(3))
        # = 1 - 0.5 * log2(3)
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 1.0 - 0.5 * math.log2(3)
        assert kl_divergence(p, q) == pytest.approx(expected)

    def test_asymmetric(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))

    def test_inf_when_q_zero_and_p_positive(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        assert math.isinf(kl_divergence(p, q))

    def test_zero_when_p_zero(self) -> None:
        # 0 * log(0/q) := 0 by convention
        p = np.array([0.0, 1.0])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.3, 0.4])
        with pytest.raises(ValueError, match="same shape"):
            kl_divergence(p, q)

    def test_negative_input_raises(self) -> None:
        p = np.array([0.5, -0.5])
        q = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="non-negative"):
            kl_divergence(p, q)

    def test_natural_log_base(self) -> None:
        # D(Bern(0.5) || Bern(0.25)) in nats
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected_bits = 1.0 - 0.5 * math.log2(3)
        expected_nats = expected_bits * math.log(2)
        assert kl_divergence(p, q, base=math.e) == pytest.approx(expected_nats)

    def test_unnormalized_input_normalized_internally(self) -> None:
        # Counts and normalized probs give the same KL
        p_counts = np.array([5.0, 5.0])
        q_counts = np.array([2.5, 7.5])
        kl_counts = kl_divergence(p_counts, q_counts)
        kl_probs = kl_divergence(np.array([0.5, 0.5]), np.array([0.25, 0.75]))
        assert kl_counts == pytest.approx(kl_probs)

    def test_strictly_positive(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.4, 0.6])
        assert kl_divergence(p, q) > 0


class TestJSDivergence:
    def test_self_divergence_is_zero(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        assert js_divergence(p, p) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.4, 0.5])
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))

    def test_max_divergence_disjoint_supports(self) -> None:
        # JS(δ_a, δ_b) = 1 in bits (maximum for log base 2)
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert js_divergence(p, q) == pytest.approx(1.0)

    def test_bounded_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.dirichlet(np.ones(15))
            q = rng.dirichlet(np.ones(15))
            js = js_divergence(p, q)
            assert 0 <= js <= 1

    def test_strictly_positive_for_distinct(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.4, 0.6])
        assert js_divergence(p, q) > 0


class TestKLFromCounts:
    def test_matches_smoothed_pipeline(self) -> None:
        from vega_mir.shannon import smoothed_probabilities

        p_counts = {"I": 50.0, "V": 30.0, "IV": 20.0}
        q_counts = {"I": 40.0, "V": 35.0, "IV": 25.0}
        alpha = 0.5
        alphabet = list(CYGNUS_15_ALPHABET)

        result = kl_divergence_from_counts(p_counts, q_counts, alphabet, alpha=alpha)

        p = smoothed_probabilities(p_counts, alphabet, alpha=alpha)
        q = smoothed_probabilities(q_counts, alphabet, alpha=alpha)
        expected = kl_divergence(p, q)

        assert result == pytest.approx(expected)

    def test_finite_when_smoothed(self) -> None:
        # With smoothing, KL is always finite (q has no zeros)
        p_counts = {"I": 100.0}
        q_counts = {"V": 100.0}  # Disjoint support before smoothing
        result = kl_divergence_from_counts(
            p_counts, q_counts, CYGNUS_15_ALPHABET, alpha=0.5
        )
        assert math.isfinite(result)
        assert result > 0


class TestKLMatrix:
    def test_diagonal_zero(self) -> None:
        distributions = {
            "A": np.array([0.5, 0.5]),
            "B": np.array([0.3, 0.7]),
            "C": np.array([0.1, 0.9]),
        }
        matrix = kl_matrix(distributions)
        for name in distributions:
            assert matrix[name][name] == 0.0

    def test_size(self) -> None:
        distributions = {chr(ord("A") + i): np.full(5, 0.2) for i in range(4)}
        matrix = kl_matrix(distributions)
        assert len(matrix) == 4
        for src in matrix:
            assert len(matrix[src]) == 4

    def test_asymmetry_preserved(self) -> None:
        distributions = {
            "A": np.array([0.5, 0.5]),
            "B": np.array([0.25, 0.75]),
        }
        matrix = kl_matrix(distributions)
        assert matrix["A"]["B"] != pytest.approx(matrix["B"]["A"])

    def test_values_match_pairwise_kl(self) -> None:
        rng = np.random.default_rng(42)
        distributions = {f"P{i}": rng.dirichlet(np.ones(15)) for i in range(3)}
        matrix = kl_matrix(distributions)
        for src in distributions:
            for tgt in distributions:
                if src == tgt:
                    continue
                expected = kl_divergence(distributions[src], distributions[tgt])
                assert matrix[src][tgt] == pytest.approx(expected)


class TestCygnusParity:
    """Parity with the Cygnus reference implementation (kl_analysis.kl_bits)."""

    def test_kl_matches_cygnus_kl_bits_when_smoothed(self) -> None:
        # Cygnus kl_bits uses an EPSILON hack to avoid log(0); our impl
        # is mathematically clean. They agree when q has no zeros, which
        # is always the case after Laplace smoothing on a fixed alphabet.
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(15))
        q = rng.dirichlet(np.ones(15))

        # Cygnus formula reproduced (without the EPSILON path since q > 0)
        mask = p > 0
        expected = float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
        actual = kl_divergence(p, q)

        assert actual == pytest.approx(expected, rel=1e-9)

    def test_js_matches_cygnus_js_bits(self) -> None:
        rng = np.random.default_rng(43)
        p = rng.dirichlet(np.ones(15))
        q = rng.dirichlet(np.ones(15))

        # Cygnus js_bits formula
        m = 0.5 * (p + q)
        mask_pm = p > 0
        mask_qm = q > 0
        kl_pm = float(np.sum(p[mask_pm] * np.log2(p[mask_pm] / m[mask_pm])))
        kl_qm = float(np.sum(q[mask_qm] * np.log2(q[mask_qm] / m[mask_qm])))
        expected = 0.5 * kl_pm + 0.5 * kl_qm

        actual = js_divergence(p, q)
        assert actual == pytest.approx(expected, rel=1e-9)
