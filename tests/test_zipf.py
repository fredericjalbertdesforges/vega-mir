"""Tests for vega_mir.zipf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from vega_mir import (
    CYGNUS_15_ALPHABET,
    ZipfFit,
    zipf_fit,
    zipf_fit_marginal,
    zipf_fit_transitions,
)


class TestZipfFitCore:
    def test_perfect_zipf_alpha_one(self) -> None:
        # Construct P_i ∝ 1/i  →  log P = -log i + C  →  alpha = 1, R² = 1
        n = 20
        ranks = np.arange(1, n + 1, dtype=np.float64)
        probs = 1.0 / ranks
        probs /= probs.sum()
        result = zipf_fit(probs)
        assert result.alpha == pytest.approx(1.0, abs=0.001)
        assert result.r_squared == pytest.approx(1.0, abs=0.001)
        assert result.n_points == n

    def test_perfect_zipf_alpha_two(self) -> None:
        # Construct P_i ∝ 1/i^2  →  alpha = 2, R² = 1
        n = 20
        ranks = np.arange(1, n + 1, dtype=np.float64)
        probs = 1.0 / ranks**2
        probs /= probs.sum()
        result = zipf_fit(probs)
        assert result.alpha == pytest.approx(2.0, abs=0.001)
        assert result.r_squared == pytest.approx(1.0, abs=0.001)

    def test_uniform_distribution_alpha_zero(self) -> None:
        # Uniform → no rank dependence → alpha = 0, R² = 0 (no slope)
        probs = np.full(15, 1 / 15)
        result = zipf_fit(probs)
        assert result.alpha == pytest.approx(0.0, abs=0.001)

    def test_zeros_dropped(self) -> None:
        probs = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
        result = zipf_fit(probs)
        assert result.n_points == 3

    def test_too_few_points_returns_zero_fit(self) -> None:
        probs = np.array([0.7, 0.3])  # only 2 non-zero entries
        result = zipf_fit(probs)
        assert result.alpha == 0.0
        assert result.r_squared == 0.0
        assert result.intercept == 0.0
        assert result.n_points == 2

    def test_returns_zipffit_namedtuple(self) -> None:
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        result = zipf_fit(probs)
        assert isinstance(result, ZipfFit)
        # Tuple unpacking works
        a, r2, ic, n = result
        assert a == result.alpha

    def test_alpha_positive_for_decreasing_distribution(self) -> None:
        probs = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
        result = zipf_fit(probs)
        assert result.alpha > 0

    def test_unnormalized_input_accepted(self) -> None:
        # Counts and normalized probs give the same fit
        counts = np.array([100.0, 50.0, 25.0, 10.0])
        probs = counts / counts.sum()
        r_counts = zipf_fit(counts)
        r_probs = zipf_fit(probs)
        assert r_counts.alpha == pytest.approx(r_probs.alpha)
        assert r_counts.r_squared == pytest.approx(r_probs.r_squared)


class TestZipfFitMarginal:
    def test_uniform_sequence_yields_low_alpha(self) -> None:
        seq = list(CYGNUS_15_ALPHABET) * 100
        result = zipf_fit_marginal(seq, collapse=False)
        assert abs(result.alpha) < 0.1

    def test_realistic_sequence(self) -> None:
        seq = (
            ["I"] * 30 + ["V"] * 25 + ["IV"] * 20
            + ["i"] * 10 + ["v"] * 8 + ["bVII"] * 5 + ["VII"] * 2
        ) * 50
        result = zipf_fit_marginal(seq)
        assert result.alpha > 0
        assert result.r_squared > 0.5
        # All 15 symbols present after smoothing
        assert result.n_points == 15

    def test_n_points_equals_alphabet_size_after_smoothing(self) -> None:
        seq = ["I"] * 100  # only one symbol observed
        result = zipf_fit_marginal(seq)
        assert result.n_points == 15  # smoothing gives all 15 a non-zero prob

    def test_default_alphabet_is_cygnus_15(self) -> None:
        seq = ["I", "V"] * 50
        result_default = zipf_fit_marginal(seq)
        result_explicit = zipf_fit_marginal(seq, alphabet=CYGNUS_15_ALPHABET)
        assert result_default == result_explicit


class TestZipfFitTransitions:
    def test_returns_zipffit(self) -> None:
        seq = ["I", "V", "I", "IV", "V", "I"] * 50
        result = zipf_fit_transitions(seq)
        assert isinstance(result, ZipfFit)

    def test_n_points_equals_alphabet_squared_after_smoothing(self) -> None:
        seq = ["I", "V", "I", "IV", "V", "I"] * 50
        result = zipf_fit_transitions(seq)
        assert result.n_points == 15 * 15

    def test_realistic_transitions(self) -> None:
        seq = ["I", "V", "I", "IV", "V", "I", "VI", "II", "V", "I"] * 100
        result = zipf_fit_transitions(seq)
        assert result.alpha > 0
        # Cygnus paper reports R² ~ 0.46 (neoclassical) to ~ 0.78 (historical)
        # for real composers on the 225-point joint distribution; synthetic
        # short cycles produce R² in the lower neoclassical range.
        assert result.r_squared > 0.3

    def test_collapse_changes_transitions(self) -> None:
        # Without collapse, the bigram (I, I) appears repeatedly; with
        # collapse it is removed and the joint distribution shifts.
        seq = ["I", "I", "I", "V", "V", "V"] * 100
        r_collapsed = zipf_fit_transitions(seq, collapse=True)
        r_raw = zipf_fit_transitions(seq, collapse=False)
        assert r_collapsed.alpha != r_raw.alpha

    def test_unknown_symbols_dropped(self) -> None:
        seq_clean = ["I", "V", "I", "V"] * 100
        seq_noisy = ["I", "X", "V", "Y", "I", "Z", "V"] * 100
        # Unknown symbols are dropped from joint counting (they break the
        # surrounding bigrams). Both fits should still produce valid output.
        r_clean = zipf_fit_transitions(seq_clean)
        r_noisy = zipf_fit_transitions(seq_noisy)
        assert r_clean.n_points == 15 * 15
        assert r_noisy.n_points == 15 * 15


class TestCygnusParity:
    """Parity with the Cygnus reference implementation (shannon_zipf_v2.zipf_fit)."""

    def test_zipf_fit_matches_cygnus_formula(self) -> None:
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(15))
        result = zipf_fit(probs)

        # Reference: reproduce Cygnus shannon_zipf_v2.zipf_fit step by step.
        order = np.argsort(-probs)
        sorted_p = probs[order]
        ranks = np.arange(1, len(sorted_p) + 1)
        mask = sorted_p > 0
        log_r = np.log2(ranks[mask])
        log_p = np.log2(sorted_p[mask])
        slope, intercept, r_value, _, _ = stats.linregress(log_r, log_p)

        assert result.alpha == pytest.approx(-slope)
        assert result.r_squared == pytest.approx(r_value ** 2)
        assert result.intercept == pytest.approx(intercept)
        assert result.n_points == int(mask.sum())

    def test_marginal_matches_cygnus_pipeline(self) -> None:
        # End-to-end parity: counts → smoothed → zipf_fit
        seq = (["I"] * 50 + ["V"] * 30 + ["IV"] * 20 + ["i"] * 10) * 20
        result = zipf_fit_marginal(seq, collapse=False)

        # Reference computation
        from collections import Counter
        counts = Counter(seq)
        alphabet = list(CYGNUS_15_ALPHABET)
        vec = np.array([counts.get(k, 0) for k in alphabet], dtype=float)
        vec = vec + 0.5
        vec /= vec.sum()
        order = np.argsort(-vec)
        sorted_p = vec[order]
        ranks = np.arange(1, len(sorted_p) + 1)
        log_r = np.log2(ranks)
        log_p = np.log2(sorted_p)
        slope, intercept, r_value, _, _ = stats.linregress(log_r, log_p)

        assert result.alpha == pytest.approx(-slope)
        assert result.r_squared == pytest.approx(r_value ** 2)
        assert result.intercept == pytest.approx(intercept)
