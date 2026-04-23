"""Tests for vega_mir.shannon."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vega_mir import (
    CYGNUS_15_ALPHABET,
    shannon_entropy,
    shannon_scale_degrees,
)
from vega_mir.shannon import (
    collapse_repetitions,
    smoothed_probabilities,
)


class TestCygnus15Alphabet:
    def test_size(self) -> None:
        assert len(CYGNUS_15_ALPHABET) == 15

    def test_distinct(self) -> None:
        assert len(set(CYGNUS_15_ALPHABET)) == 15

    def test_contains_diatonic(self) -> None:
        for d in ("I", "II", "III", "IV", "V", "VI", "VII"):
            assert d in CYGNUS_15_ALPHABET

    def test_contains_chromatic(self) -> None:
        for d in ("bII", "bIII", "#IV", "bVI", "bVII"):
            assert d in CYGNUS_15_ALPHABET

    def test_mode_pairs(self) -> None:
        for upper, lower in (("I", "i"), ("IV", "iv"), ("V", "v")):
            assert upper in CYGNUS_15_ALPHABET
            assert lower in CYGNUS_15_ALPHABET


class TestShannonEntropy:
    def test_uniform_two(self) -> None:
        assert shannon_entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)

    def test_uniform_fifteen(self) -> None:
        probs = np.full(15, 1 / 15)
        assert shannon_entropy(probs) == pytest.approx(math.log2(15))

    def test_certain(self) -> None:
        assert shannon_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_zeros_filtered(self) -> None:
        assert shannon_entropy(np.array([0.5, 0.5, 0.0])) == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        assert shannon_entropy(np.array([])) == 0.0

    def test_natural_log_base(self) -> None:
        h = shannon_entropy(np.array([0.5, 0.5]), base=math.e)
        assert h == pytest.approx(math.log(2))

    def test_log10_base(self) -> None:
        h = shannon_entropy(np.array([0.5, 0.5]), base=10.0)
        assert h == pytest.approx(math.log10(2))

    def test_accepts_python_list(self) -> None:
        assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)

    def test_unnormalized_input(self) -> None:
        h_raw = shannon_entropy(np.array([3.0, 7.0]))
        h_normalized = shannon_entropy(np.array([0.3, 0.7]))
        assert h_raw == pytest.approx(h_normalized)

    def test_monotonic_with_uniformity(self) -> None:
        h_skewed = shannon_entropy(np.array([0.9, 0.1]))
        h_uniform = shannon_entropy(np.array([0.5, 0.5]))
        assert h_skewed < h_uniform


class TestSmoothedProbabilities:
    def test_normalizes(self) -> None:
        counts = {"a": 10.0, "b": 5.0}
        probs = smoothed_probabilities(counts, ["a", "b"], alpha=0.0)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] == pytest.approx(10 / 15)
        assert probs[1] == pytest.approx(5 / 15)

    def test_jeffreys_smoothing_uniform_when_empty(self) -> None:
        counts: dict[str, float] = {"a": 0.0, "b": 0.0}
        probs = smoothed_probabilities(counts, ["a", "b"], alpha=0.5)
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)

    def test_alphabet_padding(self) -> None:
        counts = {"I": 100.0}
        probs = smoothed_probabilities(counts, CYGNUS_15_ALPHABET, alpha=0.5)
        assert probs.shape == (15,)
        assert probs.sum() == pytest.approx(1.0)
        i_idx = CYGNUS_15_ALPHABET.index("I")
        v_idx = CYGNUS_15_ALPHABET.index("V")
        assert probs[i_idx] > probs[v_idx]
        assert probs[v_idx] > 0

    def test_empty_alphabet(self) -> None:
        probs = smoothed_probabilities({}, [], alpha=0.5)
        assert probs.shape == (0,)

    def test_unknown_symbols_ignored(self) -> None:
        counts = {"I": 50.0, "X": 999.0}
        probs = smoothed_probabilities(counts, ["I", "V"], alpha=0.0)
        assert probs[0] == pytest.approx(1.0)
        assert probs[1] == pytest.approx(0.0)


class TestCollapseRepetitions:
    def test_basic(self) -> None:
        assert collapse_repetitions(["I", "I", "V", "V", "V", "I"]) == ["I", "V", "I"]

    def test_no_repetitions(self) -> None:
        assert collapse_repetitions(["I", "V", "vi", "IV"]) == ["I", "V", "vi", "IV"]

    def test_all_same(self) -> None:
        assert collapse_repetitions(["I"] * 10) == ["I"]

    def test_empty(self) -> None:
        assert collapse_repetitions([]) == []

    def test_single(self) -> None:
        assert collapse_repetitions(["I"]) == ["I"]

    def test_alternating_preserved(self) -> None:
        assert collapse_repetitions(["I", "V", "I", "V"]) == ["I", "V", "I", "V"]


class TestShannonScaleDegrees:
    def test_uniform_alphabet_no_smoothing(self) -> None:
        seq = list(CYGNUS_15_ALPHABET) * 50
        h = shannon_scale_degrees(seq, alpha=0.0, collapse=False)
        assert h == pytest.approx(math.log2(15))

    def test_collapse_changes_distribution(self) -> None:
        seq = ["I"] * 100 + ["V"] * 1
        h_collapsed = shannon_scale_degrees(seq, collapse=True, alpha=0.0)
        h_raw = shannon_scale_degrees(seq, collapse=False, alpha=0.0)
        assert h_raw < h_collapsed
        assert h_collapsed == pytest.approx(1.0)

    def test_default_smoothing_pulls_constant_above_zero(self) -> None:
        seq = ["I"] * 100
        h = shannon_scale_degrees(seq)
        assert h > 0
        assert h < math.log2(15)

    def test_realistic_range(self) -> None:
        # Tonic-heavy distribution mimicking a real composer
        seq = (
            ["I"] * 30 + ["V"] * 25 + ["IV"] * 20
            + ["i"] * 10 + ["v"] * 8 + ["bVII"] * 5 + ["VII"] * 2
        )
        seq = seq * 50
        h = shannon_scale_degrees(seq)
        # Cygnus arXiv reports range 3.33-3.86 bits for real composers.
        # Synthetic distribution should land in a similar order of magnitude.
        assert 2.0 < h < math.log2(15)

    def test_unknown_symbols_silently_dropped(self) -> None:
        seq_with_noise = ["I", "V", "X", "Y", "I", "V"]
        seq_clean = ["I", "V", "I", "V"]
        h_noise = shannon_scale_degrees(seq_with_noise, collapse=False, alpha=0.0)
        h_clean = shannon_scale_degrees(seq_clean, collapse=False, alpha=0.0)
        assert h_noise == pytest.approx(h_clean)

    def test_default_log_base_is_bits(self) -> None:
        seq = ["I", "V"] * 50
        h = shannon_scale_degrees(seq, alphabet=("I", "V"), alpha=0.0)
        assert h == pytest.approx(1.0)


class TestCygnusParity:
    """Parity with the Cygnus reference implementation (kl_analysis.smoothed_vector)."""

    def test_smoothed_probabilities_matches_cygnus_formula(self) -> None:
        counts = {"I": 10.0, "V": 5.0, "iv": 3.0}
        alphabet = list(CYGNUS_15_ALPHABET)
        alpha = 0.5
        probs = smoothed_probabilities(counts, alphabet, alpha=alpha)

        # Reproduce kl_analysis.smoothed_vector exactly.
        ref = np.array([counts.get(k, 0) for k in alphabet], dtype=float)
        ref = ref + alpha
        ref /= ref.sum()

        np.testing.assert_allclose(probs, ref)

    def test_default_alpha_is_jeffreys(self) -> None:
        counts: dict[str, float] = {"a": 0.0, "b": 0.0}
        probs = smoothed_probabilities(counts, ["a", "b"])
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)
