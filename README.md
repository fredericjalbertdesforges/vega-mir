# vega-mir

Information-theoretic analysis of symbolic music (MIDI).

[![CI](https://github.com/fredericjalbertdesforges/vega-mir/actions/workflows/ci.yml/badge.svg)](https://github.com/fredericjalbertdesforges/vega-mir/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/vega-mir.svg)](https://pypi.org/project/vega-mir/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19711033.svg)](https://doi.org/10.5281/zenodo.19711033)

A focused, well-tested Python library that bundles **nine information-theoretic metrics** for the analysis of symbolic music corpora.

## Why

The MIR community uses `music21`, `partitura`, and `jSymbolic` for broad symbolic music analysis, but no Python library currently offers a focused, reproducible, well-tested information-theoretic toolkit. Researchers stitch together `scipy.stats.entropy`, `networkx`, and hand-rolled key detection by hand. `vega-mir` is the missing piece.

## The 9 metrics

1. **Shannon entropy** on scale degrees (alphabet `|D| = 15`)
2. **Zipf's law** fits on marginal and transition distributions
3. **Network analysis** on chord progression graphs
4. **Fractal dimension** via box-counting on note density curves
5. **Interval distribution** statistics
6. **Multi-dimensional Gini** coefficients (harmonic / dynamic / rhythmic)
7. **Harmonic stationarity** tests on time-series harmonic profiles
8. **Asymmetric Kullback-Leibler divergence** between composer corpora (with Laplace smoothing and bootstrap CIs)
9. **Rubato spectral analysis** via FFT on tempo curves

## Status

**Pre-alpha (v0.0.1).** API is unstable. Under active development.

## Installation

```bash
pip install vega-mir
```

## Quick start

`vega-mir` operates on **symbolic** input — sequences, distributions, time series, or graphs — not raw audio. Use any upstream tool (music21, partitura, your own pipeline) to extract scale-degree sequences, then feed them in:

```python
from vega_mir import shannon_scale_degrees

# A scale-degree sequence on the 15-symbol Cygnus alphabet
seq = ["I", "V", "vi", "IV", "I", "V", "I"] * 50
H = shannon_scale_degrees(seq)  # Jeffreys-Laplace smoothing, log base 2
print(f"Shannon entropy: {H:.3f} bits")
```

## Notebooks

Two executed notebooks live in [`notebooks/`](notebooks/) and double as the documentation:

- **[01_introduction.ipynb](notebooks/01_introduction.ipynb)** — pedagogical tour of all 9 metrics on synthetic examples whose answers are known analytically (uniform → `log2(N)`, perfect Zipf → `alpha = 1`, white noise → `D = 2`, etc.).
- **[02_paper_reproduction.ipynb](notebooks/02_paper_reproduction.ipynb)** — reproduces three flagship findings of the Cygnus arXiv paper from bundled real scale-degree counts (8 composers, ~250K observations): Shannon entropy range `[3.33, 3.86]` bits, KL matrix recovering documented stylistic lineages, Zipf-on-transitions historical vs neoclassical gap.

To execute them locally:

```bash
pip install -e ".[dev]" jupyter matplotlib
jupyter lab notebooks/
```

## Reproducibility

Each metric is unit-tested against (a) theoretical anchors on canonical inputs and (b) exact parity with the Cygnus reference implementation. The 181-test suite runs in under one second. Releases are archived on Zenodo with a citable DOI.

## Citation

If you use `vega-mir` in your research, please cite the archived release via its DOI. The `CITATION.cff` file at the repo root carries the same metadata in machine-readable form (GitHub renders a "Cite this repository" widget directly from it).

```bibtex
@software{jalbertdesforges_2026_vegamir,
  author    = {Jalbert-Desforges, Fred},
  title     = {vega-mir: A Python library for information-theoretic
               analysis of symbolic music},
  version   = {v0.0.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19711033},
  url       = {https://doi.org/10.5281/zenodo.19711033}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built within the [CYGNUS ANALYSIS](https://cygnusanalysis.com) research program. Methodology validated against the certified Cygnus pipeline (F1 = 0.9791 on 1238 MAESTRO pieces, ~6.3M notes).
