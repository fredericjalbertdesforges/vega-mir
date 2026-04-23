# vega-mir

Information-theoretic analysis of symbolic music (MIDI).

[![CI](https://github.com/fredericjalbertdesforges/vega-mir/actions/workflows/ci.yml/badge.svg)](https://github.com/fredericjalbertdesforges/vega-mir/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/vega-mir.svg)](https://pypi.org/project/vega-mir/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

```python
from vega_mir import shannon_scale_degrees

H = shannon_scale_degrees("path/to/score.mid")
print(f"Shannon entropy: {H:.3f} bits")
```

## Reproducibility

Each metric is unit-tested against published values from the Cygnus arXiv paper. Releases are archived on Zenodo with a citable DOI.

## Citation

If you use `vega-mir` in your research, please cite it via the `CITATION.cff` file or:

```bibtex
@software{vega_mir,
  author    = {Jalbert-Desforges, Fred},
  title     = {vega-mir: Information-theoretic analysis of symbolic music},
  year      = {2026},
  publisher = {Zenodo},
  url       = {https://github.com/fredericjalbertdesforges/vega-mir}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built within the [CYGNUS ANALYSIS](https://cygnusanalysis.com) research program. Methodology validated against the certified Cygnus pipeline (F1 = 0.9791 on 1238 MAESTRO pieces, ~6.3M notes).
