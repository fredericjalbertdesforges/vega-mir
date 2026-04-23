# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-04-23

First public release.

### Added

- **Nine information-theoretic metrics** for symbolic music analysis,
  each shipped with a dedicated module, a `NamedTuple` return type where
  appropriate, analytic ground-truth tests, and exact parity against the
  upstream Cygnus reference implementation:
  - `shannon`: Shannon entropy, smoothed probability helper, collapse
    helper, 15-symbol Cygnus scale-degree alphabet constant
  - `zipf`: marginal and joint Zipf OLS fits (`ZipfFit` NamedTuple)
  - `kl`: asymmetric Kullback-Leibler and symmetric Jensen-Shannon
    divergences, counts-based convenience, pairwise matrix builder
  - `gini`: Gini coefficient with `from_counts` and multi-dimension helpers
  - `stationarity`: chi-squared contingency test on segmented sequences
    with Cramer's V effect size (`StationarityResult` NamedTuple)
  - `intervals`: Exponential / Laplace MLE fits on absolute intervals
    with KS scoring and summary statistics (`IntervalAnalysis` NamedTuple)
  - `network`: directed weighted graph builders and a `network_analysis`
    function returning PageRank, clustering, modularity-based communities,
    diameter on the largest strongly-connected component, and a
    small-world flag (`NetworkAnalysis` NamedTuple)
  - `fractal`: Higuchi fractal dimension on 1-D time series
    (`FractalDimension` NamedTuple)
  - `rubato`: FFT-based spectral analysis of tempo curves with four-way
    rubato classification (`RubatoAnalysis` and `DominantPeriod`
    NamedTuples)
- **181 unit tests** passing in under one second, covering theoretical
  anchors (uniform Shannon, perfect Zipf, deterministic Gini, white-noise
  Higuchi, sinusoidal rubato, etc.) and exact parity with the Cygnus
  reference implementation where applicable.
- **Two executed Jupyter notebooks** under `notebooks/`:
  - `01_introduction.ipynb` — pedagogical tour of the nine metrics on
    synthetic inputs whose answers are known analytically.
  - `02_paper_reproduction.ipynb` — recovers three flagship findings of
    the Cygnus arXiv paper from bundled real scale-degree counts on
    eight composers (Shannon entropy range, KL matrix stylistic lineages,
    Zipf-on-transitions historical vs neoclassical gap).
- **Bundled demonstration data** under `notebooks/data/`
  (~410 KB total): aggregated marginal and joint counts per composer,
  five sample scale-degree sequences per composer, and the subset of
  published Cygnus paper reference values for parity comparison.
- **JOSS paper draft** under `paper/` (`paper.md` + `paper.bib`),
  within the JOSS word budget (689 words).
- **Packaging** with the `src` layout, Hatchling build backend, MIT
  license, `CITATION.cff`, and a GitHub Actions CI matrix covering
  ubuntu-latest, macos-latest, windows-latest x Python 3.10, 3.11, 3.12
  and 3.13 with `ruff`, `mypy --strict`, and `pytest` (12 green jobs).
