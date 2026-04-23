---
title: 'vega-mir: A Python library for information-theoretic analysis of symbolic music'
tags:
  - Python
  - music information retrieval
  - symbolic music
  - information theory
  - entropy
  - Kullback-Leibler divergence
  - Zipf's law
  - musicology
authors:
  - name: Fred Jalbert-Desforges
    orcid: 0009-0002-4357-6942
    affiliation: 1
affiliations:
  - name: Independent Researcher, Cygnus Analysis, Montreal, Canada
    index: 1
date: 23 April 2026
bibliography: paper.bib
---

# Summary

`vega-mir` is a focused, well-tested Python library that bundles nine
information-theoretic and statistical metrics for the analysis of
symbolic music corpora. Given a sequence of scale degrees, a chord
progression graph, or a tempo curve, the library returns Shannon
entropy [@shannon1948mathematical], Zipf's law fits on marginal and
joint distributions, Kullback-Leibler and Jensen-Shannon divergences
(with a pairwise matrix builder), the Gini coefficient, a chi-squared
stationarity test with Cramer's V effect size, Exponential / Laplace
fits on melodic intervals, network-analysis metrics on chord graphs
(PageRank, clustering, modularity-based community detection), the
Higuchi fractal dimension of a 1-D time series
[@higuchi1988approach], and a spectral analysis of the tempo curve
that classifies rubato into four categories. Each metric ships
with theoretical ground-truth tests on canonical inputs, exact parity
against the reference implementation in the upstream Cygnus pipeline,
and explicit error handling on degenerate inputs. The 181-test suite
runs in under one second.

# Statement of need

Symbolic music research routinely computes information-theoretic
quantities — Shannon entropy, KL divergence, Zipf fits — to summarise
harmonic and melodic distributions and to compare composers, periods
or genres. Yet no current Python library exposes these as first-class,
focused, well-tested functions. The standard symbolic toolboxes solve
adjacent problems: `music21` [@cuthbert2010music21] provides a broad
computational-musicology surface (parsing, harmonic analysis, key
detection) without dedicated information-theoretic primitives;
`partitura` [@cancino2022partitura] focuses on score parsing and
performance alignment; `jSymbolic` [@mckay2018jsymbolic] is a
comprehensive Java-based feature extractor whose primary focus is
broad histogram-based descriptors rather than information-theoretic
primitives, and which is not directly callable from a Python pipeline; `musif`
[@llorens2023musif] targets stylometric feature extraction with a
similarity-classification framing rather than an information-theoretic
one. Researchers therefore stitch together `scipy.stats.entropy`,
hand-rolled smoothing, custom KL implementations, and ad hoc
NetworkX glue every time, with no shared API, no shared tests, and
no shared reference values.

`vega-mir` fills that gap. Its public API is small (one function
plus convenience helpers per metric), strictly typed, and parameter
defaults match a documented, peer-reviewable methodology
[@jalbert2026cygnus]. Each metric is validated by both analytic
ground truths (uniform Shannon equals `log2(N)`, perfect Zipf
generated as `P_i ∝ 1/i` recovers `α = 1` with `R² = 1`, Higuchi
on white noise approaches `D = 2`, and so on) and by exact parity
against the reference computation used in the upstream Cygnus
pipeline. Two executed Jupyter notebooks ship with the library: a
pedagogical tour on synthetic inputs whose answers are known
analytically, and a reproduction notebook that recovers three
flagship findings of the Cygnus arXiv paper from bundled real
scale-degree counts on eight composers (Bach, Haydn, Beethoven,
Chopin, Liszt, Rachmaninoff, Glass, Richter): the Shannon entropy
range of `[3.33, 3.86]` bits across the 15-symbol alphabet, the
Kullback-Leibler matrix recovering documented stylistic lineages
without supervision, and the Zipf-on-transitions gap between
historical and neoclassical composers.

The intended audience is researchers in music information retrieval,
computational musicology, and complex-systems studies of music who
need a citable, reproducible toolkit for distributional analyses of
symbolic corpora.

# Implementation

`vega-mir` targets Python 3.10 and above and depends on `numpy`
[@harris2020numpy], `scipy` [@virtanen2020scipy], `networkx`
[@hagberg2008networkx], `mido`, and `pretty_midi`. The package uses
the modern `src` layout with `hatchling` as the build backend. The
public API is exposed at the top level (e.g. `vega_mir.shannon_entropy`,
`vega_mir.kl_divergence`, `vega_mir.network_analysis`); each metric
returns a `NamedTuple` for structured results. Type hints are present
throughout; the test suite is run on every push by GitHub Actions
across the matrix `ubuntu-latest`, `macos-latest`, `windows-latest` x
Python 3.10 - 3.13 with `ruff`, `mypy --strict`, and `pytest`. Releases
are archived on Zenodo and given a citable DOI.

A minimal example, using the high-level convenience for the marginal
Shannon entropy on a scale-degree sequence:

```python
from vega_mir import shannon_scale_degrees

seq = ["I", "V", "vi", "IV", "I", "V", "I"] * 50
H = shannon_scale_degrees(seq)
print(f"Shannon entropy: {H:.3f} bits")
```

The Higuchi fractal dimension of a 1-D series:

```python
import numpy as np
from vega_mir import higuchi_fractal_dimension

rng = np.random.default_rng(42)
brownian = np.cumsum(rng.normal(size=2000))
result = higuchi_fractal_dimension(brownian)
print(f"D = {result.dimension:.3f}")  # ~ 1.5 for fractional Brownian motion
```

# Acknowledgements

`vega-mir` extracts the methodology already documented and validated
in the Cygnus pipeline [@jalbert2026cygnus] and packages it as a
standalone, citable library for the broader MIR community. The author
thanks the developers of NumPy, SciPy, NetworkX and the Jupyter stack,
on which `vega-mir` rests.

# References
