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
that classifies rubato into four categories. Each metric ships with
theoretical ground-truth tests on canonical inputs, exact parity
against the reference implementation in the upstream Cygnus pipeline,
and explicit error handling on degenerate inputs. The 181-test suite
runs in under one second.

# Statement of need

Symbolic music research routinely computes information-theoretic
quantities — Shannon entropy, KL divergence, Zipf fits — to summarise
harmonic and melodic distributions and to compare composers, periods,
or genres. Each new study tends to roll its own implementation:
stitched-together calls to `scipy.stats.entropy`, hand-rolled Laplace
smoothing, custom KL formulas, ad-hoc NetworkX glue. There is no
shared API, no shared test suite, and no shared reference values
against which results can be cross-validated.

`vega-mir` addresses this gap. Its public API is small (one function
plus convenience helpers per metric), strictly typed, and the
parameter defaults match a documented, peer-reviewable methodology
[@jalbert2026cygnus]. Each metric is validated by both analytic
ground truths (uniform Shannon equals `log2(N)`, perfect Zipf
generated as `P_i ∝ 1/i` recovers `α = 1` with `R² = 1`, Higuchi on
white noise approaches `D = 2`, and so on) and by exact parity
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

# State of the field

Several mature Python and Java libraries cover adjacent problems in
symbolic music research, but none directly target information-theoretic
distributional analysis as a first-class concern. `music21`
[@cuthbert2010music21] is a comprehensive computational-musicology
framework with strong parsing, harmonic analysis, and key-detection
facilities; its analytical surface is broad but does not expose
dedicated entropy, divergence, or power-law fitting primitives.
`partitura` [@cancino2022partitura] specialises in lossless score
parsing and performance alignment, leaving downstream statistical
analysis to the user. `jSymbolic` [@mckay2018jsymbolic] is a
comprehensive Java-based feature extractor whose primary focus is
broad histogram-based descriptors, and which is not directly callable
from a Python pipeline. `musif` [@llorens2023musif] targets stylometric
feature extraction with a similarity-classification framing rather
than an information-theoretic one; entropy and divergence are not
first-class outputs.

Researchers therefore typically compose calls to `scipy.stats.entropy`,
hand-rolled Laplace smoothing, custom KL formulas, and ad-hoc NetworkX
code each time they need information-theoretic summaries on symbolic
data. `vega-mir` consolidates these computations behind a single
tested API with consistent defaults and a shared set of reference
values, allowing direct comparison of results across studies.

# Software design

`vega-mir` targets Python 3.10 and above and depends on `numpy`
[@harris2020numpy], `scipy` [@virtanen2020scipy], `networkx`
[@hagberg2008networkx], `mido`, and `pretty_midi`. The package uses
the modern `src` layout with `hatchling` as the build backend.

The public API follows a uniform three-layer pattern per metric: a
primitive function that operates on a normalized probability vector
or 1-D time series, a counts-based convenience that applies Laplace
smoothing on a fixed alphabet, and a sequence-based convenience for
the most common chord or scale-degree input. Each computation that
returns more than a single scalar is wrapped in a `NamedTuple` for
structured, immutable, type-checkable access (`ZipfFit`,
`IntervalAnalysis`, `NetworkAnalysis`, `RubatoAnalysis`,
`StationarityResult`, `FractalDimension`, `DominantPeriod`). Type
hints are present throughout, and `mypy --strict` runs on every push.
Defaults match the documented Cygnus methodology v2 (Jeffreys
`α = 0.5` smoothing, log base 2, consecutive-duplicate collapsing on
scale-degree sequences). Degenerate inputs (too-short sequences, a
single distinct symbol, sub-threshold sample sizes) raise `ValueError`
with explicit messages rather than returning silent zeros, and the
tests cover those boundaries explicitly.

A continuous-integration matrix on GitHub Actions exercises the
181-test suite on `ubuntu-latest`, `macos-latest`, and `windows-latest`
against Python 3.10 through 3.13 with `ruff`, `mypy --strict`, and
`pytest`. Releases are archived on Zenodo and assigned a citable DOI.

# Research impact statement

By packaging the nine metrics behind a tested, citable API,
`vega-mir` lowers the activation cost for several lines of inquiry
in symbolic music research. Cross-corpus and cross-composer
distributional comparisons (KL, JS, Gini) become a single function
call rather than a custom pipeline, encouraging studies that were
previously deferred for engineering reasons. The Higuchi fractal
dimension and the spectral rubato analysis open the symbolic-music
toolkit to descriptors traditionally restricted to neuroscience and
signal-processing literatures. The reproduction notebook included
with the library demonstrates that flagship findings of an upstream
analysis pipeline [@jalbert2026cygnus] are recovered exactly from
bundled aggregated counts on eight composers, providing an
end-to-end reproducibility template for downstream studies. Beyond
music, the alphabet-agnostic functions (`shannon_entropy`,
`kl_divergence`, `gini`, `higuchi_fractal_dimension`) are usable in
any discrete-sequence or 1-D time-series domain.

# AI usage disclosure

`vega-mir` was developed with assistance from the Claude language
model (Anthropic, claude-opus-4.7) for code drafting, test design,
documentation, and paper preparation. All scientific decisions —
selection of the nine metrics, choice of methodology defaults,
validation strategy, and the final review of every commit and
paragraph — remain the responsibility of the author. AI-assisted
commits in the repository carry the standard `Co-Authored-By` git
trailer for transparency. The core algorithmic content of `vega-mir`
reproduces a previously documented methodology and is independently
validated against a human-written reference implementation, so the
AI assistance affects development velocity and editorial polish
rather than scientific substance.

# Acknowledgements

`vega-mir` extracts the methodology already documented and validated
in the upstream Cygnus pipeline [@jalbert2026cygnus] and packages it
as a standalone, citable library for the broader MIR community. The
author thanks the maintainers of NumPy, SciPy, NetworkX, and the
Jupyter stack on which `vega-mir` rests.

# References
