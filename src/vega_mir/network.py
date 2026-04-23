"""Network analysis on chord-progression graphs.

A chord progression induces a directed, weighted graph: nodes are
distinct chords (or scale degrees), edges connect successive chords,
and edge weights are transition probabilities. Standard network
metrics applied to this graph reveal structural properties of the
underlying harmonic style:

* **PageRank** identifies the *gravity center* — the chord toward which
  the harmonic flow tends to settle.
* **Mean clustering coefficient** measures how locally cliquey the
  transitions are.
* **Modularity-based community detection** groups chords that share
  similar transition neighborhoods.
* **Diameter and average shortest path** on the largest strongly
  connected component capture the "distance" of the harmonic universe.
* The *small-world* flag is the conjunction of high clustering and
  short average paths.

Used in the Cygnus methodology (Jalbert-Desforges, 2026).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import networkx as nx
import numpy as np

from vega_mir.shannon import collapse_repetitions


class NetworkAnalysis(NamedTuple):
    """Network analysis metrics on a directed weighted chord graph.

    Attributes
    ----------
    n_nodes : int
        Number of distinct chords (graph nodes).
    n_edges : int
        Number of transitions present (graph edges).
    density : float
        ``n_edges / (n_nodes * (n_nodes - 1))`` for a directed graph.
    pagerank : dict[str, float]
        Mapping ``chord -> PageRank score``. Sums to ~ 1.
    gravity_center : str
        Chord with the highest PageRank.
    mean_clustering : float
        Mean local clustering coefficient (computed on the directed graph).
    n_communities : int
        Number of communities detected by greedy modularity on the
        undirected projection.
    communities : tuple of tuple of str
        Partition of nodes into communities; each inner tuple is sorted.
    largest_scc_size : int
        Size of the largest strongly connected component.
    diameter : int
        Diameter of the largest strongly connected component.
    avg_shortest_path : float
        Average shortest path length within the largest SCC.
    small_world_candidate : bool
        ``True`` if ``mean_clustering > 0.3`` and ``avg_shortest_path < 3.0``
        (the heuristic used in the Cygnus methodology).
    """

    n_nodes: int
    n_edges: int
    density: float
    pagerank: dict[str, float]
    gravity_center: str
    mean_clustering: float
    n_communities: int
    communities: tuple[tuple[str, ...], ...]
    largest_scc_size: int
    diameter: int
    avg_shortest_path: float
    small_world_candidate: bool


def chord_graph(
    transitions: Mapping[str, Mapping[str, float]],
    threshold: float = 0.0,
) -> nx.DiGraph:
    """Build a directed weighted graph from a transition matrix dict.

    Parameters
    ----------
    transitions : mapping of str to (mapping of str to float)
        ``transitions[src][tgt]`` is the weight of the directed edge
        ``src -> tgt``. Typically a transition probability.
    threshold : float, default 0.0
        Edges with weight strictly less than or equal to ``threshold``
        are excluded. Use ``0.01`` to match the Cygnus methodology.

    Returns
    -------
    networkx.DiGraph
    """
    g = nx.DiGraph()
    for src, targets in transitions.items():
        for tgt, weight in targets.items():
            if weight > threshold:
                g.add_edge(src, tgt, weight=float(weight))
    return g


def chord_graph_from_sequence(
    sequence: Sequence[str],
    alphabet: Sequence[str] | None = None,
    threshold: float = 0.0,
    collapse: bool = True,
) -> nx.DiGraph:
    """Build a directed weighted graph from a chord sequence via bigrams.

    Bigram counts are normalized per source row (so out-edge weights
    sum to 1 per node).

    Parameters
    ----------
    sequence : sequence of str
        Chord or scale-degree sequence.
    alphabet : sequence of str, optional
        If given, only nodes in this alphabet are considered (others
        are dropped from the graph and break their surrounding bigrams).
    threshold : float, default 0.0
        Edges with normalized weight ``<= threshold`` are excluded.
    collapse : bool, default True
        If ``True``, collapse consecutive duplicates before extracting
        bigrams (i.e., exclude self-loops induced by held chords).

    Returns
    -------
    networkx.DiGraph
    """
    seq = collapse_repetitions(sequence) if collapse else list(sequence)
    if alphabet is not None:
        valid = set(alphabet)
        seq = [s for s in seq if s in valid]
    bigrams: Counter = Counter()
    src_totals: Counter = Counter()
    for s, t in zip(seq[:-1], seq[1:], strict=False):
        bigrams[(s, t)] += 1
        src_totals[s] += 1
    transitions: dict[str, dict[str, float]] = {}
    for (s, t), count in bigrams.items():
        weight = count / src_totals[s]
        transitions.setdefault(s, {})[t] = weight
    return chord_graph(transitions, threshold=threshold)


def network_analysis(graph: nx.DiGraph) -> NetworkAnalysis:
    """Compute the suite of network metrics on a chord graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed weighted graph (edges should carry a ``weight`` attribute
        for PageRank). Must have at least 2 nodes.

    Returns
    -------
    NetworkAnalysis

    Raises
    ------
    ValueError
        If the graph has fewer than 2 nodes.
    """
    if graph.number_of_nodes() < 2:
        raise ValueError(
            f"graph must have at least 2 nodes, got {graph.number_of_nodes()}"
        )

    pagerank = nx.pagerank(graph, weight="weight")
    gravity_center = max(pagerank, key=lambda k: pagerank[k])

    clustering = nx.clustering(graph)
    mean_clustering = float(np.mean(list(clustering.values()))) if clustering else 0.0

    from networkx.algorithms.community import greedy_modularity_communities

    undirected = graph.to_undirected()
    communities_raw = list(greedy_modularity_communities(undirected))
    communities = tuple(tuple(sorted(c)) for c in communities_raw)

    sccs = list(nx.strongly_connected_components(graph))
    largest_scc = max(sccs, key=len)
    if len(largest_scc) >= 2:
        sub = graph.subgraph(largest_scc)
        diameter = int(nx.diameter(sub))
        avg_path = float(nx.average_shortest_path_length(sub))
    else:
        diameter = 0
        avg_path = 0.0

    small_world = bool(mean_clustering > 0.3 and 0.0 < avg_path < 3.0)

    return NetworkAnalysis(
        n_nodes=int(graph.number_of_nodes()),
        n_edges=int(graph.number_of_edges()),
        density=float(nx.density(graph)),
        pagerank={k: float(v) for k, v in pagerank.items()},
        gravity_center=str(gravity_center),
        mean_clustering=mean_clustering,
        n_communities=len(communities),
        communities=communities,
        largest_scc_size=len(largest_scc),
        diameter=diameter,
        avg_shortest_path=avg_path,
        small_world_candidate=small_world,
    )
