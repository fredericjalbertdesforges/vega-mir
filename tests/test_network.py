"""Tests for vega_mir.network."""

from __future__ import annotations

import networkx as nx
import pytest

from vega_mir import (
    NetworkAnalysis,
    chord_graph,
    chord_graph_from_sequence,
    network_analysis,
)


class TestChordGraph:
    def test_builds_edges_from_dict(self) -> None:
        transitions = {
            "I": {"V": 0.5, "IV": 0.3, "vi": 0.2},
            "V": {"I": 0.8, "vi": 0.2},
        }
        g = chord_graph(transitions)
        assert g.number_of_nodes() == 4  # I, V, IV, vi
        assert g.number_of_edges() == 5

    def test_threshold_prunes_edges(self) -> None:
        transitions = {
            "I": {"V": 0.5, "IV": 0.3, "vi": 0.005},  # vi below 0.01
        }
        g = chord_graph(transitions, threshold=0.01)
        assert ("I", "vi") not in g.edges()
        assert ("I", "V") in g.edges()

    def test_edge_weights_preserved(self) -> None:
        transitions = {"I": {"V": 0.42}}
        g = chord_graph(transitions)
        assert g["I"]["V"]["weight"] == pytest.approx(0.42)

    def test_empty_dict_returns_empty_graph(self) -> None:
        g = chord_graph({})
        assert g.number_of_nodes() == 0


class TestChordGraphFromSequence:
    def test_simple_sequence(self) -> None:
        seq = ["I", "V", "I", "V", "I"]
        g = chord_graph_from_sequence(seq)
        # Bigrams: (I,V) ×2, (V,I) ×2 → out-degree-normalized
        # I: V at weight 1.0; V: I at weight 1.0
        assert g["I"]["V"]["weight"] == pytest.approx(1.0)
        assert g["V"]["I"]["weight"] == pytest.approx(1.0)

    def test_collapse_default(self) -> None:
        # ["I", "I", "V", "V"] → collapsed to ["I", "V"] → only one bigram
        seq = ["I", "I", "I", "V", "V", "V"]
        g = chord_graph_from_sequence(seq, collapse=True)
        assert g.number_of_edges() == 1
        assert ("I", "V") in g.edges()
        # No self-loops because of collapse
        assert ("I", "I") not in g.edges()

    def test_collapse_off_includes_self_loops(self) -> None:
        seq = ["I", "I", "V"]
        g = chord_graph_from_sequence(seq, collapse=False)
        assert ("I", "I") in g.edges()

    def test_alphabet_filter(self) -> None:
        seq = ["I", "X", "V", "Y", "I"]
        g = chord_graph_from_sequence(seq, alphabet=["I", "V"])
        assert "X" not in g.nodes()
        assert "Y" not in g.nodes()

    def test_per_source_normalization(self) -> None:
        # I has out-degree weights summing to 1
        seq = ["I", "V", "I", "IV", "I", "V", "I", "IV", "I", "V"] * 10
        g = chord_graph_from_sequence(seq)
        out_weights = sum(d["weight"] for _, _, d in g.out_edges("I", data=True))
        assert out_weights == pytest.approx(1.0)


class TestNetworkAnalysisFields:
    def test_namedtuple(self) -> None:
        seq = ["I", "V", "vi", "IV"] * 25
        g = chord_graph_from_sequence(seq)
        result = network_analysis(g)
        assert isinstance(result, NetworkAnalysis)
        for field in (
            "n_nodes", "n_edges", "density", "pagerank", "gravity_center",
            "mean_clustering", "n_communities", "communities",
            "largest_scc_size", "diameter", "avg_shortest_path",
            "small_world_candidate",
        ):
            assert hasattr(result, field)


class TestNetworkAnalysisOnKnownGraphs:
    def test_complete_graph_has_density_one(self) -> None:
        # Complete directed graph on 4 nodes has density 1
        g = nx.DiGraph()
        nodes = ["I", "V", "IV", "vi"]
        for s in nodes:
            for t in nodes:
                if s != t:
                    g.add_edge(s, t, weight=1.0 / 3)
        result = network_analysis(g)
        assert result.density == pytest.approx(1.0)
        assert result.n_nodes == 4
        assert result.n_edges == 12  # 4 * 3

    def test_pagerank_sums_to_one(self) -> None:
        seq = ["I", "V", "IV", "vi", "ii", "iii", "VII"] * 30
        g = chord_graph_from_sequence(seq)
        result = network_analysis(g)
        assert sum(result.pagerank.values()) == pytest.approx(1.0, abs=1e-6)

    def test_gravity_center_in_pagerank(self) -> None:
        seq = ["I", "V", "IV", "vi"] * 25
        g = chord_graph_from_sequence(seq)
        result = network_analysis(g)
        assert result.gravity_center in result.pagerank
        assert result.pagerank[result.gravity_center] == max(result.pagerank.values())

    def test_communities_partition_nodes(self) -> None:
        seq = ["I", "V", "IV", "vi", "ii", "iii", "VII"] * 30
        g = chord_graph_from_sequence(seq)
        result = network_analysis(g)
        community_nodes: set[str] = set()
        for community in result.communities:
            community_nodes.update(community)
        assert community_nodes == set(g.nodes())
        # Communities should be disjoint
        total = sum(len(c) for c in result.communities)
        assert total == g.number_of_nodes()

    def test_largest_scc_at_least_one(self) -> None:
        seq = ["I", "V", "I", "V"] * 25
        g = chord_graph_from_sequence(seq)
        result = network_analysis(g)
        assert result.largest_scc_size >= 2  # I ↔ V is mutually reachable

    def test_diameter_zero_for_disconnected(self) -> None:
        # Chain I → V → IV with no return → no SCC of size ≥ 2
        g = nx.DiGraph()
        g.add_edge("I", "V", weight=1.0)
        g.add_edge("V", "IV", weight=1.0)
        result = network_analysis(g)
        # Largest SCC is a singleton → diameter falls back to 0
        assert result.diameter == 0

    def test_small_world_flag_logic(self) -> None:
        # Build a tightly clustered cycle + extra chords to force conditions
        # Easy case: ring of 4 mutually connected nodes
        g = nx.DiGraph()
        nodes = ["I", "V", "IV", "vi"]
        for s in nodes:
            for t in nodes:
                if s != t:
                    g.add_edge(s, t, weight=1.0 / 3)
        result = network_analysis(g)
        # Complete graph: clustering = 1.0, avg_path = 1.0 → small-world candidate
        assert result.mean_clustering > 0.3
        assert result.avg_shortest_path < 3.0
        assert result.small_world_candidate is True


class TestEdgeCases:
    def test_single_node_raises(self) -> None:
        g = nx.DiGraph()
        g.add_node("I")
        with pytest.raises(ValueError, match="at least 2 nodes"):
            network_analysis(g)

    def test_empty_graph_raises(self) -> None:
        g = nx.DiGraph()
        with pytest.raises(ValueError, match="at least 2 nodes"):
            network_analysis(g)


class TestCygnusParity:
    """Parity with Cygnus math_metrics.network_analysis."""

    def test_matches_cygnus_pageranking(self) -> None:
        transitions = {
            "I": {"V": 0.5, "IV": 0.3, "vi": 0.2},
            "V": {"I": 0.7, "vi": 0.3},
            "IV": {"I": 0.6, "V": 0.4},
            "vi": {"ii": 0.5, "IV": 0.5},
            "ii": {"V": 1.0},
        }
        g = chord_graph(transitions, threshold=0.01)
        result = network_analysis(g)

        # Reference: Cygnus uses nx.pagerank with weight="weight"
        expected = nx.pagerank(g, weight="weight")
        assert set(result.pagerank.keys()) == set(expected.keys())
        for k, v in expected.items():
            assert result.pagerank[k] == pytest.approx(v)

    def test_matches_cygnus_clustering(self) -> None:
        transitions = {
            "I": {"V": 0.5, "IV": 0.3, "vi": 0.2},
            "V": {"I": 0.7, "vi": 0.3},
            "IV": {"I": 0.6, "V": 0.4},
            "vi": {"ii": 0.5, "IV": 0.5},
            "ii": {"V": 1.0},
        }
        g = chord_graph(transitions, threshold=0.01)
        result = network_analysis(g)

        clustering = nx.clustering(g)
        import numpy as np
        expected_mean = float(np.mean(list(clustering.values())))
        assert result.mean_clustering == pytest.approx(expected_mean)

    def test_matches_cygnus_density(self) -> None:
        transitions = {
            "I": {"V": 0.5, "IV": 0.3, "vi": 0.2},
            "V": {"I": 0.7, "vi": 0.3},
        }
        g = chord_graph(transitions, threshold=0.01)
        result = network_analysis(g)
        assert result.density == pytest.approx(nx.density(g))
