# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the effective (resolved) graph view.

These tests use fixed DOT inputs and hard-coded expectations; they do
not invoke Graphviz to determine the answer.
"""

from __future__ import annotations

import textwrap

import pytest

import pydot
from pydot import EffectiveGraph, effective_graph


def parse(dot: str) -> pydot.Dot:
    graphs = pydot.graph_from_dot_data(textwrap.dedent(dot))
    assert graphs is not None
    return graphs[0]


def attrs(objs):
    return [dict(o.get_attributes()) for o in objs]


def test_module_level_and_method_api_agree() -> None:
    graph = parse("digraph { a -> b; }")
    assert isinstance(graph.effective_graph(), EffectiveGraph)
    assert isinstance(effective_graph(graph), EffectiveGraph)


def test_node_defaults_applied_but_not_retroactive() -> None:
    graph = parse(
        """
        digraph G {
          node [shape=box, color=red];
          a;
          node [shape=circle, color=blue];
          b;
        }
        """
    )
    view = graph.effective_graph()

    nodes = {n.get_name(): n for n in view.get_nodes()}
    assert attrs([nodes["a"]]) == [{"shape": "box", "color": "red"}]
    # `b` takes the new defaults; `a` is not revisited.
    assert attrs([nodes["b"]]) == [{"shape": "circle", "color": "blue"}]


def test_edge_defaults_and_implicit_endpoint_nodes() -> None:
    graph = parse(
        """
        digraph G {
          node [shape=box];
          edge [color=black];
          a -> b;
          edge [color=green];
          b -> c;
        }
        """
    )
    view = graph.effective_graph()

    # Endpoint nodes are materialized implicitly, in first-touch order.
    assert [n.get_name() for n in view.get_nodes()] == ["a", "b", "c"]
    assert all(
        n.get_attributes().get("shape") == "box" for n in view.get_nodes()
    )

    edges = view.get_edges()
    assert [(e.get_source(), e.get_destination()) for e in edges] == [
        ("a", "b"),
        ("b", "c"),
    ]
    assert attrs(edges) == [
        {"color": "black"},
        {"color": "green"},
    ]


def test_node_restatement_merges_attributes() -> None:
    graph = parse(
        """
        digraph G {
          node [shape=box];
          a [color=red];
          a [label="A"];
        }
        """
    )
    view = graph.effective_graph()
    assert [n.get_name() for n in view.get_nodes()] == ["a"]
    assert attrs(view.get_nodes()) == [
        {"shape": "box", "color": "red", "label": '"A"'}
    ]


def test_subgraph_default_scope_and_cross_subgraph_restatement() -> None:
    graph = parse(
        """
        digraph G {
          node [shape=box, color=black];
          outer;

          subgraph cluster_0 {
            node [color=red];
            inner [shape=circle];

            subgraph cluster_1 {
              node [shape=triangle];
              deep;
            }

            inner [label="I"];
          }

          inner [color=blue];
        }
        """
    )
    view = graph.effective_graph()

    effective = {n.get_name(): n for n in view.get_nodes()}
    assert attrs([effective["outer"]]) == [
        {"shape": "box", "color": "black"}
    ]
    # Nested scope inherits the red color from cluster_0.
    assert attrs([effective["deep"]]) == [
        {"shape": "triangle", "color": "red"}
    ]
    # `inner` is one global node; restating it merges attributes across
    # the subgraph boundary without re-applying any defaults.
    assert attrs([effective["inner"]]) == [
        {"shape": "circle", "color": "blue", "label": '"I"'}
    ]

    cluster_0 = view.get_subgraphs()[0]
    cluster_1 = cluster_0.get_subgraphs()[0]
    assert [n.get_name() for n in cluster_0.get_nodes()] == [
        "inner",
        "deep",
    ]
    assert [n.get_name() for n in cluster_1.get_nodes()] == ["deep"]


def test_subgraph_defaults_do_not_escape_their_scope() -> None:
    graph = parse(
        """
        digraph G {
          node [shape=box];
          subgraph cluster_0 {
            node [color=red];
            a;
          }
          b;
        }
        """
    )
    view = graph.effective_graph()
    nodes = {n.get_name(): n for n in view.get_nodes()}
    assert attrs([nodes["a"]]) == [{"shape": "box", "color": "red"}]
    assert attrs([nodes["b"]]) == [{"shape": "box"}]


def test_graph_defaults_in_subgraph_apply_to_that_subgraph() -> None:
    graph = parse(
        """
        digraph G {
          rankdir=LR;
          subgraph cluster_0 {
            graph [style=filled, color=lightgrey];
            a;
          }
        }
        """
    )
    view = graph.effective_graph()
    assert view.get_attributes() == {"rankdir": "LR"}
    cluster = view.get_subgraphs()[0]
    assert cluster.get_attributes() == {
        "style": "filled",
        "color": "lightgrey",
    }


def test_subgraph_endpoint_expands_to_node_product() -> None:
    graph = parse(
        """
        digraph G {
          { a; b; } -> { c; d; };
        }
        """
    )
    view = graph.effective_graph()
    assert [n.get_name() for n in view.get_nodes()] == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert [
        (e.get_source(), e.get_destination()) for e in view.get_edges()
    ] == [
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
    ]


def test_subgraph_endpoint_expansion_includes_nested_members() -> None:
    graph = parse(
        """
        digraph G {
          { a; subgraph { b; } } -> c;
        }
        """
    )
    view = graph.effective_graph()
    assert [
        (e.get_source(), e.get_destination()) for e in view.get_edges()
    ] == [("a", "c"), ("b", "c")]


def test_strict_directed_graph_merges_duplicate_edges() -> None:
    graph = parse(
        """
        strict digraph G {
          edge [color=black];
          a -> b [penwidth=2];
          edge [color=green];
          a -> b [style=dashed];
        }
        """
    )
    view = graph.effective_graph()
    edges = view.get_edges()
    assert len(edges) == 1
    # First edge keeps creation-time defaults; later defaults are not
    # applied retroactively. Explicit attributes from the duplicate
    # statement merge in.
    assert attrs(edges) == [
        {"color": "black", "penwidth": "2", "style": "dashed"}
    ]


def test_non_strict_directed_graph_keeps_parallel_edges() -> None:
    graph = parse(
        """
        digraph G {
          a -> b [penwidth=2];
          a -> b [style=dashed];
        }
        """
    )
    view = graph.effective_graph()
    edges = view.get_edges()
    assert len(edges) == 2
    assert attrs(edges) == [
        {"penwidth": "2"},
        {"style": "dashed"},
    ]


def test_strict_undirected_graph_treats_reverse_as_duplicate() -> None:
    graph = parse(
        """
        strict graph G {
          edge [color=black];
          a -- b [penwidth=2];
          b -- a [color=red, style=dashed];
        }
        """
    )
    view = graph.effective_graph()
    edges = view.get_edges()
    assert len(edges) == 1
    assert attrs(edges) == [
        {"color": "red", "penwidth": "2", "style": "dashed"}
    ]


def test_non_strict_undirected_graph_keeps_reverse_parallel_edges() -> None:
    graph = parse(
        """
        graph G {
          a -- b [penwidth=2];
          b -- a [style=dashed];
        }
        """
    )
    view = graph.effective_graph()
    edges = view.get_edges()
    assert len(edges) == 2
    assert [
        (e.get_source(), e.get_destination()) for e in edges
    ] == [("a", "b"), ("b", "a")]


def test_edges_declared_inside_subgraph_are_collected_in_its_view() -> None:
    graph = parse(
        """
        digraph G {
          subgraph cluster_0 {
            a -> b;
          }
          b -> c;
        }
        """
    )
    view = graph.effective_graph()
    cluster = view.get_subgraphs()[0]
    assert [
        (e.get_source(), e.get_destination()) for e in cluster.get_edges()
    ] == [("a", "b")]
    assert [
        (e.get_source(), e.get_destination()) for e in view.get_edges()
    ] == [("a", "b"), ("b", "c")]


def test_port_endpoints_kept_on_edges_but_stripped_for_node_identity() -> None:
    graph = parse("digraph G { p -> q:head; }")
    view = graph.effective_graph()
    edge = view.get_edges()[0]
    assert edge.get_destination() == "q:head"
    assert [n.get_name() for n in view.get_nodes()] == ["p", "q"]


def test_subview_get_node_is_scoped_to_membership() -> None:
    graph = parse(
        """
        digraph G {
          a;
          subgraph cluster_0 {
            b;
          }
        }
        """
    )
    view = graph.effective_graph()
    cluster = view.get_subgraphs()[0]
    assert [n.get_name() for n in cluster.get_node("b")] == ["b"]
    assert cluster.get_node("a") == []
    assert [n.get_name() for n in view.get_node("a")] == ["a"]


def test_get_edge_filters_by_endpoint_pair() -> None:
    graph = parse(
        """
        graph G {
          a -- b;
          b -- c;
        }
        """
    )
    view = graph.effective_graph()
    assert len(view.get_edge("a", "b")) == 1
    # Undirected: order of the query does not matter.
    assert len(view.get_edge("b", "a")) == 1
    assert view.get_edge("a", "c") == []


def test_original_declaration_tree_is_preserved() -> None:
    source = """
        digraph G {
          node [shape=box];
          a -> b;
        }
        """
    graph = parse(source)
    before = graph.to_string()
    view = graph.effective_graph()
    after = graph.to_string()

    assert before == after
    # Implicit nodes live in the view only.
    assert "b" not in graph.obj_dict["nodes"]
    assert "b" in {n.get_name() for n in view.get_nodes()}
    # Default pseudo-nodes remain in the raw tree.
    assert "node" in graph.obj_dict["nodes"]


def test_repeated_calls_are_consistent() -> None:
    graph = parse("digraph G { node [shape=box]; a -> b; }")
    first = graph.effective_graph()
    second = graph.effective_graph()
    assert attrs(first.get_nodes()) == attrs(second.get_nodes())
    assert attrs(first.get_edges()) == attrs(second.get_edges())
    assert first is not second


def test_effective_graph_rejects_non_graph() -> None:
    with pytest.raises(TypeError):
        effective_graph("digraph { a }")  # type: ignore[arg-type]
