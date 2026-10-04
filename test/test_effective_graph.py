# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import textwrap

import pydot
from pydot import dot_parser


def attributes(objects: list[pydot.Common]) -> dict[str, dict[str, object]]:
    return {
        object_.get_name(): object_.get_attributes()
        for object_ in objects
    }


def test_effective_graph_default_scopes_and_cross_subgraph_nodes() -> None:
    src = textwrap.dedent(
        """
        digraph G {
            graph [bgcolor=white];
            node [color=red];
            a;

            node [shape=box];
            b;

            edge [color=blue];
            a -> b;

            edge [style=dashed];
            a -> c;

            subgraph S {
                graph [label="inside"];
                node [color=green, shape=circle];
                a [penwidth=2];
                d;

                edge [color=black];
                a -> d;
            }

            e;
        }
        """
    )
    raw = dot_parser.parse_dot_data(src)[0]
    raw_dot = raw.to_string()
    effective = raw.get_effective_graph()

    assert effective is not raw
    assert raw.to_string() == raw_dot
    assert raw.get_node("a")[0].get_attributes() == {}
    assert raw.get_node_defaults() == [
        {"color": "red"},
        {"shape": "box"},
    ]
    assert effective.get_attributes() == {"bgcolor": "white"}
    assert effective.get_subgraph("S")[0].get_attributes() == {
        "label": '"inside"'
    }

    assert attributes(effective.get_nodes()) == {
        "a": {"color": "red", "penwidth": "2"},
        "b": {"color": "red", "shape": "box"},
        "c": {"color": "red", "shape": "box"},
        "d": {"color": "green", "shape": "circle"},
        "e": {"color": "red", "shape": "box"},
    }
    assert [node.get_name() for node in effective.get_nodes()] == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    assert [
        (
            edge.get_source(),
            edge.get_destination(),
            edge.get_attributes(),
        )
        for edge in effective.get_edges()
    ] == [
        ("a", "b", {"color": "blue"}),
        ("a", "c", {"color": "blue", "style": "dashed"}),
        (
            "a",
            "d",
            {"color": "black", "style": "dashed"},
        ),
    ]

    subgraph = effective.get_subgraph("S")[0]
    assert attributes(subgraph.get_nodes()) == {
        "a": {"color": "red", "penwidth": "2"},
        "d": {"color": "green", "shape": "circle"},
    }

    assert {node.get_name() for node in raw.get_node("a")} == {"a"}


def test_effective_graph_expands_subgraph_endpoints() -> None:
    src = textwrap.dedent(
        """
        digraph G {
            node [color=red];
            x -> subgraph S {
                a;
                b;
            } [color=blue];
        }
        """
    )
    effective = dot_parser.parse_dot_data(src)[0].effective_graph

    assert [node.get_name() for node in effective.get_nodes()] == [
        "x",
        "a",
        "b",
    ]
    assert all(
        node.get_attributes() == {"color": "red"}
        for node in effective.get_nodes()
    )
    assert [
        (edge.get_source(), edge.get_destination())
        for edge in effective.get_edges()
    ] == [("x", "a"), ("x", "b")]
    assert all(
        edge.get_attributes() == {"color": "blue"}
        for edge in effective.get_edges()
    )


def test_effective_strict_graph_merges_reverse_undirected_edge() -> None:
    src = textwrap.dedent(
        """
        strict graph G {
            edge [color=red];
            a -- b [style=solid];

            edge [color=green];
            b -- a [penwidth=2];
        }
        """
    )
    effective = dot_parser.parse_dot_data(src)[0].effective_graph

    assert len(effective.get_edges()) == 1
    edge = effective.get_edges()[0]
    assert (edge.get_source(), edge.get_destination()) == ("a", "b")
    assert edge.get_attributes() == {
        "color": "red",
        "style": "solid",
        "penwidth": "2",
    }


def test_effective_normal_graph_keeps_parallel_reverse_edge() -> None:
    src = textwrap.dedent(
        """
        graph G {
            edge [color=red];
            a -- b [style=solid];

            edge [color=green];
            b -- a [penwidth=2];
        }
        """
    )
    effective = dot_parser.parse_dot_data(src)[0].effective_graph

    assert [
        (
            edge.get_source(),
            edge.get_destination(),
            edge.get_attributes(),
        )
        for edge in effective.get_edges()
    ] == [
        ("a", "b", {"color": "red", "style": "solid"}),
        ("b", "a", {"color": "green", "penwidth": "2"}),
    ]


def test_effective_strict_graph_merges_expanded_edge_combinations() -> None:
    src = textwrap.dedent(
        """
        strict digraph G {
            {a b} -> {c d} [color=red];
            {a b} -> {c d} [penwidth=2];
        }
        """
    )
    effective = dot_parser.parse_dot_data(src)[0].effective_graph

    assert [
        (
            edge.get_source(),
            edge.get_destination(),
            edge.get_attributes(),
        )
        for edge in effective.get_edges()
    ] == [
        ("a", "c", {"color": "red", "penwidth": "2"}),
        ("a", "d", {"color": "red", "penwidth": "2"}),
        ("b", "c", {"color": "red", "penwidth": "2"}),
        ("b", "d", {"color": "red", "penwidth": "2"}),
    ]
