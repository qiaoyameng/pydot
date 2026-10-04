# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit testing of escString label interpretation."""

from __future__ import annotations

import pydot
from pydot import LineBreak, interpret_esc_string


def test_node_name_placeholder() -> None:
    node = pydot.Node("mynode", label=r"\N")
    assert interpret_esc_string(node.get("label"), node) == ["mynode"]


def test_double_backslash_is_not_a_placeholder() -> None:
    node = pydot.Node("mynode", label=r"\\N")
    assert interpret_esc_string(node.get("label"), node) == [r"\N"]


def test_line_breaks_with_alignment() -> None:
    fragments = interpret_esc_string(r"a\nb\lc\rd")
    assert fragments == [
        "a",
        LineBreak("center"),
        "b",
        LineBreak("left"),
        "c",
        LineBreak("right"),
        "d",
    ]
    alignments = [
        f.alignment for f in fragments if isinstance(f, LineBreak)
    ]
    assert alignments == ["center", "left", "right"]


def test_graph_name_placeholder() -> None:
    graph = pydot.Dot(graph_name="MyGraph")
    node = pydot.Node("n", label=r"\G")
    graph.add_node(node)
    assert interpret_esc_string(node.get("label"), node) == ["MyGraph"]
    graph.set("label", r"graph: \G")
    assert interpret_esc_string(graph.get("label"), graph) == [
        "graph: MyGraph"
    ]


def test_anonymous_graph_name() -> None:
    graph = pydot.graph_from_dot_data('digraph { a [label="\\G"] }')[0]
    node = graph.get_node("a")[0]
    assert interpret_esc_string(node.get("label"), node) == [""]


def test_edge_endpoints_with_ports() -> None:
    graph = pydot.Dot(graph_type="digraph")
    edge = pydot.Edge("a:p0", "b:p1", label=r"\T -> \H")
    graph.add_edge(edge)
    assert interpret_esc_string(edge.get("label"), edge) == [
        "a:p0 -> b:p1"
    ]


def test_edge_name_placeholder() -> None:
    digraph = pydot.Dot(graph_type="digraph")
    edge = pydot.Edge("a:p0", "b:p1", label=r"\E")
    digraph.add_edge(edge)
    assert interpret_esc_string(edge.get("label"), edge) == ["a:p0->b:p1"]

    graph = pydot.Graph(graph_type="graph")
    edge = pydot.Edge("a", "b", label=r"\E")
    graph.add_edge(edge)
    assert interpret_esc_string(edge.get("label"), edge) == ["a--b"]


def test_replacement_text_is_not_reinterpreted() -> None:
    graph = pydot.Dot(graph_name=r"g\Gl")
    node = pydot.Node(r"we\Nird", label=r"\N in \G")
    graph.add_node(node)
    assert interpret_esc_string(node.get("label"), node) == [
        r"we\Nird in g\Gl"
    ]


def test_quoted_names_are_unquoted_in_fragments() -> None:
    graph = pydot.graph_from_dot_data(
        r'digraph "my graph" { "we\Nird" [label="\N of \G"] }'
    )[0]
    node = graph.get_nodes()[0]
    assert interpret_esc_string(node.get("label"), node) == [
        r"we\Nird of my graph"
    ]


def test_html_like_label_is_verbatim() -> None:
    node = pydot.Node("n", label=r"<<i>\N</i>>")
    assert interpret_esc_string(node.get("label"), node) == [r"<<i>\N</i>>"]


def test_record_label_is_verbatim() -> None:
    node = pydot.Node("n", label=r"{<p0> \N | right\nside}")
    assert interpret_esc_string(node.get("label"), node) == [
        r"{<p0> \N | right\nside}"
    ]


def test_record_shape_label_is_verbatim() -> None:
    node = pydot.Node("n", shape="record", label=r"a|b\N")
    assert interpret_esc_string(node.get("label"), node) == [r"a|b\N"]


def test_quoted_label_from_parsed_dot() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { a [label="\\N of \\G"] }'
    )[0]
    node = graph.get_node("a")[0]
    assert node.get("label") == r'"\N of \G"'
    assert interpret_esc_string(node.get("label"), node) == ["a of G"]


def test_placeholder_without_context_is_kept_literally() -> None:
    edge = pydot.Edge("a", "b", label=r"\N")
    assert interpret_esc_string(edge.get("label"), edge) == [r"\N"]
    node = pydot.Node("n", label=r"\T")
    assert interpret_esc_string(node.get("label"), node) == [r"\T"]
    assert interpret_esc_string(r"\N") == [r"\N"]


def test_unknown_escape_drops_backslash() -> None:
    assert interpret_esc_string(r'say \"hi\"') == ['say "hi"']


def test_common_interpret_esc_label() -> None:
    node = pydot.Node("n", label=r"x\l")
    assert node.interpret_esc_label() == ["x", LineBreak("left")]
    assert pydot.Node("n").interpret_esc_label() == []


def test_interpretation_preserves_original_label() -> None:
    graph = pydot.Dot(graph_name="G")
    node = pydot.Node("n", label=r"\N\n\G")
    graph.add_node(node)
    before = graph.to_string()
    interpret_esc_string(node.get("label"), node)
    node.interpret_esc_label()
    assert node.get("label") == r"\N\n\G"
    assert graph.to_string() == before
    assert r'label="\N\n\G"' in graph.to_string()
