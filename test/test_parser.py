# SPDX-FileCopyrightText: 2025 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit testing of individual dot_parser classes."""

from __future__ import annotations

import textwrap

import pyparsing as pp
import pytest

from pydot import dot_parser
from pydot.core import EdgeEndpoint
from pydot.dot_parser import HTML, GraphParser


def test_HTML_valid() -> None:
    """Test successful HTML parses."""
    parsed = HTML().parse_string("<<b>Bold Text</b>>")
    assert isinstance(parsed, pp.ParseResults)
    assert list(parsed) == ["<<b>Bold Text</b>>"]

    parsed2 = HTML().parse_string("<Label #1>")
    assert isinstance(parsed2, pp.ParseResults)
    assert list(parsed2) == ["<Label #1>"]


def test_HTML_invalid() -> None:
    """Test HTML parsing failure."""
    with pytest.raises(pp.ParseException) as exc:
        HTML().parse_string("<<b>Unbalanced tags</b>")
    assert "HTML: expected '>' to match '<' on line 1" in str(exc.value)


def test_edge_subgraph_anon() -> None:
    """Test parsing of an edge with an anonymous subgraph endpoint."""
    parser = GraphParser.edge_stmt
    res = parser.parse_string("""a -- { b; c; }""")
    assert len(res) == 1
    edge = res[0]
    expected = textwrap.dedent("""
        a -- {
        b;
        c;
        };""").strip()
    assert edge.to_string() == expected


def test_edge_subgraph_explicit() -> None:
    """Test parsing of an edge with an explicit subgraph endpoint."""
    parser = GraphParser.edge_stmt
    res = parser.parse_string("""a -- subgraph XY { b; c; }""")
    assert len(res) == 1
    edge = res[0]
    expected = textwrap.dedent("""
        a -- subgraph XY {
        b;
        c;
        };""").strip()
    assert edge.to_string() == expected


def test_AttrList_repr() -> None:
    parser = GraphParser.attr_list("a_list")
    res = parser.parse_string("[color=red, shape=square]")
    assert isinstance(res, pp.ParseResults)
    a_list = res.a_list
    assert isinstance(a_list, pp.ParseResults)
    assert len(a_list) == 1
    repr_str = repr(a_list[0])
    assert repr_str == "P_AttrList({'color': 'red', 'shape': 'square'})"


def test_DefaultStatement_repr() -> None:
    parser = GraphParser.default_stmt("defaults")
    res = parser.parse_string("node [color=blue];")
    assert isinstance(res, pp.ParseResults)
    defaults = res.defaults
    repr_str = repr(defaults)
    assert repr_str == "DefaultStatement(node, {'color': 'blue'})"


def test_keyword_case_insensitive_graph_type() -> None:
    """Graph type keywords are case-insensitive."""
    for src, expected_type in [
        ("DIGRAPH G { a -> b }", "digraph"),
        ("GRAPH G { a -- b }", "graph"),
        ("Digraph G { a -> b }", "digraph"),
        ("Graph G { a -- b }", "graph"),
    ]:
        (g,) = dot_parser.parse_dot_data(src)
        assert g.get_type() == expected_type, src


def test_keyword_case_insensitive_strict() -> None:
    """STRICT keyword is case-insensitive."""
    for src in [
        "STRICT DIGRAPH G { a -> b }",
        "STRICT GRAPH G { a -- b }",
        "Strict Digraph G { a -> b }",
        "strict GRAPH G { a -- b }",
    ]:
        (g,) = dot_parser.parse_dot_data(src)
        assert g.get_strict(), src


def test_keyword_case_insensitive_node_edge_defaults() -> None:
    """NODE and EDGE default-attribute statements are case-insensitive."""
    for node_kw, edge_kw in [
        ("NODE", "EDGE"),
        ("Node", "Edge"),
        ("node", "edge"),
    ]:
        src = (
            f"digraph G {{ {node_kw} [color=red]; {edge_kw}"
            " [style=dashed]; a -> b }}"
        )
        (g,) = dot_parser.parse_dot_data(src)
        assert g.get_node_defaults() == [{"color": "red"}], src
        assert g.get_edge_defaults() == [{"style": "dashed"}], src


def test_keyword_case_insensitive_subgraph() -> None:
    """SUBGRAPH keyword is case-insensitive."""
    for kw in ["subgraph", "SUBGRAPH", "Subgraph"]:
        src = f"digraph G {{ {kw} SUB {{ a; b; }} }}"
        (g,) = dot_parser.parse_dot_data(src)
        subs = g.get_subgraphs()
        assert len(subs) == 1, src
        assert subs[0].get_name() == "SUB", src


def test_strict_graph_parsing() -> None:
    res = dot_parser.parse_dot_data("strict graph G { a; b; }")
    assert isinstance(res, list)
    assert len(res) == 1
    graph = res[0]
    assert graph.get_strict()
    assert graph.to_string() == "strict graph G {\na;\nb;\n}\n"

    res2 = dot_parser.parse_dot_data(
        """
        graph G { a; b; }
        strict digraph H { c; d; }
        """
    )
    assert isinstance(res2, list)
    assert len(res2) == 2
    assert not res2[0].get_strict()
    assert res2[1].get_strict()


def test_backslash_continuations() -> None:
    src = textwrap.dedent(r"""
        graph G {
            "my very long node \
        name" [color=red];
            "my indented and wrapped \
            node name" [shape=square];
            "my node name containing \ backslash";
        }""")
    res = dot_parser.parse_dot_data(src)
    assert isinstance(res, list)
    assert len(res) == 1
    graph = res[0]
    nodes = graph.get_nodes()
    assert len(nodes) == 3
    assert nodes[0].get_name() == '"my very long node name"'
    assert nodes[1].get_name() == '"my indented and wrapped     node name"'
    assert nodes[2].get_name() == '"my node name containing \\ backslash"'


def test_plus_concatenation() -> None:
    src = textwrap.dedent(r"""
        digraph G {
            "my" + "concatenated" + "name";
            "myconcatenated" + " with " + "ports" [color="r" + "ed"];
            "my\
        concatenated" + " with ports":p45:sw -> "my\
        concatenated\
        name":ne [penwidth=5, arrows="b" + "o" + "t" + "h"];
        "con" + "catenated" [label="this is a long\
        long label" + " that just goes on \
        and on and on"];
        }""")
    res = dot_parser.parse_dot_data(src)
    assert isinstance(res, list)
    assert len(res) == 1
    graph = res[0]
    nodes = graph.get_nodes()
    edges = graph.get_edges()

    assert len(nodes) == 3
    assert nodes[0].get_name() == '"myconcatenatedname"'
    assert nodes[1].get_name() == '"myconcatenated with ports"'
    assert nodes[1].get("color") == '"red"'
    assert nodes[2].get_name() == '"concatenated"'
    assert nodes[2].get("label") == (
        '"this is a longlong label that just goes on and on and on"'
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.get_source() == f"{nodes[1].get_name()}:p45:sw"
    assert edge.get_destination().node_id == "myconcatenatedname"
    assert edge.get_destination().port is None
    assert edge.get_destination().compass == "ne"
    assert edge.get_destination() == '"myconcatenatedname":ne'
    assert edge.get("arrows") == '"both"'


def test_structured_edge_endpoint_roundtrip() -> None:
    src = textwrap.dedent(r'''
        digraph G {
            "api:v2";
            "api:v2":v2:n -> x::ne;
            y:"ne" -> z:"a\:b";
        }''')
    (graph,) = dot_parser.parse_dot_data(src)
    node = graph.get_node('"api:v2"')[0]
    edge_a, edge_b = graph.get_edges()

    assert node.get_name() == '"api:v2"'
    assert edge_a.get_source().node_id == "api:v2"
    assert edge_a.get_source().port == "v2"
    assert edge_a.get_source().compass == "n"
    assert edge_a.get_destination().node_id == "x"
    assert edge_a.get_destination().port == ""
    assert edge_a.get_destination().compass == "ne"
    assert edge_b.get_source().node_id == "y"
    assert edge_b.get_source().port == "ne"
    assert edge_b.get_source().compass is None
    assert edge_b.get_destination().node_id == "z"
    assert edge_b.get_destination().port == "a\\:b"

    (reparsed,) = dot_parser.parse_dot_data(graph.to_string())
    reparsed_edges = reparsed.get_edges()
    assert reparsed_edges[0].get_source() == edge_a.get_source()
    assert reparsed_edges[0].get_destination() == edge_a.get_destination()


def test_colon_node_name_is_distinct_from_port_reference() -> None:
    src = (
        'digraph G { "api:v2"; "api:v2" -> z; '
        'api:v2 -> "api:v2":v2:ne; }'
    )
    (graph,) = dot_parser.parse_dot_data(src)

    assert graph.get_node('"api:v2"')[0].get_name() == '"api:v2"'
    node_endpoint, port_reference = graph.get_edges()
    assert node_endpoint.get_source().node_id == "api:v2"
    assert node_endpoint.get_source().port is None
    assert node_endpoint.get_source().compass is None
    assert port_reference.get_source().node_id == "api"
    assert port_reference.get_source().port == "v2"
    assert port_reference.get_destination().node_id == "api:v2"
    assert port_reference.get_destination().port == "v2"
    assert port_reference.get_destination().compass == "ne"

    (reparsed,) = dot_parser.parse_dot_data(graph.to_string())
    reparsed_edges = reparsed.get_edges()
    assert reparsed_edges[0].get_source() == node_endpoint.get_source()
    assert reparsed_edges[1].get_source() == port_reference.get_source()
    assert (
        reparsed_edges[1].get_destination()
        == port_reference.get_destination()
    )


def test_compass_directions_are_case_sensitive() -> None:
    with pytest.raises(pp.ParseException):
        GraphParser.edge_endpoint.parse_string("a:b:NE", parse_all=True)

    with pytest.raises(pp.ParseException):
        GraphParser.edge_endpoint.parse_string("a:b:north", parse_all=True)


def test_legacy_endpoint_strings_distinguish_compass_and_port() -> None:
    compass = EdgeEndpoint.from_string("n:ne")
    port = EdgeEndpoint.from_string('n:"ne"')
    plain_n = EdgeEndpoint.from_string("n")
    plain_ne = EdgeEndpoint.from_string("ne")

    assert (plain_n.node_id, plain_n.port, plain_n.compass) == (
        "n",
        None,
        None,
    )
    assert (plain_ne.node_id, plain_ne.port, plain_ne.compass) == (
        "ne",
        None,
        None,
    )
    assert (compass.node_id, compass.port, compass.compass) == (
        "n",
        None,
        "ne",
    )
    assert (port.node_id, port.port, port.compass) == (
        "n",
        "ne",
        None,
    )

    with pytest.raises(ValueError, match="empty port"):
        EdgeEndpoint("a", "")


def test_bad_parse_brace(capsys) -> None:
    src = textwrap.dedent(r"""
        graph G {
            a -- b;
            {{[{foo:bar:baz}]}}};
        }""")
    expected = textwrap.dedent("""
        {{[{foo:bar:baz}]}}};
        ^
    Expected '}', found '{'  (at char 27), (line:4, col:5)
    """).strip()
    res = dot_parser.parse_dot_data(src)
    assert res is None
    captured = capsys.readouterr()
    assert captured.out.strip() == expected


def test_bad_parse_bracket(capsys) -> None:
    src = textwrap.dedent(r"""
        graph G {
            a -- b;
            node [shape=box;
        }""")
    expected = textwrap.dedent("""
        node [shape=box;
             ^
    Expected '}', found '['  (at char 32), (line:4, col:10)
    """).strip()
    res = dot_parser.parse_dot_data(src)
    assert res is None
    captured = capsys.readouterr()
    assert captured.out.strip() == expected


def test_comments_are_discarded() -> None:
    """Test that all three comment forms are ignored.

    The DOT language accepts C-style block comments, C++-style line
    comments, and lines beginning with '#', which are discarded as
    preprocessor output.
    """
    expected = "digraph {\na -> b;\n}\n"

    for source in [
        "digraph { /* block */ a -> b }",
        "digraph {\n/* spanning\n   two lines */\na -> b\n}",
        "digraph { a -> b // trailing\n}",
        '# 1 "preprocessed.dot"\ndigraph { a -> b }',
        "digraph {\n# a line comment\na -> b\n}",
    ]:
        (g,) = dot_parser.parse_dot_data(source)
        assert g.to_string() == expected, f"differs for: {source!r}"
