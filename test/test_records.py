# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit tests for record/Mrecord label parsing, building and checks."""

from __future__ import annotations

import pytest

import pydot
from pydot.records import (
    DuplicatePortError,
    RecordError,
    RecordField,
    RecordLabel,
    RecordSyntaxError,
    UnknownPortError,
    check_graph_records,
    get_node_record,
    is_record_node,
)


def _field_signature(label: RecordLabel) -> list:
    """Structural, whitespace-independent snapshot of a field tree."""

    def walk(fields):
        result = []
        for field in fields:
            if field.is_group:
                result.append(
                    ("group", field.port, field.text, walk(field.fields or []))
                )
            else:
                result.append(("leaf", field.port, field.text))
        return result

    return walk(label.fields)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_plain_fields() -> None:
    label = RecordLabel.parse("a|b|c")
    assert _field_signature(label) == [
        ("leaf", None, "a"),
        ("leaf", None, "b"),
        ("leaf", None, "c"),
    ]
    assert label.ports() == set()


def test_parse_port_declaration() -> None:
    label = RecordLabel.parse("<id> identifier|<name> full name")
    assert label.ports() == {"id", "name"}
    id_field = label.find("id")
    assert id_field is not None
    assert id_field.text == "identifier"
    assert label.find("name").text == "full name"


def test_parse_nested_groups() -> None:
    label = RecordLabel.parse("{<p1> id|name|{x|y}}|tail")
    assert len(label.fields) == 2
    group = label.fields[0]
    assert group.is_group
    id_field = group.fields[0]
    assert id_field.port == "p1" and id_field.text == "id"
    assert [f.text for f in group.fields if not f.is_group][1:2] == [
        "name"
    ]
    inner = group.fields[2]
    assert inner.is_group
    assert [child.text for child in inner.fields] == ["x", "y"]
    assert label.fields[1].text == "tail"
    assert label.ports() == {"p1"}


def test_parse_keeps_escaped_text() -> None:
    label = RecordLabel.parse(r"name\|x|a\{b\}|c\<d\>|sp\ ace")
    texts = [field.text for field in label.fields]
    assert texts == ["name|x", "a{b}", "c<d>", "sp ace"]


def test_parse_quoted_special_characters() -> None:
    label = RecordLabel.parse('he said "a|b{c}" done')
    assert len(label.fields) == 1
    assert label.fields[0].text == 'he said a|b{c} done'


def test_parse_empty_fields() -> None:
    assert [f.text for f in RecordLabel.parse("a||b").fields] == [
        "a", "", "b"
    ]
    assert [f.text for f in RecordLabel.parse("|a").fields] == ["", "a"]
    assert [f.text for f in RecordLabel.parse("a|").fields] == ["a", ""]
    assert [f.text for f in RecordLabel.parse("|").fields] == ["", ""]
    assert RecordLabel.parse("").fields == []


def test_parse_empty_group() -> None:
    label = RecordLabel.parse("{}")
    assert len(label.fields) == 1
    assert label.fields[0].is_group
    assert label.fields[0].fields == []
    label = RecordLabel.parse("{|}")
    assert [f.text for f in label.fields[0].fields] == ["", ""]


def test_parse_ignores_unescaped_whitespace() -> None:
    label = RecordLabel.parse("  a  |\t b \n | c ")
    assert [f.text for f in label.fields] == ["a", "b", "c"]


def test_parse_quoted_dot_label() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { n [shape=record, label="{<p> a|b\\|c}"]; }'
    )[0]
    node = graph.get_node("n")[0]
    label = node.get_record_label()
    assert label is not None
    assert label.ports() == {"p"}
    group = label.fields[0]
    assert group.is_group
    assert [f.text for f in group.fields] == ["a", "b|c"]


@pytest.mark.parametrize(
    "bad",
    [
        "{a|b",
        "a}",
        "<p",
        "a|{b",
        "<>",
        "<p>x<a>",
        '"abc',
        "a\\",
    ],
)
def test_parse_syntax_errors(bad: str) -> None:
    with pytest.raises(RecordSyntaxError):
        RecordLabel.parse(bad)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def test_build_from_strings_and_tuples() -> None:
    label = RecordLabel.build(
        [
            "id",
            ("<p2>", "name|x"),
            ["left", "right", ("<p3>", "deep")],
        ]
    )
    assert label.serialize() == r"id|<p2> name\|x|{left|right|<p3> deep}"
    assert label.ports() == {"p2", "p3"}


def test_build_group_with_port_and_text() -> None:
    label = RecordLabel.build([("<gp>", "header", ["a", "b"])])
    assert label.serialize() == "<gp>header{a|b}"
    group = label.fields[0]
    assert group.is_group and group.port == "gp" and group.text == "header"


def test_build_dict_specs() -> None:
    label = RecordLabel.build(
        [{"port": "p", "text": "t", "fields": ["a", "b"]}]
    )
    assert label.serialize() == "<p>t{a|b}"


def test_build_empty_fields() -> None:
    label = RecordLabel.build(["a", None, "b"])
    assert label.serialize() == "a||b"


def test_add_field() -> None:
    label = RecordLabel()
    group = label.add_field(port="gp", fields=["x", "y"])
    group.fields.append(RecordField(text="z", port="zp"))
    label.add_field(text="tail")
    assert label.serialize() == "<gp>{x|y|<zp> z}|tail"
    assert label.ports() == {"gp", "zp"}


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

COMPLEX_LABEL = (
    r"{<id> id|<name> name\|with bar|"
    r"{<c1> child1|empty after||<c2> c2}|<esc> a\{b\}\|c}"
)


@pytest.mark.parametrize(
    "raw",
    [
        "a|b|c",
        "<p1> one|<p2> two|{x|y|<p3> z}",
        COMPLEX_LABEL,
        "|a||b|",
        "{}",
        "{|}|{}",
        r"\|\{\}\<\>\ \ ",
        'just "a|b" text',
    ],
)
def test_build_serialize_parse_roundtrip(raw: str) -> None:
    parsed = RecordLabel.parse(raw)
    serialized = parsed.serialize()
    reparsed = RecordLabel.parse(serialized)

    assert _field_signature(parsed) == _field_signature(reparsed)
    assert parsed.ports() == reparsed.ports()


def test_complex_example_explicit() -> None:
    parsed = RecordLabel.parse(COMPLEX_LABEL)
    reparsed = RecordLabel.parse(parsed.serialize())
    assert _field_signature(parsed) == _field_signature(reparsed)
    assert parsed.ports() == {"id", "name", "c1", "c2", "esc"}
    assert reparsed.ports() == parsed.ports()

    rebuilt = RecordLabel.build(
        [
            [
                ("<id>", "id"),
                ("<name>", "name|with bar"),
                [
                    ("<c1>", "child1"),
                    "empty after",
                    None,
                    ("<c2>", "c2"),
                ],
                ("<esc>", "a{b}|c"),
            ]
        ]
    )
    rebuilt_parsed = RecordLabel.parse(rebuilt.serialize())
    assert rebuilt_parsed.ports() == parsed.ports()
    assert _field_signature(rebuilt_parsed) == _field_signature(parsed)


# ---------------------------------------------------------------------------
# rankdir layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rankdir,expected",
    [
        ("TB", ["TB", "LR", "TB", "LR"]),
        ("LR", ["LR", "TB", "LR", "TB"]),
        ("BT", ["BT", "RL", "BT", "RL"]),
        ("RL", ["RL", "BT", "RL", "BT"]),
        ("bogus", ["TB", "LR", "TB", "LR"]),
    ],
)
def test_level_directions(rankdir: str, expected: list[str]) -> None:
    label = RecordLabel.parse("a|{b|{c|{d|e}}}")
    assert label.level_directions(rankdir) == expected


def test_single_outer_group_is_layout_transparent() -> None:
    outer = RecordLabel.parse("{a|{b|c}}")
    plain = RecordLabel.parse("a|{b|c}")
    assert outer.level_directions("TB") == plain.level_directions("TB")
    assert outer.level_directions("TB") == ["TB", "LR"]
    directions = outer.field_directions("TB")
    assert directions[()] == "TB"
    assert directions[("[0]",)] == "TB"
    assert directions[("[0]", "[1]")] == "LR"


def test_field_directions_paths() -> None:
    label = RecordLabel.parse("a|{b|c}|{d|{e|f}}")
    directions = label.field_directions("LR")
    assert directions[()] == "LR"
    assert directions[("[1]",)] == "TB"
    assert directions[("[2]",)] == "TB"
    assert directions[("[2]", "[1]")] == "LR"


# ---------------------------------------------------------------------------
# Duplicate ports
# ---------------------------------------------------------------------------

def test_duplicate_port_error_names_path() -> None:
    label = RecordLabel.parse("<p> a|<p> b")
    with pytest.raises(DuplicatePortError) as excinfo:
        label.check_duplicate_ports("node42")
    error = excinfo.value
    assert error.node == "node42"
    assert error.port == "p"
    assert error.path == ("[1]",)
    assert "[1]" in str(error)
    assert "node42" in str(error)


def test_duplicate_port_in_nested_group() -> None:
    label = RecordLabel.parse("{a|<q> b}|{c|<q> d}")
    with pytest.raises(DuplicatePortError) as excinfo:
        label.check_duplicate_ports("n")
    assert excinfo.value.path == ("[1]", "[1]")


def test_duplicate_port_on_group_field() -> None:
    label = RecordLabel.parse("<g>{a|b}|<g>c")
    with pytest.raises(DuplicatePortError):
        label.check_duplicate_ports("n")


# ---------------------------------------------------------------------------
# Graph-level checks
# ---------------------------------------------------------------------------

def _graph_with_record() -> pydot.Dot:
    graph = pydot.Dot(rankdir="LR")
    graph.set_node_defaults(shape="record")
    graph.add_node(
        pydot.Node(
            "t",
            label="{<id> id|<name> name|{<inner> a|b}}",
        )
    )
    graph.add_node(pydot.Node("u", shape="box"))
    return graph


def test_check_only_applies_to_record_shapes() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("plain", shape="box", label="a|b|{unclosed"))
    graph.add_node(pydot.Node("html", shape="record", label="<<b>x</b>>"))
    # No exception: plain labels and HTML labels are not record syntax.
    check_graph_records(graph)
    assert get_node_record(graph.get_node("plain")[0]) is None
    assert get_node_record(graph.get_node("html")[0]) is None


def test_node_is_record() -> None:
    graph = pydot.Dot()
    graph.set_node_defaults(shape="Mrecord")
    graph.add_node(pydot.Node("r", label="a|b"))
    graph.add_node(pydot.Node("b", shape="ellipse", label="a|b"))
    assert is_record_node(graph.get_node("r")[0])
    assert graph.get_node("r")[0].is_record()
    assert not graph.get_node("b")[0].is_record()


def test_valid_edge_references_pass() -> None:
    graph = _graph_with_record()
    graph.add_edge(pydot.Edge("t:id", "u"))
    graph.add_edge(pydot.Edge("u", "t:name:e"))
    graph.add_edge(pydot.Edge("t:inner:nw", "u"))
    check_graph_records(graph)
    graph.check_records()


def test_edge_to_unknown_port_raises() -> None:
    graph = _graph_with_record()
    graph.add_edge(pydot.Edge("t:ghost", "u"))
    with pytest.raises(UnknownPortError) as excinfo:
        check_graph_records(graph)
    error = excinfo.value
    assert error.node == "t"
    assert error.port == "ghost"


def test_edge_to_nested_port() -> None:
    graph = _graph_with_record()
    graph.add_edge(pydot.Edge("t:inner", "u"))
    check_graph_records(graph)


def test_duplicate_port_in_graph_raises() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("t", shape="record", label="<p> a|<p> b"))
    with pytest.raises(DuplicatePortError) as excinfo:
        check_graph_records(graph)
    assert excinfo.value.path == ("[1]",)


def test_malformed_record_label_in_graph_raises() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("t", shape="record", label="{a|b"))
    with pytest.raises(RecordSyntaxError):
        check_graph_records(graph)


def test_check_from_dot_data() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { rankdir=LR; node [shape=record]; '
        'a [label="{<p1> id|x}"]; b; a:p1 -> b; }'
    )[0]
    check_graph_records(graph)
    assert graph.get_node("a")[0].get_record_label().level_directions(
        graph.get_rankdir()
    ) == ["LR"]


def test_non_record_node_with_pipe_label_unchanged() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("n", label="a|b"))
    assert get_node_record(graph.get_node("n")[0]) is None
    # An edge ":port" on a non-record node is not checked either.
    graph.add_node(pydot.Node("m"))
    graph.add_edge(pydot.Edge("n:whatever", "m"))
    check_graph_records(graph)


def test_record_error_base_class() -> None:
    with pytest.raises(RecordError):
        RecordLabel.build([("too", "many", "plain", "items")])


def test_check_recurses_into_subgraphs_with_defaults() -> None:
    graph = pydot.Dot()
    sub = pydot.Subgraph("cluster_0")
    sub.set_node_defaults(shape="record")
    sub.add_node(pydot.Node("t", label="<p> a|b"))
    graph.add_subgraph(sub)
    graph.add_node(pydot.Node("u"))
    graph.add_edge(pydot.Edge("t:p", "u"))
    check_graph_records(graph)


def test_check_subgraph_unknown_port() -> None:
    graph = pydot.Dot()
    sub = pydot.Subgraph("cluster_0")
    sub.set_node_defaults(shape="record")
    sub.add_node(pydot.Node("t", label="<p> a|b"))
    graph.add_subgraph(sub)
    graph.add_node(pydot.Node("u"))
    graph.add_edge(pydot.Edge("t:missing", "u"))
    with pytest.raises(UnknownPortError):
        check_graph_records(graph)


def test_inner_subgraph_default_overrides_outer() -> None:
    graph = pydot.Dot()
    graph.set_node_defaults(shape="record")
    sub = pydot.Subgraph("cluster_0")
    sub.set_node_defaults(shape="box")
    sub.add_node(pydot.Node("t", label="x|y|{unclosed"))
    graph.add_subgraph(sub)
    # Inner shape=box wins, so malformed record syntax is ignored.
    check_graph_records(graph)


def test_quoted_node_name_endpoint() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { "n" [shape=record,label="<p> a"]; u;'
        ' "n":p -> u; }'
    )[0]
    check_graph_records(graph)


def test_quoted_node_name_unknown_port() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { "n" [shape=record,label="<p> a"]; u;'
        ' "n":bad -> u; }'
    )[0]
    with pytest.raises(UnknownPortError) as excinfo:
        check_graph_records(graph)
    assert excinfo.value.node == "n"
    assert excinfo.value.port == "bad"
