# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit testing of graph canonicalization."""

from __future__ import annotations

import pytest

import pydot


def make_original() -> pydot.Dot:
    graph = pydot.Dot("dependency", graph_type="digraph")
    graph.add_node(pydot.Node("alpha", kind="entry", hidden="x"))
    graph.add_node(pydot.Node("beta", kind="worker"))
    graph.add_node(pydot.Node("gamma", kind="worker"))
    graph.add_node(pydot.Node("delta", kind="store"))
    graph.add_edge(
        pydot.Edge("alpha", "beta", relation="calls", ignored="y")
    )
    graph.add_edge(pydot.Edge("beta", "gamma", relation="sync"))
    graph.add_edge(pydot.Edge("gamma", "delta", relation="sync"))
    graph.add_edge(pydot.Edge("delta", "beta", relation="wakes"))
    graph.add_edge(pydot.Edge("beta", "beta", relation="retry"))
    graph.add_edge(pydot.Edge("beta", "gamma", relation="sync"))
    return graph


def make_renamed() -> pydot.Dot:
    graph = pydot.Dot("different-name", graph_type="digraph")
    graph.add_node(pydot.Node("q7", kind="store"))
    graph.add_node(pydot.Node("z3", kind="entry"))
    graph.add_node(pydot.Node("x9", kind="worker", hidden="z"))
    graph.add_node(pydot.Node("m1", kind="worker"))
    graph.add_edge(
        pydot.Edge("z3", "x9", relation="calls", ignored="different")
    )
    graph.add_edge(
        pydot.Edge("x9", "m1", relation="sync", ignored="different")
    )
    graph.add_edge(
        pydot.Edge("m1", "q7", relation="sync", ignored="different")
    )
    graph.add_edge(
        pydot.Edge("q7", "x9", relation="wakes", ignored="different")
    )
    graph.add_edge(
        pydot.Edge("x9", "x9", relation="retry", ignored="different")
    )
    graph.add_edge(
        pydot.Edge("x9", "m1", relation="sync", ignored="different")
    )
    return graph


def selected() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return ("kind",), ("relation",)


def test_canonicalization_is_invariant_to_ids_and_order() -> None:
    node_attrs, edge_attrs = selected()
    first = pydot.canonicalize_graph(
        make_original(), node_attrs, edge_attrs
    )
    second = pydot.canonicalize_graph(
        make_renamed(),
        edge_attributes=edge_attrs,
        node_attributes=node_attrs,
    )

    assert first.complete
    assert second.complete
    assert first.representation == second.representation
    assert first.digest == second.digest
    assert {node["id"] for node in first.representation["nodes"]} == {
        0,
        1,
        2,
        3,
    }
    assert "alpha" not in str(first.representation)
    assert "ignored" not in str(first.representation)
    assert "hidden" not in str(first.representation)


def test_attribute_declaration_order_is_not_semantic() -> None:
    graph = {
        "nodes": [
            {"id": "a", "first": "one", "second": "two"},
            {"id": "b", "first": "one", "second": "two"},
        ],
        "edges": [("a", "b", {"kind": "x", "weight": "y"})],
    }

    first = pydot.canonicalize_graph(
        graph,
        node_attributes=["first", "second"],
        edge_attributes=["kind", "weight"],
    )
    second = pydot.canonicalize_graph(
        graph,
        node_attributes=["second", "first"],
        edge_attributes=["weight", "kind"],
    )

    assert first.complete
    assert first.representation == second.representation
    assert first.digest == second.digest


def test_complex_attribute_values_are_order_independent() -> None:
    first = {
        "nodes": [
            {
                "id": "a",
                "meta": {"b": [1, {"z": 2, "a": 3}], "a": {3, 1}},
            }
        ]
    }
    second = {
        "nodes": [
            {
                "id": "renamed",
                "meta": {"a": {1, 3}, "b": [1, {"a": 3, "z": 2}]},
            }
        ]
    }

    left = pydot.canonicalize_graph(first, node_attributes=["meta"])
    right = pydot.canonicalize_graph(second, node_attributes=["meta"])

    assert left.digest == right.digest


def test_canonicalization_retains_parallel_edges_and_loops() -> None:
    graph = pydot.Dot(graph_type="digraph")
    graph.add_edge(pydot.Edge("a", "b", relation="one"))
    graph.add_edge(pydot.Edge("a", "b", relation="two"))
    graph.add_edge(pydot.Edge("a", "a", relation="loop"))

    result = pydot.canonicalize_graph(graph, edge_attributes="relation")

    assert result.complete
    assert len(result.representation["edges"]) == 3
    assert sorted(
        (edge["source"], edge["target"])
        for edge in result.representation["edges"]
    ) == [(0, 0), (0, 1), (0, 1)]


def test_selected_attributes_change_canonical_form() -> None:
    left = pydot.Dot(graph_type="digraph")
    right = pydot.Dot(graph_type="digraph")
    left.add_node(pydot.Node("a", color="red"))
    right.add_node(pydot.Node("b", color="blue"))

    with_attrs = pydot.canonicalize_graph(
        left, node_attributes=["color"]
    )
    other_attrs = pydot.canonicalize_graph(
        right, node_attributes=["color"]
    )
    without_attrs = pydot.canonicalize_graph(right)

    assert with_attrs.digest != other_attrs.digest
    assert with_attrs.digest != without_attrs.digest


def test_equal_degree_non_isomorphic_graphs_do_not_collide() -> None:
    first = {
        "directed": False,
        "nodes": [{"id": value} for value in "abcdef"],
        "edges": [
            ("a", "b", {}),
            ("b", "c", {}),
            ("c", "a", {}),
            ("d", "e", {}),
            ("e", "f", {}),
            ("f", "d", {}),
        ],
    }
    second = {
        "directed": False,
        "nodes": [{"id": value} for value in "uvwxyz"],
        "edges": [
            ("u", "v", {}),
            ("v", "w", {}),
            ("w", "x", {}),
            ("x", "u", {}),
            ("y", "z", {}),
            ("z", "y", {}),
        ],
    }

    left = pydot.canonicalize_graph(first)
    right = pydot.canonicalize_graph(second)

    assert left.complete
    assert right.complete
    assert left.digest != right.digest


def test_empty_graph_has_stable_canonical_result() -> None:
    graph = pydot.Dot(graph_type="digraph")
    graph.set_node_defaults(shape="box")

    first = pydot.canonicalize_graph(graph)
    second = pydot.canonicalize_graph(graph)

    assert first.status == "complete"
    assert first.representation == {
        "directed": True,
        "nodes": [],
        "edges": [],
    }
    assert first.digest == second.digest
    assert first.searched_candidates == 1


def test_budget_exhausted_is_not_reported_as_complete() -> None:
    graph = {
        "nodes": [{"id": index} for index in range(4)],
        "edges": [
            (0, 1, {}),
            (1, 2, {}),
            (2, 3, {}),
            (3, 0, {}),
        ],
    }

    result = pydot.canonicalize_graph(graph, search_budget=3)

    assert result.status == "incomplete"
    assert not result.complete
    assert result.representation is None
    assert result.digest is None
    assert result.searched_candidates == 0


def test_undirected_edges_are_direction_invariant() -> None:
    first = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [("a", "b", {}), ("a", "a", {})],
    }
    second = {
        "nodes": [{"id": "x"}, {"id": "y"}],
        "edges": [("x", "y", {}), ("y", "y", {})],
    }

    assert pydot.canonicalize_graph(first).digest != (
        pydot.canonicalize_graph(second).digest
    )

    first["directed"] = False
    second["directed"] = False

    assert pydot.canonicalize_graph(first).digest == (
        pydot.canonicalize_graph(second).digest
    )


def test_undirected_refinement_is_direction_invariant() -> None:
    first = {
        "directed": False,
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [("a", "b", {}), ("b", "c", {})],
    }
    second = {
        "directed": False,
        "nodes": [{"id": "x"}, {"id": "y"}, {"id": "z"}],
        "edges": [("z", "y", {}), ("y", "x", {})],
    }

    assert pydot.canonicalize_graph(first).digest == (
        pydot.canonicalize_graph(second).digest
    )


def test_invalid_budget_is_rejected() -> None:
    with pytest.raises(TypeError):
        pydot.canonicalize_graph({}, search_budget=True)
    with pytest.raises(ValueError):
        pydot.canonicalize_graph({}, search_budget=-1)


def test_explicit_null_is_distinct_from_missing_attribute() -> None:
    missing = {"nodes": [{"id": "a"}], "edges": []}
    explicit_null = {
        "nodes": [{"id": "a", "label": None}],
        "edges": [],
    }

    missing_result = pydot.canonicalize_graph(
        missing, node_attributes=["label"]
    )
    null_result = pydot.canonicalize_graph(
        explicit_null, node_attributes=["label"]
    )

    assert missing_result.digest != null_result.digest
    assert null_result.representation["nodes"][0]["attributes"] == {
        "label": None
    }


def test_unsupported_attribute_value_is_rejected() -> None:
    graph = {"nodes": [{"id": "a", "value": object()}]}

    with pytest.raises(TypeError, match="does not support"):
        pydot.canonicalize_graph(
            graph, node_attributes=["value"]
        )


def test_unexpanded_subgraph_endpoint_is_rejected() -> None:
    parsed = pydot.graph_from_dot_data(
        "digraph G {a; b; subgraph cluster_x {c} a -> cluster_x;}"
    )[0]

    with pytest.raises(ValueError, match="expanded"):
        pydot.canonicalize_graph(parsed)
