# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Tests for finite-graph canonicalization."""

from __future__ import annotations

import random

import pytest

import pydot
from pydot.canonical import (
    CanonicalForm,
    canonicalize,
    canonicalize_graph,
    encode_attrs,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def cycle_edges(number: int, directed: bool, prefix: str = ""):
    return [
        (f"{prefix}{i}", f"{prefix}{(i + 1) % number}", {})
        for i in range(number)
    ]


def two_triangle_edges(directed: bool, prefix: str = ""):
    edges = [
        (f"{prefix}{i}", f"{prefix}{(i + 1) % 3}", {}) for i in range(3)
    ]
    edges += [
        (f"{prefix}{3 + i}", f"{prefix}{3 + (i + 1) % 3}", {})
        for i in range(3)
    ]
    return edges


def rename(nodes, edges, permutation):
    return (
        {permutation[name]: attrs for name, attrs in nodes.items()},
        [
            (permutation[s], permutation[t], dict(attrs))
            for s, t, attrs in edges
        ],
    )


# ---------------------------------------------------------------------
# Empty graph and basic form
# ---------------------------------------------------------------------


def test_empty_graph_is_stable_and_complete():
    first = canonicalize({}, [])
    # Directionality is part of the declared graph semantics, but each
    # empty form is individually stable and independent of budget.
    second = canonicalize({}, [], budget=0)
    undirected = canonicalize({}, [], directed=False)
    assert first.status == "complete"
    assert second.status == "complete"
    assert undirected.status == "complete"
    assert first.representation is not None
    assert first.representation.startswith("pydot-canonical-v1")
    assert first.digest == second.digest
    assert undirected.digest != first.digest
    assert first.node_count == 0
    assert first.edge_count == 0
    assert first.mapping == {}


def test_result_is_value_object():
    form = canonicalize({}, [])
    assert isinstance(form, CanonicalForm)
    with pytest.raises(Exception):
        form.complete = False  # type: ignore[misc]


# ---------------------------------------------------------------------
# Rename / declaration-order invariance
# ---------------------------------------------------------------------


def test_rename_and_shuffle_preserve_digest():
    nodes = {
        "a": {},
        "b": {"kind": "x"},
        "c": {"kind": "x"},
        "d": {},
    }
    edges = [
        ("a", "b", {}),
        ("a", "c", {}),
        ("b", "c", {"w": 1}),
        ("c", "c", {}),
    ]
    baseline = canonicalize(
        nodes,
        edges,
        node_attrs=["kind"],
        edge_attrs=["w"],
    )

    permutation = {"a": "z9", "b": "q1", "c": "m2", "d": "p3"}
    renamed_nodes, renamed_edges = rename(nodes, edges, permutation)

    shuffled = list(renamed_edges)
    random.Random(1234).shuffle(shuffled)
    shuffled = list(reversed(shuffled))

    again = canonicalize(
        dict(reversed(list(renamed_nodes.items()))),
        shuffled,
        # Duplicated and reordered declaration of semantics.
        node_attrs=["kind", "kind"],
        edge_attrs=["w", "w"],
    )
    assert again.complete
    assert again.digest == baseline.digest
    assert again.representation == baseline.representation
    assert sorted(again.mapping.values()) == list(range(4))


def test_isomorphic_pair_across_unrelated_names():
    nodes = {f"v{i}": {"role": i % 2} for i in range(8)}
    edges = cycle_edges(8, True, prefix="v")
    first = canonicalize(nodes, edges, node_attrs=["role"])

    letters = "abcdefgh"
    permutation = {f"v{i}": letters[i] for i in range(8)}
    other_nodes, other_edges = rename(nodes, edges, permutation)
    second = canonicalize(
        other_nodes, other_edges, node_attrs=["role"]
    )
    assert first.digest == second.digest


def test_node_identifiers_are_not_semantic():
    left = canonicalize({"a": {}}, [])
    right = canonicalize({"totally-different-id": {}}, [])
    assert left.digest == right.digest


# ---------------------------------------------------------------------
# Non-isomorphic, same-degree graph pairs must not collide
# ---------------------------------------------------------------------


def test_undirected_cycle_vs_two_triangles():
    nodes = {str(i): {} for i in range(6)}
    cycle = canonicalize(
        nodes, cycle_edges(6, False), directed=False
    )
    triangles = canonicalize(
        nodes, two_triangle_edges(False), directed=False
    )
    assert cycle.complete and triangles.complete
    assert cycle.digest != triangles.digest


def test_directed_cycle_vs_two_directed_triangles():
    # Every vertex has in/out degree 1; neighbour hashing alone cannot
    # distinguish them -- the IR search must.
    nodes = {str(i): {} for i in range(6)}
    cycle = canonicalize(nodes, cycle_edges(6, True), directed=True)
    triangles = canonicalize(
        nodes, two_triangle_edges(True), directed=True
    )
    assert cycle.complete and triangles.complete
    assert cycle.digest != triangles.digest


def test_direction_matters():
    nodes = {"a": {}, "b": {}}
    one_way = canonicalize(nodes, [("a", "b", {})], directed=True)
    other_way = canonicalize(nodes, [("b", "a", {})], directed=True)
    assert one_way.digest == other_way.digest  # isomorphic via rename

    undirected = canonicalize(
        nodes, [("a", "b", {})], directed=False
    )
    assert undirected.digest != one_way.digest


# ---------------------------------------------------------------------
# Parallel edges and self-loops
# ---------------------------------------------------------------------


def test_parallel_edges_count():
    nodes = {"a": {}, "b": {}}
    single = canonicalize(nodes, [("a", "b", {})])
    double = canonicalize(
        nodes, [("a", "b", {}), ("a", "b", {})]
    )
    assert single.digest != double.digest


def test_parallel_edge_replaces_self_loop_not_collide():
    nodes = {"a": {}, "b": {}}
    parallel = canonicalize(
        nodes, [("a", "b", {}), ("a", "b", {})]
    )
    loop = canonicalize(
        nodes, [("a", "b", {}), ("b", "b", {})]
    )
    assert parallel.digest != loop.digest


def test_self_loop_rename_invariant():
    directed = canonicalize({"a": {}}, [("a", "a", {})], directed=True)
    renamed = canonicalize(
        {"other": {}}, [("other", "other", {})], directed=True
    )
    assert directed.digest == renamed.digest
    undirected = canonicalize(
        {"a": {}}, [("a", "a", {})], directed=False
    )
    # Directionality is a declared property of the graph itself.
    assert undirected.digest != directed.digest


def test_semantically_typed_parallel_edges_distinguish():
    nodes = {"a": {}, "b": {}}
    typed_pair = canonicalize(
        nodes,
        [("a", "b", {"kind": "x"}), ("a", "b", {"kind": "y"})],
        edge_attrs=["kind"],
    )
    swapped = canonicalize(
        nodes,
        [("a", "b", {"kind": "y"}), ("a", "b", {"kind": "x"})],
        edge_attrs=["kind"],
    )
    assert typed_pair.digest == swapped.digest

    same_kind = canonicalize(
        nodes,
        [("a", "b", {"kind": "x"}), ("a", "b", {"kind": "x"})],
        edge_attrs=["kind"],
    )
    assert same_kind.digest != typed_pair.digest


# ---------------------------------------------------------------------
# Semantic attributes
# ---------------------------------------------------------------------


def test_undeclared_attributes_are_ignored():
    left = canonicalize(
        {"a": {"noise": 1}}, [("a", "a", {})]
    )
    right = canonicalize(
        {"a": {"noise": 2}}, [("a", "a", {})]
    )
    assert left.digest == right.digest


def test_declared_attributes_differentiate_nodes():
    left = canonicalize(
        {"a": {"kind": "x"}, "b": {"kind": "y"}},
        [("a", "b", {})],
        node_attrs=["kind"],
    )
    right = canonicalize(
        {"a": {"kind": "x"}, "b": {"kind": "x"}},
        [("a", "b", {})],
        node_attrs=["kind"],
    )
    assert left.digest != right.digest


def test_missing_distinct_from_null_value():
    assert encode_attrs({"a": None}, ["a"]) != encode_attrs({}, ["a"])


def test_attribute_order_in_mapping_is_irrelevant():
    first = encode_attrs({"z": 1, "a": [3, 1, 2]}, ["a", "z"])
    second = encode_attrs({"a": [3, 1, 2], "z": 1}, ["z", "a"])
    assert first == second


def test_nested_attribute_values_compare_stably():
    left = canonicalize(
        {"a": {"meta": {"x": 1, "y": 2}}, "b": {}},
        [],
        node_attrs=["meta"],
    )
    right = canonicalize(
        {"p": {"meta": {"y": 2, "x": 1}}, "q": {}},
        [],
        node_attrs=["meta"],
    )
    assert left.digest == right.digest


# ---------------------------------------------------------------------
# Search budget
# ---------------------------------------------------------------------


def test_budget_exhaustion_is_reported_incomplete():
    nodes = {str(i): {} for i in range(6)}
    form = canonicalize(
        nodes, two_triangle_edges(False), directed=False, budget=1
    )
    assert not form.complete
    assert form.status == "incomplete"
    assert form.representation is None
    assert form.digest is None
    assert form.mapping == {}
    assert form.budget_used > form.budget  # type: ignore[operator]


def test_unlimited_budget_completes_hard_pair():
    nodes = {str(i): {} for i in range(6)}
    form = canonicalize(
        nodes, two_triangle_edges(True), directed=True, budget=None
    )
    assert form.complete
    assert form.digest is not None


def test_incomplete_differs_from_any_complete_digest():
    nodes = {str(i): {} for i in range(6)}
    incomplete = canonicalize(
        nodes, two_triangle_edges(False), directed=False, budget=1
    )
    complete = canonicalize(
        nodes, two_triangle_edges(False), directed=False
    )
    assert incomplete.digest is None
    assert complete.digest is not None


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_unknown_endpoint_raises():
    with pytest.raises(ValueError, match="unknown node"):
        canonicalize({"a": {}}, [("a", "ghost", {})])


def test_bad_edge_tuple_raises():
    bad_edge = ("a", "a", {}, "extra")
    with pytest.raises(ValueError):
        canonicalize({"a": {}}, [bad_edge])  # type: ignore[list-item]


# ---------------------------------------------------------------------
# pydot adapter
# ---------------------------------------------------------------------


def test_canonicalize_graph_rename_invariant():
    def build(prefix):
        graph = pydot.Dot(graph_type="digraph")
        graph.add_node(pydot.Node(prefix + "a", color="red"))
        graph.add_node(pydot.Node(prefix + "b"))
        graph.add_node(pydot.Node(prefix + "c"))
        graph.add_edge(
            pydot.Edge(prefix + "a", prefix + "b", weight=7)
        )
        graph.add_edge(pydot.Edge(prefix + "b", prefix + "c"))
        graph.add_edge(
            pydot.Edge(prefix + "c", prefix + "a", weight=7)
        )
        return graph

    first = canonicalize_graph(
        build(""), node_attrs=["color"], edge_attrs=["weight"]
    )
    second = canonicalize_graph(
        build("renamed-"),
        edge_attrs=["weight"],
        node_attrs=["color"],
    )
    assert first.complete and second.complete
    assert first.digest == second.digest


def test_canonicalize_graph_skips_defaults_and_honors_semantics():
    graph = pydot.Dot(graph_type="graph")
    graph.set_node_defaults(shape="box")
    graph.set_edge_defaults(color="gray")
    graph.add_node(pydot.Node("x"))
    graph.add_node(pydot.Node("y"))
    graph.add_edge(pydot.Edge("x", "y", label="hello"))

    without_label = canonicalize_graph(graph)
    with_label = canonicalize_graph(graph, edge_attrs=["label"])
    assert without_label.digest != with_label.digest
    # The default pseudo-nodes must not appear as real vertices.
    assert without_label.node_count == 2


def test_canonicalize_graph_implicit_endpoints():
    graph = pydot.Dot(graph_type="digraph")
    graph.add_edge(pydot.Edge("alpha", "beta"))
    form = canonicalize_graph(graph)
    assert form.complete
    assert form.node_count == 2


def test_canonicalize_graph_self_loop_and_parallel():
    graph = pydot.Dot(graph_type="digraph")
    graph.add_edge(pydot.Edge("a", "b"))
    graph.add_edge(pydot.Edge("a", "b", label="second"))
    graph.add_edge(pydot.Edge("b", "b"))
    form = canonicalize_graph(graph, edge_attrs=["label"])
    assert form.complete
    assert form.edge_count == 3
