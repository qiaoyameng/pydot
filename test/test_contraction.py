from __future__ import annotations

import pytest

import pydot


def _service_graph() -> pydot.Dot:
    graph = pydot.Dot("services")
    for name in ("a", "b", "c", "d", "e"):
        graph.add_node(pydot.Node(name))
    for source, destination in (
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("c", "e"),
        ("d", "e"),
        ("a", "b"),
        ("c", "c"),
    ):
        graph.add_edge(
            pydot.Edge(source, destination, label=source + destination)
        )
    return graph


def test_contract_two_groups_keeps_edges_and_provenance() -> None:
    result = pydot.contract_nodes(
        _service_graph(),
        {"x": ["a", "b"], "y": ["d", "e"]},
    )

    assert result.node_sources["x"] == ("a", "b")
    assert result.node_sources["y"] == ("d", "e")
    assert [
        (edge.get_source(), edge.get_destination())
        for edge in result.graph.get_edges()
    ] == [
        ("x", "c"),
        ("x", "c"),
        ("c", "y"),
        ("c", "y"),
        ("y", "y"),
        ("x", "x"),
        ("c", "c"),
    ]

    accounted_for = {
        edge_id
        for origin in result.edge_origins
        for edge_id in origin.original_edge_ids
    } | {deleted.original_edge_id for deleted in result.deleted_edges}
    assert accounted_for == set(range(7))


def test_self_loops_can_be_deleted_with_explicit_records() -> None:
    result = pydot.contract_nodes(
        _service_graph(),
        {"x": ["a", "b"], "y": ["d", "e"]},
        self_loops="drop",
    )

    assert [
        (edge.get_source(), edge.get_destination())
        for edge in result.graph.get_edges()
    ] == [
        ("x", "c"),
        ("x", "c"),
        ("c", "y"),
        ("c", "y"),
        ("c", "c"),
    ]
    assert [
        (deleted.original_edge_id, deleted.reason)
        for deleted in result.deleted_edges
    ] == [(4, "self_loop"), (5, "self_loop")]


def test_parallel_edges_are_merged_with_reducer() -> None:
    graph = pydot.Dot("calls")
    graph.add_node(pydot.Node("a"))
    graph.add_node(pydot.Node("b"))
    graph.add_node(pydot.Node("c"))
    graph.add_edge(pydot.Edge("a", "c", label="a-to-c", weight=1))
    graph.add_edge(pydot.Edge("b", "c", label="b-to-c", weight=1))

    result = pydot.contract_nodes(
        graph,
        {"s": ["a", "b"]},
        merge_parallel_edges=True,
        attribute_reducer=lambda name, values: "|".join(map(str, values)),
    )

    assert len(result.graph.get_edges()) == 1
    edge = result.graph.get_edges()[0]
    assert edge.get_source() == "s"
    assert edge.get_destination() == "c"
    assert edge.get("label") == "a-to-c|b-to-c"
    assert edge.get("weight") == 1
    assert result.edge_origins[0].original_edge_ids == (0, 1)
    assert result.edge_sources[0] == (0, 1)
    assert result.edge_origins[0].sources == ("a", "b")


def test_conflicting_merged_attributes_require_reducer() -> None:
    graph = pydot.Dot("calls")
    graph.add_edge(pydot.Edge("a", "c", label="one"))
    graph.add_edge(pydot.Edge("b", "c", label="two"))

    with pytest.raises(pydot.ContractionError) as exc_info:
        pydot.contract_nodes(
            graph,
            {"s": ["a", "b"]},
            merge_parallel_edges=True,
        )

    conflict = exc_info.value.conflicts[0]
    assert conflict.kind == "attribute_conflict"
    assert conflict.attribute == "label"
    assert conflict.edge_ids == (0, 1)


def test_conflicting_node_attributes_require_reducer() -> None:
    graph = pydot.Dot("attributes")
    graph.add_node(pydot.Node("a", color="red"))
    graph.add_node(pydot.Node("b", color="blue"))

    with pytest.raises(pydot.ContractionError) as exc_info:
        pydot.contract_nodes(graph, {"s": ["a", "b"]})

    conflict = exc_info.value.conflicts[0]
    assert conflict.kind == "attribute_conflict"
    assert conflict.group == "s"
    assert conflict.attribute == "color"
    assert conflict.values == ("red", "blue")


def test_ports_require_explicit_mapping() -> None:
    graph = pydot.Dot("ports")
    graph.add_edge(pydot.Edge("a:p", "c"))

    with pytest.raises(pydot.ContractionError) as exc_info:
        pydot.contract_nodes(graph, {"s": ["a"]})

    conflict = exc_info.value.conflicts[0]
    assert conflict.kind == "missing_port_mapping"
    assert conflict.edge_id == 0
    assert conflict.node == "a"


def test_ports_are_rewritten_with_explicit_mapping() -> None:
    graph = pydot.Dot("ports")
    graph.add_edge(pydot.Edge("a:p", "b:q"))

    result = pydot.contract_nodes(
        graph,
        {"s": ["a", "b"]},
        port_mapping={("s", "p"): "pp", ("s", "q"): "qq"},
    )

    edge = result.graph.get_edges()[0]
    assert edge.get_source() == "s:pp"
    assert edge.get_destination() == "s:qq"
    assert result.edge_origins[0].sources == ("a:p",)


def test_nested_cluster_group_conflict_is_locatable() -> None:
    graph = pydot.Dot("clusters")
    outer = pydot.Cluster("outer")
    first = pydot.Cluster("first")
    second = pydot.Cluster("second")
    first.add_node(pydot.Node("a"))
    second.add_node(pydot.Node("b"))
    outer.add_subgraph(first)
    outer.add_subgraph(second)
    graph.add_subgraph(outer)

    with pytest.raises(pydot.ContractionError) as exc_info:
        pydot.contract_nodes(graph, {"s": ["a", "b"]})

    conflict = exc_info.value.conflicts[0]
    assert conflict.kind == "cluster_group_membership_conflict"
    assert conflict.group == "s"
    assert ("cluster_outer", "cluster_first") in conflict.paths
    assert ("cluster_outer", "cluster_second") in conflict.paths
