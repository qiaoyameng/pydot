# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit testing of pydot node contraction."""

from __future__ import annotations

import pytest

import pydot
from pydot.exceptions import (
    AttributeConflictError,
    ClusterConflictError,
    ContractionError,
    OverlappingGroupsError,
    PortMappingError,
    UnknownNodeError,
)


def _service_graph() -> pydot.Dot:
    """A directed multigraph of service instances.

    Instances a1, a2 belong to service A; b1, b2 to service B; c is a
    standalone node. Edges include parallel calls, an intra-service
    call and a pre-existing self-loop.
    """
    graph = pydot.Dot("deps", graph_type="digraph")
    for name in ("a1", "a2", "b1", "b2", "c"):
        graph.add_node(pydot.Node(name))
    for src, dst in [
        ("a1", "b1"),
        ("a2", "b2"),  # parallel to a1->b1 after contraction
        ("a1", "a2"),  # becomes a self-loop on A
        ("b1", "a1"),
        ("c", "a1"),
        ("a2", "c"),
        ("b2", "b2"),  # already a self-loop, stays one
        ("c", "b1"),
    ]:
        graph.add_edge(pydot.Edge(src, dst))
    return graph


def _edge_pairs(graph: pydot.Graph) -> list[tuple[str, str]]:
    return [
        (str(edge.get_source()), str(edge.get_destination()))
        for edge in graph.get_edge_list()
    ]


def test_contract_multigraph_every_original_edge_is_accounted_for():
    graph = _service_graph()
    original_edges = _edge_pairs(graph)

    result = pydot.contract_nodes(
        graph, {"svcA": ["a1", "a2"], "svcB": ["b1", "b2"]}
    )

    # The input graph is untouched.
    assert _edge_pairs(graph) == original_edges

    # Every original edge appears exactly once in the provenance
    # records or the explicit deletion records.
    origins = result.all_origins()
    assert len(origins) == len(original_edges)
    assert [origin.index for origin in origins] == list(
        range(len(original_edges))
    )
    accounted = [(o.source, o.destination) for o in origins]
    assert sorted(accounted) == sorted(original_edges)

    # Nothing was dropped with the default self-loop policy.
    assert result.dropped == []

    # Each contracted edge points at its original call(s).
    by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for edge in result.edges:
        by_pair.setdefault((edge.source, edge.destination), []).extend(
            (o.source, o.destination) for o in edge.origins
        )
    assert by_pair[("svcA", "svcB")] == [("a1", "b1"), ("a2", "b2")]
    assert by_pair[("svcA", "svcA")] == [("a1", "a2")]
    assert by_pair[("svcB", "svcA")] == [("b1", "a1")]
    assert by_pair[("c", "svcA")] == [("c", "a1")]
    assert by_pair[("svcA", "c")] == [("a2", "c")]
    assert by_pair[("svcB", "svcB")] == [("b2", "b2")]
    assert by_pair[("c", "svcB")] == [("c", "b1")]

    # The contracted graph itself matches the provenance records.
    assert sorted(_edge_pairs(result.graph)) == sorted(
        (edge.source, edge.destination) for edge in result.edges
    )
    assert result.node_map == {
        "a1": "svcA",
        "a2": "svcA",
        "b1": "svcB",
        "b2": "svcB",
        "c": "c",
    }
    assert {n.get_name() for n in result.graph.get_node_list()} == {
        "svcA",
        "svcB",
        "c",
    }


def test_contract_multigraph_drop_self_loops_and_merge_parallel():
    graph = _service_graph()
    original_edges = _edge_pairs(graph)

    result = pydot.contract_nodes(
        graph,
        {"svcA": ["a1", "a2"], "svcB": ["b1", "b2"]},
        self_loops="drop",
        merge_parallel=True,
    )

    # Dropped self-loops are explicitly recorded, not silently lost.
    dropped = [(d.origin.source, d.origin.destination) for d in result.dropped]
    assert sorted(dropped) == [("a1", "a2"), ("b2", "b2")]
    assert all(d.reason == "self_loop" for d in result.dropped)

    # Still, every original edge is accounted for exactly once.
    origins = result.all_origins()
    assert [origin.index for origin in origins] == list(
        range(len(original_edges))
    )
    assert sorted((o.source, o.destination) for o in origins) == sorted(
        original_edges
    )

    # Parallel edges merged into one summary edge keeping both origins.
    merged = [
        edge
        for edge in result.edges
        if (edge.source, edge.destination) == ("svcA", "svcB")
    ]
    assert len(merged) == 1
    assert [(o.source, o.destination) for o in merged[0].origins] == [
        ("a1", "b1"),
        ("a2", "b2"),
    ]
    assert _edge_pairs(result.graph) == [
        ("svcA", "svcB"),
        ("svcB", "svcA"),
        ("c", "svcA"),
        ("svcA", "c"),
        ("c", "svcB"),
    ]


def test_contract_self_loops_kept_by_default():
    graph = _service_graph()
    result = pydot.contract_nodes(graph, {"svcA": ["a1", "a2"]})
    assert ("svcA", "svcA") in _edge_pairs(result.graph)
    assert result.dropped == []


def test_contract_invalid_self_loop_policy():
    graph = _service_graph()
    with pytest.raises(ValueError, match="self_loops"):
        pydot.contract_nodes(
            graph, {"svcA": ["a1", "a2"]}, self_loops="sometimes"
        )


def test_contract_overlapping_groups_are_rejected():
    graph = _service_graph()
    with pytest.raises(OverlappingGroupsError) as excinfo:
        pydot.contract_nodes(
            graph, {"svcA": ["a1", "a2"], "dup": ["a2", "c"]}
        )
    assert excinfo.value.overlapping == {"a2": ["dup", "svcA"]}


def test_contract_unknown_member_is_rejected():
    graph = _service_graph()
    with pytest.raises(UnknownNodeError) as excinfo:
        pydot.contract_nodes(graph, {"svcA": ["a1", "ghost"]})
    assert excinfo.value.unknown == ["ghost"]


def test_contract_name_collision_with_existing_node():
    graph = _service_graph()
    with pytest.raises(ContractionError, match="collides"):
        pydot.contract_nodes(graph, {"c": ["a1", "a2"]})


def test_contract_merged_edge_attribute_conflict_needs_reducer():
    graph = pydot.Dot("g", graph_type="digraph")
    graph.add_edge(pydot.Edge("a1", "b1", color="red", weight="1"))
    graph.add_edge(pydot.Edge("a2", "b2", color="blue", weight="1"))

    with pytest.raises(AttributeConflictError) as excinfo:
        pydot.contract_nodes(
            graph,
            {"svcA": ["a1", "a2"], "svcB": ["b1", "b2"]},
            merge_parallel=True,
        )
    assert excinfo.value.attribute == "color"
    assert excinfo.value.values == ["red", "blue"]

    result = pydot.contract_nodes(
        graph,
        {"svcA": ["a1", "a2"], "svcB": ["b1", "b2"]},
        merge_parallel=True,
        attribute_reducer=lambda name, values: "+".join(sorted(values)),
    )
    (edge,) = result.edges
    # Agreeing values are kept, conflicts go through the reducer.
    assert edge.attributes == {"color": "blue+red", "weight": "1"}


def test_contract_node_attribute_conflict_needs_reducer():
    graph = pydot.Dot("g", graph_type="digraph")
    graph.add_node(pydot.Node("a1", shape="box", region="east"))
    graph.add_node(pydot.Node("a2", shape="box", region="west"))

    with pytest.raises(AttributeConflictError):
        pydot.contract_nodes(graph, {"svcA": ["a1", "a2"]})

    result = pydot.contract_nodes(
        graph,
        {"svcA": ["a1", "a2"]},
        attribute_reducer=lambda name, values: ",".join(sorted(values)),
    )
    (node,) = result.graph.get_node("svcA")
    assert node.get_attributes() == {
        "shape": "box",
        "region": "east,west",
    }


def test_contract_ported_edges_require_port_map():
    graph = pydot.Dot("g", graph_type="digraph")
    graph.add_edge(pydot.Edge("a1:out", "b1:in"))
    graph.add_edge(pydot.Edge("ext:plain", "a1"))

    # Ports on contracted nodes without a mapping refuse contraction.
    with pytest.raises(PortMappingError) as excinfo:
        pydot.contract_nodes(
            graph, {"svcA": ["a1"], "svcB": ["b1"]}
        )
    assert excinfo.value.node == "a1"
    assert excinfo.value.port == "out"
    assert excinfo.value.edge == ("a1:out", "b1:in")

    # With a mapping, ports are rewritten onto the contracted nodes.
    result = pydot.contract_nodes(
        graph,
        {"svcA": ["a1"], "svcB": ["b1"]},
        port_map={("a1", "out"): "out", ("b1", "in"): "in"},
    )
    pairs = _edge_pairs(result.graph)
    assert ("svcA:out", "svcB:in") in pairs
    # Ports on non-contracted nodes are left untouched.
    assert ("ext:plain", "svcA") in pairs


def test_contract_port_map_may_drop_port():
    graph = pydot.Dot("g", graph_type="digraph")
    graph.add_edge(pydot.Edge("a1:out", "b1"))
    result = pydot.contract_nodes(
        graph, {"svcA": ["a1"]}, port_map={("a1", "out"): None}
    )
    assert _edge_pairs(result.graph) == [("svcA", "b1")]


def test_contract_nested_cluster_conflict_is_locatable():
    graph = pydot.Dot("g", graph_type="digraph")
    cluster_0 = pydot.Cluster("frontend")
    cluster_0.add_node(pydot.Node("a1"))
    inner = pydot.Cluster("backend")
    inner.add_node(pydot.Node("a2"))
    cluster_1 = pydot.Cluster("services")
    cluster_1.add_subgraph(inner)
    graph.add_subgraph(cluster_0)
    graph.add_subgraph(cluster_1)
    graph.add_edge(pydot.Edge("a1", "a2"))

    with pytest.raises(ClusterConflictError) as excinfo:
        pydot.contract_nodes(graph, {"svcA": ["a1", "a2"]})

    # The conflict pinpoints each member's cluster path.
    assert excinfo.value.group == "svcA"
    assert excinfo.value.memberships == {
        "a1": [("cluster_frontend",)],
        "a2": [("cluster_services", "cluster_backend")],
    }
    message = str(excinfo.value)
    assert "cluster_frontend" in message
    assert "cluster_services/cluster_backend" in message


def test_contract_places_node_in_shared_cluster():
    graph = pydot.Dot("g", graph_type="digraph")
    cluster = pydot.Cluster("backend")
    cluster.add_node(pydot.Node("a1"))
    cluster.add_node(pydot.Node("a2"))
    graph.add_subgraph(cluster)
    graph.add_node(pydot.Node("c"))
    graph.add_edge(pydot.Edge("c", "a1"))

    result = pydot.contract_nodes(graph, {"svcA": ["a1", "a2"]})

    # The contracted node lives in the same cluster as its members.
    assert result.graph.get_node("svcA") == []
    (new_cluster,) = result.graph.get_subgraph("cluster_backend")
    assert [n.get_name() for n in new_cluster.get_node_list()] == ["svcA"]
    assert _edge_pairs(result.graph) == [("c", "svcA")]


def test_contract_undirected_merges_regardless_of_orientation():
    graph = pydot.Dot("g", graph_type="graph")
    graph.add_edge(pydot.Edge("a1", "b1"))
    graph.add_edge(pydot.Edge("b2", "a2"))
    result = pydot.contract_nodes(
        graph,
        {"svcA": ["a1", "a2"], "svcB": ["b1", "b2"]},
        merge_parallel=True,
    )
    assert len(result.edges) == 1
    assert len(result.edges[0].origins) == 2


def test_contract_result_graph_renders():
    graph = _service_graph()
    result = pydot.contract_nodes(
        graph, {"svcA": ["a1", "a2"], "svcB": ["b1", "b2"]}
    )
    dot_source = result.graph.to_string()
    assert "svcA" in dot_source
    assert "svcA -> svcB" in dot_source
    parsed = pydot.graph_from_dot_data(dot_source)
    assert parsed is not None
