# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Tests for finite rank-constraint checking."""

from __future__ import annotations

import pytest

import pydot


def test_rank_constraints_feasible_with_implicit_nodes() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            a -> c;
            b -> c;
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert result.feasible is True
    assert result.conflicts == ()
    assert result.unsupported == ()
    assert result.ranks["a"] == result.ranks["b"]
    assert result.ranks["c"] == result.ranks["a"] + 1


def test_rank_constraint_false_is_ignored() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            a -> b [minlen=3, constraint=false];
        }
        """
    )[0]

    result = pydot.check_rank_constraints(graph)

    assert result.feasible is True
    assert result.ranks["a"] == result.ranks["b"]


def test_rank_same_minlen_conflict_chain() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            a -> b [minlen=2];
        }
        """
    )[0]

    result = graph.check_rank_constraints()
    (conflict,) = result.conflicts
    steps = conflict.steps

    assert result.status == "infeasible"
    assert result.feasible is False
    assert result.ranks == {}
    assert steps[0].source == "a"
    assert steps[0].destination == "b"
    assert steps[0].kind == "edge"
    assert steps[0].min_rank_gap == 2
    assert steps[1].source == "b"
    assert steps[1].destination == "a"
    assert steps[1].kind == "same"
    assert steps[1].min_rank_gap == 0
    assert steps[-1].destination == steps[0].source
    assert sum(step.min_rank_gap for step in steps) > 0
    assert {constraint.kind for constraint in conflict.constraints} == {
        "edge",
        "same",
    }


def test_parallel_edges_keep_declaration_identity() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            a -> b [id=e0, minlen=0];
            a -> b [id=e1, minlen=1];
        }
        """
    )[0]

    result = graph.check_rank_constraints()
    (conflict,) = result.conflicts
    edge_steps = [step for step in conflict.steps if step.kind == "edge"]

    assert result.status == "infeasible"
    assert len(edge_steps) == 1
    edge_step = edge_steps[0]
    second_edge = graph.get_edge("a", "b")[1]
    assert edge_step.edge.get_attributes() == second_edge.get_attributes()
    assert edge_step.edge.get("id") == "e1"
    assert edge_step.declaration != conflict.steps[-1].declaration


def test_zero_minlen_makes_same_rank_edge_feasible() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            a -> b [minlen=0];
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert result.ranks["a"] == result.ranks["b"] == 0


def test_directed_positive_cycle_is_infeasible() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            a -> b [minlen=1];
            b -> c [minlen=2];
            c -> a [minlen=3];
        }
        """
    )[0]

    result = graph.check_rank_constraints()
    (conflict,) = result.conflicts

    assert result.status == "infeasible"
    assert sum(step.min_rank_gap for step in conflict.steps) == 6
    assert {step.destination for step in conflict.steps} == {"a", "b", "c"}


def test_ports_refer_to_their_base_nodes() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            a:p1 -> b:p2 [constraint=false];
            a:p1 -> c:p3;
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert "a" in result.ranks
    assert "b" in result.ranks
    assert result.ranks["c"] == result.ranks["a"] + 1


def test_rank_min_is_unsupported() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=min; a; b; }
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "unsupported"
    assert result.feasible is None
    assert result.ranks == {}
    assert result.unsupported[0].feature == "rank"
    assert 'rank="min"' in result.unsupported[0].message


def test_undirected_graph_is_unsupported() -> None:
    graph = pydot.Graph(graph_type="graph")
    graph.add_edge(pydot.Edge("a", "b"))

    result = graph.check_rank_constraints()

    assert result.status == "unsupported"
    assert result.unsupported[0].feature == "graph-type"


def test_compound_edge_is_unsupported() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            subgraph cluster_x { a; }
            a -> b [lhead=cluster_x];
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "unsupported"
    assert result.unsupported[0].feature == "compound-endpoint"


def test_edge_defaults_apply_by_declaration_order() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            edge [constraint=false];
            a -> b [minlen=3];
            edge [constraint=true];
            b -> c [minlen=2];
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert result.ranks["b"] == result.ranks["a"]
    assert result.ranks["c"] == result.ranks["b"] + 2


def test_overlapping_same_groups_are_transitive() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; }
            { rank=same; b; c; }
            a -> d;
            c -> d;
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert result.ranks["a"] == result.ranks["b"] == result.ranks["c"]
    assert result.ranks["d"] == result.ranks["a"] + 1


def test_numeric_implicit_endpoint_is_node_name() -> None:
    graph = pydot.Dot()
    graph.add_edge(pydot.Edge(1, 2, minlen=2))

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert result.ranks["2"] == result.ranks["1"] + 2


def test_edge_direction_does_not_change_rank_constraint() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            a -> b [dir=back];
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "feasible"
    assert result.ranks["b"] == result.ranks["a"] + 1


def test_newrank_is_unsupported() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            newrank=true;
            { rank=same; a; b; }
        }
        """
    )[0]

    result = graph.check_rank_constraints()

    assert result.status == "unsupported"
    assert result.unsupported[0].feature == "newrank"


def test_edge_inside_same_group_keeps_declaration_identity() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            { rank=same; a; b; a -> b [id=inside, minlen=1]; }
        }
        """
    )[0]

    result = graph.check_rank_constraints()
    edge_step = [
        step for step in result.conflicts[0].steps if step.kind == "edge"
    ][0]

    assert result.status == "infeasible"
    assert edge_step.edge.get("id") == "inside"
    assert edge_step.source == "a"
    assert edge_step.destination == "b"


def test_negative_minlen_is_rejected() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            a -> b [minlen=-1];
        }
        """
    )[0]

    with pytest.raises(pydot.Error, match="non-negative integer"):
        graph.check_rank_constraints()
