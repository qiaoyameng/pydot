# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit testing of `pydot.ranking`."""

from __future__ import annotations

import pytest

import pydot
from pydot import ConstraintStatus
from pydot.ranking import check_rank_constraints


def _result(source: str):
    graph = pydot.graph_from_dot_data(source)[0]
    return check_rank_constraints(graph)


def test_minlen_only_is_feasible() -> None:
    result = _result("digraph G { a -> b; b -> c [minlen=2]; }")
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.layers == {"a": 0, "b": 1, "c": 3}


def test_implicit_nodes_get_layers() -> None:
    graph = pydot.Dot(graph_type="digraph")
    graph.add_edge(pydot.Edge("a", "b"))
    result = check_rank_constraints(graph)
    assert result.status is ConstraintStatus.FEASIBLE
    assert set(result.layers) == {"a", "b"}


def test_disconnected_nodes_are_layer_zero() -> None:
    result = _result("digraph G { a -> b; d; }")
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.layers["d"] == 0


def test_same_rank_forces_equal_layers() -> None:
    result = _result(
        "digraph G { { rank=same; a; b } a -> b [minlen=0]; }"
    )
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.layers["a"] == result.layers["b"]
    group = result.same_rank_groups[0]
    assert group.index == 1
    assert group.members == ("a", "b")


def test_same_rank_conflicts_with_minlen_one() -> None:
    result = _result(
        "digraph G { { rank=same; a; b } a -> b [minlen=1]; }"
    )
    assert result.status is ConstraintStatus.INFEASIBLE
    assert result.layers is None
    assert len(result.conflicts) == 1
    chain = result.conflicts[0]
    assert chain.start_node == "b"
    assert chain.total_difference == 1
    kinds = [(step.kind, step.min_difference) for step in chain.steps]
    assert kinds == [("same_rank", 0), ("edge", 1)]


def test_conflict_chain_is_closed_and_verifiable() -> None:
    result = _result(
        """
        digraph G {
          { rank=same; a; c }
          a -> b; b -> c;
        }
        """
    )
    assert result.status is ConstraintStatus.INFEASIBLE
    chain = result.conflicts[0]
    for i, step in enumerate(chain.steps):
        following = chain.steps[(i + 1) % len(chain.steps)]
        assert step.head == following.tail
    assert chain.steps[0].tail == chain.start_node
    assert chain.steps[-1].head == chain.start_node
    assert (
        sum(step.min_difference for step in chain.steps)
        == chain.total_difference
        > 0
    )


def test_parallel_edges_keep_separate_identities() -> None:
    result = _result(
        """
        digraph G {
          { rank=same; a; b }
          a -> b [minlen=1, id="first"];
          a -> b [minlen=1, id="second"];
        }
        """
    )
    assert result.status is ConstraintStatus.INFEASIBLE
    assert [con.edge_index for con in result.constraints] == [1, 2]
    assert [con.edge_id for con in result.constraints] == [
        "first",
        "second",
    ]
    indices = {
        step.edge_index
        for chain in result.conflicts
        for step in chain.steps
        if step.kind == "edge"
    }
    assert indices == {1, 2}


def test_constraint_false_is_ignored() -> None:
    result = _result(
        """
        digraph G {
          { rank=same; a; b }
          a -> b [minlen=9, constraint=false];
        }
        """
    )
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.layers["a"] == result.layers["b"]
    assert result.constraints == ()


def test_constraint_quoted_bool_is_parsed() -> None:
    result = _result(
        'digraph G { { rank=same; a; b } '
        'a -> b [minlen=9, constraint="false"]; }'
    )
    assert result.status is ConstraintStatus.FEASIBLE


def test_edge_default_constraint_false() -> None:
    result = _result(
        "digraph G { edge [constraint=false]; a -> b [minlen=5]; }"
    )
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.constraints == ()


def test_edge_default_minlen_is_inherited() -> None:
    result = _result(
        "digraph G { edge [minlen=3]; a -> b; b -> c [minlen=1]; }"
    )
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.layers["b"] - result.layers["a"] == 3
    assert result.layers["c"] - result.layers["b"] == 1


def test_edge_defaults_are_scoped_to_subgraph() -> None:
    result = _result(
        """
        digraph G {
          { edge [constraint=false]; a -> b; }
          a -> b;
          b -> c; c -> a;
        }
        """
    )
    assert result.status is ConstraintStatus.INFEASIBLE


def test_minlen_zero_is_allowed_both_ways() -> None:
    result = _result(
        "digraph G { a -> b [minlen=0]; b -> a [minlen=0]; }"
    )
    assert result.status is ConstraintStatus.FEASIBLE


def test_positive_self_loop_is_infeasible() -> None:
    result = _result("digraph G { a -> a [minlen=1]; }")
    assert result.status is ConstraintStatus.INFEASIBLE
    chain = result.conflicts[0]
    assert chain.steps[0].tail == "a"
    assert chain.steps[0].head == "a"
    assert chain.total_difference == 1


def test_zero_self_loop_is_feasible() -> None:
    result = _result("digraph G { a -> a [minlen=0]; }")
    assert result.status is ConstraintStatus.FEASIBLE


def test_inline_subgraph_endpoint_is_expanded() -> None:
    result = _result("digraph G { a -> {b c} [minlen=1]; }")
    assert result.status is ConstraintStatus.FEASIBLE
    assert {con.tail for con in result.constraints} == {"a"}
    assert {con.head for con in result.constraints} == {"b", "c"}
    assert {con.edge_index for con in result.constraints} == {1}
    assert all(con.expanded for con in result.constraints)
    assert set(result.layers) == {"a", "b", "c"}


def test_ports_are_stripped_to_node_names() -> None:
    result = _result("digraph G { a:p1 -> b:q1; }")
    assert result.status is ConstraintStatus.FEASIBLE
    assert set(result.layers) == {"a", "b"}


def test_chained_edges_share_attributes() -> None:
    result = _result("digraph G { a -> b -> c [minlen=2]; }")
    assert [con.minlen for con in result.constraints] == [2, 2]
    assert result.layers == {"a": 0, "b": 2, "c": 4}


def test_rank_min_is_unsupported() -> None:
    result = _result("digraph G { { rank=min; a; b } }")
    assert result.status is ConstraintStatus.UNSUPPORTED
    assert result.layers is None
    assert result.unsupported[0].kind == "rank_value"
    assert result.same_rank_groups == ()


@pytest.mark.parametrize("value", ["source", "max", "sink"])
def test_other_rank_values_are_unsupported(value: str) -> None:
    result = _result(f"digraph G {{ {{ rank={value}; a }} }}")
    assert result.status is ConstraintStatus.UNSUPPORTED
    assert result.unsupported[0].kind == "rank_value"


def test_undirected_graph_is_unsupported() -> None:
    graph = pydot.graph_from_dot_data("graph G { a -- b; }")[0]
    result = check_rank_constraints(graph)
    assert result.status is ConstraintStatus.UNSUPPORTED
    assert result.unsupported[0].kind == "undirected_graph"


def test_lhead_endpoint_is_unsupported() -> None:
    result = _result(
        "digraph G { a -> b [lhead=cluster_x]; }"
    )
    assert result.status is ConstraintStatus.UNSUPPORTED
    assert result.unsupported[0].kind == "logical_endpoint"


def test_rank_on_graph_root_is_unsupported() -> None:
    result = _result("digraph G { rank=same; a -> b; }")
    assert result.status is ConstraintStatus.UNSUPPORTED
    assert result.unsupported[0].kind == "rank_on_graph"


def test_invalid_minlen_negative_raises() -> None:
    result_graph = pydot.graph_from_dot_data(
        "digraph G { a -> b [minlen=-1]; }"
    )[0]
    with pytest.raises(ValueError, match="minlen"):
        check_rank_constraints(result_graph)


def test_invalid_minlen_non_numeric_raises() -> None:
    result_graph = pydot.graph_from_dot_data(
        'digraph G { a -> b [minlen="wide"]; }'
    )[0]
    with pytest.raises(ValueError, match="minlen"):
        check_rank_constraints(result_graph)

def test_invalid_constraint_value_raises() -> None:
    result_graph = pydot.graph_from_dot_data(
        "digraph G { a -> b [constraint=maybe]; }"
    )[0]
    with pytest.raises(ValueError, match="constraint"):
        check_rank_constraints(result_graph)


def test_non_graph_argument_raises_type_error() -> None:
    with pytest.raises(TypeError):
        check_rank_constraints("digraph {}")


def test_graph_method_delegates() -> None:
    graph = pydot.Dot(graph_type="digraph")
    graph.add_edge(pydot.Edge("a", "b"))
    result = graph.check_rank_constraints()
    assert result.status is ConstraintStatus.FEASIBLE


def test_independent_infeasible_groups_are_both_reported() -> None:
    result = _result(
        """
        digraph G {
          { rank=same; a1; a2 } a1 -> a2;
          { rank=same; b1; b2 } b1 -> b2;
        }
        """
    )
    assert result.status is ConstraintStatus.INFEASIBLE
    assert len(result.conflicts) == 2


def test_transitive_same_rank_groups_merge() -> None:
    result = _result(
        """
        digraph G {
          { rank=same; a; b }
          { rank=same; b; c }
          a -> c;
        }
        """
    )
    assert result.status is ConstraintStatus.INFEASIBLE
    group_indices = {
        step.group_index
        for chain in result.conflicts
        for step in chain.steps
        if step.kind == "same_rank"
    }
    assert group_indices == {1, 2}


def test_empty_graph_is_feasible() -> None:
    graph = pydot.Dot(graph_type="digraph")
    result = check_rank_constraints(graph)
    assert result.status is ConstraintStatus.FEASIBLE
    assert result.layers == {}


def test_conflict_step_points_to_declared_edge() -> None:
    result = _result(
        'digraph G { { rank=same; a; b } '
        'a -> b [minlen=1, id="label-e"]; }'
    )
    step = next(
        step
        for chain in result.conflicts
        for step in chain.steps
        if step.kind == "edge"
    )
    assert step.edge_index == 1
    assert step.edge_id == "label-e"
    assert step.description.startswith("edge #1")


def test_same_rank_step_points_to_group() -> None:
    result = _result(
        "digraph G { { rank=same; a; b } a -> b; }"
    )
    step = next(
        step
        for chain in result.conflicts
        for step in chain.steps
        if step.kind == "same_rank"
    )
    assert step.group_index == 1
    assert step.min_difference == 0


def test_nested_inline_endpoint_expands_to_leaf_nodes() -> None:
    result = _result("digraph G { a -> { {b c} d }; }")
    assert result.status is ConstraintStatus.FEASIBLE
    assert set(result.layers) == {"a", "b", "c", "d"}
    assert {
        (con.tail, con.head) for con in result.constraints
    } == {("a", "b"), ("a", "c"), ("a", "d")}


def test_group_members_include_edge_endpoints_in_scope() -> None:
    result = _result(
        "digraph G { { rank=same; a -> b [minlen=0]; c; } }"
    )
    assert result.status is ConstraintStatus.FEASIBLE
    assert set(result.same_rank_groups[0].members) == {"a", "b", "c"}
    assert result.layers["a"] == result.layers["b"] == result.layers["c"]

    # An internal edge with the default minlen contradicts the group.
    contradictory = _result(
        "digraph G { { rank=same; a -> b; c; } }"
    )
    assert contradictory.status is ConstraintStatus.INFEASIBLE


def test_internal_edge_of_endpoint_subgraph_is_a_constraint() -> None:
    result = _result(
        "digraph G { { rank=same; a; b; a -> b; } -> z; }"
    )
    assert result.status is ConstraintStatus.INFEASIBLE
    expanded = [con for con in result.constraints if con.expanded]
    assert {(con.tail, con.head) for con in expanded} == {
        ("a", "z"),
        ("b", "z"),
    }


def test_invalid_minlen_on_disabled_edge_still_raises() -> None:
    graph = pydot.graph_from_dot_data(
        "digraph G { a -> b [minlen=bad, constraint=false]; }"
    )[0]
    with pytest.raises(ValueError, match="minlen"):
        check_rank_constraints(graph)


def test_numeric_endpoints_become_string_names() -> None:
    graph = pydot.Dot(graph_type="digraph")
    graph.add_edge(pydot.Edge(1, 2))
    result = check_rank_constraints(graph)
    assert set(result.layers) == {"1", "2"}


def test_quoted_node_name_keeps_literal_colon() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { "a:b" -> c; }'
    )[0]
    result = check_rank_constraints(graph)
    assert set(result.layers) == {'"a:b"', "c"}
