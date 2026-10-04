# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Tests for spatial hit indexes on laid-out Dot graphs."""

from __future__ import annotations

import pytest

import pydot


def test_rectangle_boundary_uses_inches_scale_and_translate() -> None:
    graph = pydot.Dot()
    graph.add_node(
        pydot.Node(
            "box",
            shape="box",
            pos="0,0",
            width="1in",
            height="0.5in",
        )
    )
    index = graph.build_layout_hit_index(
        scale=(2.0, 2.0), translate=(10.0, 20.0)
    )

    assert index.hit(83.0, 56.0, tolerance=0.5) is None
    hit = index.hit(83.0, 56.0, tolerance=1.0)
    assert hit is not None
    assert hit.name == "box"
    assert hit.kind == "rectangle"
    assert hit.distance == pytest.approx(1.0)
    assert index.hit(10.0, 20.0).distance == 0.0


def test_scale_and_translate_accept_separate_axes() -> None:
    graph = pydot.Dot()
    graph.add_node(
        pydot.Node(
            "box",
            shape="rectangle",
            pos="10,10",
            width=1,
            height=1,
        )
    )
    index = graph.build_layout_hit_index(
        scale=(2.0, -1.0), translate=(5.0, 20.0)
    )
    assert index.hit(25.0, 10.0).distance == 0.0
    assert index.hit(98.0, 10.0, tolerance=0.9) is None


def test_ellipse_boundary_and_inside_distance() -> None:
    graph = pydot.Dot()
    graph.add_node(
        pydot.Node(
            "oval",
            shape="ellipse",
            pos="0,0",
            width=2.0,
            height=1.0,
        )
    )
    index = graph.build_layout_hit_index()

    assert index.nearest(100.0, 0.0).distance == pytest.approx(28.0)
    assert index.nearest(0.0, 100.0).distance == pytest.approx(64.0)
    assert index.hit(0.0, 0.0).distance == 0.0
    assert index.hit(25.0, 25.0).distance == pytest.approx(0.0)


def test_curved_edge_is_not_hit_from_its_bbox_interior() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("a", pos="0,0", width=0.1, height=0.1))
    graph.add_node(pydot.Node("b", pos="20,0", width=0.1, height=0.1))
    graph.add_edge(
        pydot.Edge(
            "a",
            "b",
            pos="0,0 0,20 20,20 20,0",
        )
    )
    index = graph.build_layout_hit_index(cell_size=20.0)

    on_curve = index.hit(10.0, 15.0, tolerance=0.2)
    assert on_curve is not None
    assert on_curve.kind == "edge"
    assert on_curve.distance == pytest.approx(0.0, abs=1e-6)

    assert index.hit(10.0, 13.0, tolerance=1.0) is None
    outside = index.hit(10.0, 13.0, tolerance=10.0)
    assert outside is not None
    assert outside.kind == "edge"
    assert outside.distance == pytest.approx(2.0, abs=1e-6)


def test_multi_segment_cubic_edge() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("a", pos="0,0", width=0.1, height=0.1))
    graph.add_node(pydot.Node("b", pos="40,0", width=0.1, height=0.1))
    graph.add_edge(
        pydot.Edge(
            "a",
            "b",
            pos=(
                "e,40,0 0,0 0,20 20,20 20,0 "
                "30,0 40,-20 40,0"
            ),
        )
    )
    index = graph.build_layout_hit_index()
    hit = index.hit(30.0, -10.0, tolerance=2.0)
    assert hit is not None
    assert hit.kind == "edge"


def test_tolerance_changes_multiple_overlap_ordering() -> None:
    graph = pydot.Dot()
    graph.add_node(
        pydot.Node("first", shape="box", pos="-40,0", width=1, height=1)
    )
    graph.add_node(
        pydot.Node("second", shape="box", pos="40,0", width=1, height=1)
    )
    index = graph.build_layout_hit_index()

    assert index.hits(0.0, 0.0, tolerance=3.0) == []
    near = index.hits(0.0, 0.0, tolerance=5.0)
    assert [hit.name for hit in near] == ["first", "second"]
    assert near[0].distance == pytest.approx(4.0)
    assert near[1].distance == pytest.approx(4.0)
    assert near[0].order < near[1].order

    assert [hit.name for hit in index.hits(0.0, 0.0, tolerance=4.0)] == [
        "first",
        "second",
    ]


def test_missing_or_corrupt_layout_identifies_object() -> None:
    graph = pydot.Dot()
    graph.add_node(pydot.Node("bad", shape="box"))
    with pytest.raises(pydot.LayoutHitError, match="node 'bad'") as exc:
        graph.build_layout_hit_index()
    assert exc.value.target.get_name() == "bad"

    graph.del_node("bad")
    graph.add_node(
        pydot.Node("good", shape="box", pos="0,0", width=1, height=1)
    )
    graph.add_edge(pydot.Edge("good", "missing", pos="bad,pos"))
    with pytest.raises(pydot.LayoutHitError, match="good -> missing"):
        graph.build_layout_hit_index()


def test_sparse_graph_query_does_not_scan_all_objects() -> None:
    graph = pydot.Dot()
    for index in range(200):
        x = float(index * 100)
        graph.add_node(
            pydot.Node(
                f"n{index}",
                shape="box",
                pos=f"{x},0",
                width=1,
                height=1,
            )
        )
        graph.add_edge(
            pydot.Edge(
                f"n{index}",
                f"n{index}",
                pos=f"{x},0 {x},10 {x + 20},10 {x + 20},0",
            )
        )

    index = graph.build_layout_hit_index()
    hit = index.hit(15000.0, 0.0, tolerance=2.0)
    assert hit is not None
    assert hit.name == "n150"
    assert index.candidate_accesses < 10
