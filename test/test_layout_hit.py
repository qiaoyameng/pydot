from __future__ import annotations

import math
from pathlib import Path

import pytest

import pydot


def laid_out_graph() -> pydot.Dot:
    return pydot.graph_from_dot_data(
        """
        digraph G {
            node [label="\\N"];
            box [
                shape=box,
                pos="100,100",
                width="1in",
                height="0.5in"
            ];
            oval [
                shape=ellipse,
                pos="300,100",
                width="72pt",
                height="36pt"
            ];
            box -> oval [
                pos="e,264,100 136,100 170,73.333333 230,73.333333 264,100"
            ];
        }
        """
    )[0]


def test_rectangle_and_ellipse_boundaries() -> None:
    index = pydot.LayoutIndex(laid_out_graph())

    assert index.hit_at(136, 100).get_name() == "box"
    assert index.hit_at(137, 115, tolerance=0.5) is None
    assert index.hit_at(100, 118).get_name() == "box"
    assert index.hit_at(100, 119, tolerance=0.5) is None

    oval = index.hit_at(300 + 36 / math.sqrt(2), 100 + 18 / math.sqrt(2))
    assert oval.get_name() == "oval"
    assert index.hit_at(337, 119, tolerance=0.5) is None


def test_coordinate_scale_and_translation_units() -> None:
    index = pydot.LayoutIndex(
        laid_out_graph(),
        scale=(2.0, -1.0),
        translate=(10.0, 300.0),
    )

    hit = index.hit_at(2 * 100 + 10, -100 + 300)
    assert hit.get_name() == "box"

    assert index.hit_at(2 * 136 + 10, -100 + 300).get_name() == "box"
    assert index.hit_at(2 * 137 + 10, -115 + 300, tolerance=0.5) is None


def test_bezier_edge_uses_curve_not_bounding_box() -> None:
    index = pydot.LayoutIndex(laid_out_graph())

    edge = index.hit_at(200, 81, tolerance=1.5)
    assert isinstance(edge, pydot.Edge)
    assert (edge.get_source(), edge.get_destination()) == ("box", "oval")

    assert index.hit_at(200, 90, tolerance=1.0) is None
    hits = index.hit_test(200, 90, tolerance=11.0)
    assert [hit.kind for hit in hits] == ["edge"]
    assert 9.5 < hits[0].distance < 10.5


def test_tolerance_changes_inside_and_outside_matches() -> None:
    index = pydot.LayoutIndex(laid_out_graph())

    assert index.hit_at(200, 82.0) is None
    assert isinstance(index.hit_at(200, 78.0, tolerance=2.0), pydot.Edge)
    assert isinstance(index.hit_at(200, 76.0, tolerance=4.0), pydot.Edge)
    assert index.hit_at(200, 76.0, tolerance=3.0) is None


def test_overlapping_hits_use_distance_then_declaration_order() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            first [shape=box, pos="100,100", width="1", height="1"];
            second [shape=box, pos="100,100", width="1", height="1"];
            third [shape=box, pos="100,103", width="1", height="1"];
        }
        """
    )[0]
    index = pydot.LayoutIndex(graph)

    hits = index.hit_test(100, 101)
    assert [hit.object.get_name() for hit in hits] == [
        "first",
        "second",
        "third",
    ]
    assert hits[0].distance == 0
    assert hits[2].distance == 0


def test_sparse_graph_query_does_not_visit_all_objects() -> None:
    graph = pydot.Dot()
    count = 256
    for number in range(count):
        graph.add_node(
            pydot.Node(
                f"n{number}",
                shape="box",
                pos=f"{number * 256},0",
                width="1",
                height="1",
            )
        )

    index = pydot.LayoutIndex(graph, cell_size=128)

    assert index.count_candidates(10 * 256, 0) == 1
    assert index.count_candidates(10 * 256, 512, tolerance=4) == 0
    assert len(index.entries) == count


def test_malformed_layout_is_attributed_to_object() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { bad [shape=box, pos="oops", width="1", height="1"]; }'
    )[0]
    with pytest.raises(pydot.LayoutError) as info:
        pydot.LayoutIndex(graph)
    assert info.value.object_type == "node"
    assert info.value.name == "bad"
    assert info.value.source.obj_dict is graph.get_node("bad")[0].obj_dict

    graph = pydot.graph_from_dot_data(
        'digraph G { a [pos="0,0"]; b [pos="10,0"]; a -> b [pos="1,0 2,0"]; }'
    )[0]
    with pytest.raises(pydot.LayoutError) as edge_info:
        pydot.LayoutIndex(graph)
    assert edge_info.value.object_type == "edge"
    assert edge_info.value.name == "a -> b"
    assert isinstance(edge_info.value.source, pydot.Edge)


def test_missing_layout_is_reported_with_the_affected_object() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            laid_out [shape=box, pos="10,10", width="1", height="1"];
            implicit;
            a [pos="0,0", width="1", height="1"];
            b [pos="20,0", width="1", height="1"];
            a -> b;
        }
        """
    )[0]

    with pytest.raises(pydot.LayoutError) as info:
        pydot.LayoutIndex(graph)
    assert info.value.object_type == "node"
    assert info.value.name == "implicit"
    assert isinstance(info.value.source, pydot.Node)

    graph.del_node("implicit")
    with pytest.raises(pydot.LayoutError) as edge_info:
        pydot.LayoutIndex(graph)
    assert edge_info.value.object_type == "edge"
    assert edge_info.value.name == "a -> b"


def test_multisegment_bezier_and_subgraph_declaration_order() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            subgraph cluster_s {
                a [
                    shape=box,
                    pos="0,0",
                    width="1",
                    height="1"
                ];
                b [
                    shape=box,
                    pos="300,0",
                    width="1",
                    height="1"
                ];
                a -> b [
                    pos="s,36,0 60,0 80,40 100,40 120,40 140,40 264,0 e,264,0"
                ];
            }
        }
        """
    )[0]
    index = pydot.LayoutIndex(graph)

    assert index.hit_at(100, 40, tolerance=0.1).get_source() == "a"
    assert index.hit_at(100, 25, tolerance=1.0) is None
    assert [entry.order for entry in index.entries] == [0, 1, 2]


def test_unsupported_shape_with_position_is_layout_error() -> None:
    graph = pydot.graph_from_dot_data(
        'digraph G { bad [shape=diamond, pos="1,2"]; }'
    )[0]
    with pytest.raises(pydot.LayoutError) as info:
        pydot.LayoutIndex(graph)
    assert info.value.name == "bad"
    assert "shape" in str(info.value).lower()


def test_unmarked_bezier_negative_scale_and_real_layout() -> None:
    graph = pydot.graph_from_dot_data(
        """
        digraph G {
            a [shape=box, pos="0,0", width="1", height="1"];
            b [shape=ellipse, pos="100,0", width="1", height="1"];
            a -> b [pos="0,0 25,50 75,50 100,0"];
        }
        """
    )[0]
    index = pydot.LayoutIndex(graph, scale=(2.0, -2.0), translate=(5.0, 5.0))

    assert index.hit_at(5.0, -5.0).get_name() == "a"
    assert isinstance(index.hit_at(105.0, -70.0, tolerance=1.0), pydot.Edge)

    graph_path = Path(__file__).parent / "my_tests" / "parsing_test.dot"
    laid_out = pydot.graph_from_dot_data(graph_path.read_text())[0]
    laid_out_index = pydot.LayoutIndex(laid_out)
    assert len(laid_out_index.entries) == 41
    assert laid_out_index.count_candidates(99, 90) == 1
    assert isinstance(
        laid_out_index.hit_at(180, 130, tolerance=1.0),
        pydot.Edge,
    )

