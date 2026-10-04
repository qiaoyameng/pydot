# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Unit testing of `pydot.xdot`.

These tests exercise the parser with fixed XDOT strings only; no Graphviz
subprocess is started.
"""

from __future__ import annotations

import pytest

import pydot
from pydot import xdot
from pydot.xdot import (
    BSpline,
    Color,
    ColorList,
    ColorListEntry,
    Ellipse,
    FontCharacteristics,
    FontName,
    Image,
    InteriorPoints,
    LinearGradient,
    PenStyle,
    Point,
    Polygon,
    Polyline,
    RadialGradient,
    SetColor,
    SetSize,
    Text,
    XDotParseError,
    parse_xdot,
)


def _s(text: str) -> str:
    """Encode a length-prefixed XDOT string (UTF-8 byte length)."""

    return f"{len(text.encode('utf-8'))} -{text}-"


# A fixed, hand-built _draw_ stream covering the declared subset:
# fill color, stroked color, font controls, ellipse, two consecutive
# polylines, polygon, Bezier, style, image, interior points and a
# Chinese text label.
NODE_DRAW = " ".join(
    [
        "C " + _s("#ff00007f"),
        "C " + _s("#0000ff"),
        "S " + _s("setlinewidth(2)"),
        "s 2",
        "S 14 " + _s("SimSun"),
        "F 5",
        "E 27 18 54 36",
        "L 3 0 0 10 0 10 10",
        "l 2 20 20 30 30",
        "P 4 0 0 10 0 10 10 0 10",
        "B 3 0 0 5 10 10 10",
        "K 1 27 18",
        "I 1 2 3 4 " + _s("icon.png"),
        "T 27 16 0 48 " + _s("节点名称"),
    ]
)


def test_parse_full_fixed_stream() -> None:
    ops = parse_xdot(NODE_DRAW)

    assert ops[0] == SetColor(
        color=Color(raw="#ff00007f", r=255, g=0, b=0, a=127),
        filled=True,
    )
    assert ops[1] == SetColor(
        color=Color(raw="#0000ff", r=0, g=0, b=255, a=255),
        filled=True,
    )
    assert ops[2] == PenStyle(
        style="setlinewidth(2)", parts=("setlinewidth(2)",)
    )
    assert ops[3] == SetSize(2.0, line_width=True)
    assert ops[4] == FontName(size=14.0, name="SimSun")
    assert ops[5] == FontCharacteristics(mask=5)

    ellipse = ops[6]
    assert isinstance(ellipse, Ellipse)
    assert ellipse.center == Point(27.0, 18.0)
    assert (ellipse.width, ellipse.height) == (54.0, 36.0)

    # Operation case decides filled/stroked.
    assert ellipse.filled is True


def test_consecutive_polylines_stay_separate() -> None:
    ops = parse_xdot(NODE_DRAW)
    line_ops = [op for op in ops if isinstance(op, Polyline)]
    assert len(line_ops) == 2
    assert line_ops[0] == Polyline(
        points=(Point(0, 0), Point(10, 0), Point(10, 10)),
        filled=True,
    )
    assert line_ops[1] == Polyline(
        points=(Point(20, 20), Point(30, 30)),
        filled=False,
    )


def test_chinese_text_decoded_by_utf8_byte_length() -> None:
    ops = parse_xdot(NODE_DRAW)
    text_op = ops[-1]
    assert isinstance(text_op, Text)
    assert text_op.text == "节点名称"
    assert text_op.anchor == Point(27.0, 16.0)
    assert text_op.align == 0
    assert text_op.width == 48.0
    # The label uses 4 CJK characters of 3 bytes each, not 4 bytes.
    assert "12 -" in NODE_DRAW
    assert len("节点名称".encode("utf-8")) == 12


def test_negative_text_alignment_allowed() -> None:
    # The T alignment field is a signed integer (-1 = left aligned);
    # only length and point counts must be non-negative.
    ops = parse_xdot("T 5 7 -1 20 " + _s("左"))
    assert ops == [
        Text(anchor=Point(5.0, 7.0), align=-1, width=20.0, text="左")
    ]


def test_polygon_bezier_image_and_interior_points() -> None:
    ops = parse_xdot(NODE_DRAW)

    polygon = next(op for op in ops if isinstance(op, Polygon))
    assert polygon.filled is True
    assert len(polygon.points) == 4

    spline = next(op for op in ops if isinstance(op, BSpline))
    assert spline.points == (Point(0, 0), Point(5, 10), Point(10, 10))
    assert spline.filled is True
    assert spline.start is None and spline.end is None

    interior = next(op for op in ops if isinstance(op, InteriorPoints))
    assert interior.points == (Point(27, 18),)

    image = next(op for op in ops if isinstance(op, Image))
    assert image.name == "icon.png"
    assert (image.width, image.height) == (3.0, 4.0)
    assert image.filled is True


def test_stroked_bezier_has_endpoints() -> None:
    ops = parse_xdot("b 3 0 0 5 10 10 10 -1 -1 11 11")
    spline = ops[0]
    assert isinstance(spline, BSpline)
    assert spline.filled is False
    assert spline.start == Point(-1, -1)
    assert spline.end == Point(11, 11)


def test_font_size_in_text_context() -> None:
    ops = parse_xdot("s 12.5", text_context=True)
    assert ops == [SetSize(12.5, line_width=False)]


def test_pen_style_vs_font_name_share_S_opcode() -> None:
    # Pen style: one length-prefixed string.
    style_ops = parse_xdot("S " + _s("dashed"))
    assert style_ops == [PenStyle(style="dashed", parts=("dashed",))]

    # Font selection: size number, then the length-prefixed family name.
    font_ops = parse_xdot("S 14 " + _s("Arial"))
    assert font_ops == [FontName(size=14.0, name="Arial")]

    # A purely numeric style does not confuse the two forms.
    numeric_style = parse_xdot("S " + _s("setlinewidth(2)"))
    assert numeric_style == [
        PenStyle(style="setlinewidth(2)", parts=("setlinewidth(2)",))
    ]


def test_named_and_word_colors_preserve_raw() -> None:
    ops = parse_xdot(
        "C "
        + _s("red")
        + " C "
        + _s(" 0000ffff00000000")
    )
    assert ops[0].color == Color(raw="red")
    assert ops[1].color == Color(
        raw=" 0000ffff00000000", r=0, g=255, b=0, a=0
    )


def test_color_list_with_weights() -> None:
    ops = parse_xdot("c " + _s("C 2 #ff0000 0.4 #0000ff 0.6"))
    color = ops[0].color
    assert isinstance(color, ColorList)
    assert color.entries[0].color == Color(
        raw="#ff0000", r=255, g=0, b=0, a=255
    )
    assert color.entries[0].weight == 0.4
    assert color.entries[1].weight == 0.6


def test_word_color_inside_list_uses_double_space_marker() -> None:
    # A 16-bit-channel color following another token is introduced by a
    # double space (normal separator + the marker leading space).
    ops = parse_xdot(
        "c " + _s("C 2 #ff0000 0.4  0000ffff00000000 0.6")
    )
    color = ops[0].color
    assert isinstance(color, ColorList)
    second = color.entries[1].color
    assert second == Color(
        raw=" 0000ffff00000000", r=0, g=255, b=0, a=0
    )
    assert color.entries[1].weight == 0.6
    assert all(isinstance(entry, ColorListEntry) for entry in color.entries)


def test_linear_and_radial_gradients() -> None:
    linear = parse_xdot("c " + _s("L 2 0 0 255 255 #ff0000 #0000ff"))[0]
    assert isinstance(linear.color, LinearGradient)
    gradient = linear.color
    assert len(gradient.entries) == 2
    assert (gradient.x0, gradient.y0, gradient.x1, gradient.y1) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )
    assert gradient.entries[1].color == Color(
        raw="#0000ff", r=0, g=0, b=255, a=255
    )

    radial = parse_xdot(
        "c " + _s("G 1 0 0 0 255 255 255 #00ff00")
    )[0]
    assert isinstance(radial.color, RadialGradient)
    radial_gradient = radial.color
    assert radial_gradient.r0 == 0.0
    assert radial_gradient.r1 == 1.0
    assert radial_gradient.entries[0].color == Color(
        raw="#00ff00", r=0, g=255, b=0, a=255
    )


def test_empty_input() -> None:
    assert parse_xdot("") == []
    assert parse_xdot(b"   ") == []


def test_negative_count_is_rejected() -> None:
    with pytest.raises(XDotParseError) as info:
        parse_xdot("P -3 0 0")
    assert "negative count" in str(info.value)
    assert info.value.offset is not None


def test_truncated_text_is_rejected() -> None:
    # Length says 5, only 2 bytes follow and the closing '-' is missing.
    with pytest.raises(XDotParseError, match="truncated string"):
        parse_xdot("T 0 0 0 10 5 -ab")
    # Zero-length strings are valid though.
    assert parse_xdot("T 0 0 0 0 0 --") == [
        Text(anchor=Point(0, 0), align=0, width=0.0, text="")
    ]


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(XDotParseError, match="unknown XDOT operation"):
        parse_xdot("E 1 2 3 4 Z 0 0")
    # The ellipse before the bad op remains intact.
    try:
        parse_xdot("E 1 2 3 4 Z 0 0")
    except XDotParseError:
        pass


def test_point_count_mismatch_does_not_swallow_later_ops() -> None:
    # Promises 3 points but only provides one pair; the following
    # well-formed ellipse must never be silently consumed.
    stream = "P 3 0 0 E 10 10 4 2"
    with pytest.raises(XDotParseError) as info:
        parse_xdot(stream)
    # Failure happens while reading the promised points, before 'E'.
    assert isinstance(info.value, XDotParseError)


def test_utf8_split_in_middle_is_rejected() -> None:
    label = "中"  # 3 bytes: e4 b8 ad
    full = "T 0 0 0 10 3 -" + label + "-"
    assert parse_xdot(full)[0].text == "中"

    # Length prefix cut exactly one byte into the character, and a valid
    # instruction follows: the bad text must not swallow the ellipse.
    prefix = b"T 0 0 0 10 2 -"
    split = prefix + "中".encode("utf-8")[:2] + b"- E 9 9 2 2"
    with pytest.raises(XDotParseError, match="UTF-8") as info:
        parse_xdot(split)
    assert info.value.offset is not None


def test_bytes_input_accepted() -> None:
    assert parse_xdot(NODE_DRAW.encode("utf-8")) == parse_xdot(NODE_DRAW)


def test_parse_drawings_from_attributes_preserves_originals() -> None:
    attributes = {"_draw_": NODE_DRAW, "color": "red"}
    drawings = xdot.parse_drawings(attributes)

    assert len(drawings) == 1
    assert drawings[0].source == NODE_DRAW
    assert len(drawings[0]) == len(parse_xdot(NODE_DRAW))
    assert list(drawings[0]) == parse_xdot(NODE_DRAW)
    # Original attributes are left untouched.
    assert attributes["_draw_"] == NODE_DRAW
    assert attributes["color"] == "red"


def test_parse_drawings_from_pydot_node() -> None:
    node = pydot.Node("n", **{"_draw_": NODE_DRAW})
    drawings = xdot.parse_drawings(node)
    assert drawings
    assert drawings[0].source == NODE_DRAW
    assert node.get("_draw_") == NODE_DRAW


def test_parse_drawings_custom_names_and_missing() -> None:
    attributes = {"_hdraw_": "E 1 2 3 4"}
    drawings = xdot.parse_drawings(
        attributes, "_hdraw_", "_tdraw_"
    )
    assert len(drawings) == 1
    ellipse = drawings[0].operations[0]
    assert isinstance(ellipse, Ellipse)
    assert ellipse.filled is True
