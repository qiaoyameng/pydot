# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the XDOT drawing-instruction parser.

The XDOT blobs are fixed fixtures stored under
``test/my_tests/xdot/``; no Graphviz subprocess is started.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pydot
from pydot import xdot
from pydot.exceptions import (
    XDotError,
    XDotNegativeLengthError,
    XDotPointCountError,
    XDotTruncatedError,
    XDotUnknownOpError,
)

FIXTURES = Path(__file__).parent / "my_tests" / "xdot"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


OPCODES_FULL = [
    "c",
    "C",
    "L",
    "E",
    "e",
    "p",
    "P",
    "B",
    "b",
    "F",
    "T",
    "S",
    "K",
]


def test_full_drawing_every_opcode() -> None:
    """All supported opcodes decode from the fixed full fixture."""
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))

    assert drawing.ok
    assert drawing.errors == []
    assert [op.opcode for op in drawing.operations] == OPCODES_FULL
    assert len(drawing) == 13
    assert [op.opcode for op in drawing] == OPCODES_FULL


def test_ellipse_operations() -> None:
    """Both stroked ``E`` and filled ``e`` ellipses decode."""
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))
    stroked, filled = drawing.operations[3], drawing.operations[4]

    assert isinstance(stroked, xdot.Ellipse)
    assert stroked.opcode == "E"
    assert (stroked.x, stroked.y) == (27.0, 18.0)
    assert (stroked.width, stroked.height) == (54.0, 36.0)
    assert stroked.filled is False

    assert isinstance(filled, xdot.Ellipse)
    assert filled.opcode == "e"
    assert filled.filled is True


def test_consecutive_paths() -> None:
    """Filled polygon, polyline and Bezier instructions follow each other."""
    drawing = xdot.parse_drawing(fixture("paths.xdot"))

    assert drawing.ok
    polygon, polyline, bezier_stroked, second_polygon = drawing.operations

    assert isinstance(polygon, xdot.Polygon)
    assert polygon.opcode == "p"
    assert polygon.filled is True
    assert polygon.points == (
        xdot.Point(0.0, 0.0),
        xdot.Point(1.0, 1.0),
        xdot.Point(1.0, 0.0),
    )

    assert isinstance(polyline, xdot.Polyline)
    assert polyline.opcode == "P"
    assert polyline.filled is False
    assert len(polyline.points) == 4

    assert isinstance(bezier_stroked, xdot.Bezier)
    assert bezier_stroked.opcode == "B"
    assert bezier_stroked.filled is False
    assert bezier_stroked.points == (
        xdot.Point(5.0, 5.0),
        xdot.Point(9.0, 9.0),
    )

    assert isinstance(second_polygon, xdot.Polygon)
    assert second_polygon.opcode == "p"
    assert second_polygon.points == (
        xdot.Point(1.0, 1.0),
        xdot.Point(2.0, 2.0),
    )


def test_chinese_text_is_utf8_decoded() -> None:
    """The length counts encoded bytes, so multibyte text decodes."""
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))
    font, text = drawing.operations[9], drawing.operations[10]

    assert isinstance(font, xdot.Font)
    assert font.size == 14.0
    assert font.name == "SimSun"

    assert isinstance(text, xdot.Text)
    assert text.text == "节点"
    assert len("节点".encode()) == 6
    assert (text.x, text.y, text.align) == (27.0, 20.0, 0)


def test_styles_and_line_style() -> None:
    """Font, style flags and line dash patterns decode."""
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))
    line_style = drawing.operations[2]
    style = drawing.operations[11]

    assert isinstance(line_style, xdot.LineStyle)
    assert line_style.width == 1.0
    assert line_style.dashes == ()

    assert isinstance(style, xdot.Style)
    assert style.flags == ("filled",)


def test_dashed_line_style() -> None:
    """Dash segments collect up to the terminating zero."""
    drawing = xdot.parse_drawing("L 2.5 5 3 0")
    line_style = drawing.operations[0]

    assert isinstance(line_style, xdot.LineStyle)
    assert line_style.width == 2.5
    assert line_style.dashes == (5.0, 3.0)


def test_plain_and_gradient_colors() -> None:
    """Plain colours stay verbatim; gradient descriptions are parsed."""
    drawing = xdot.parse_drawing(fixture("colors.xdot"))
    plain, gradient, rgba = drawing.operations

    assert isinstance(plain, xdot.Color)
    assert plain.opcode == "C"
    assert plain.color == "black"
    assert plain.gradient is None

    assert isinstance(gradient, xdot.Color)
    assert gradient.opcode == "c"
    assert gradient.fill is True
    assert gradient.color == "45:#ff0000:#0000ff"
    assert gradient.gradient is not None
    assert gradient.gradient.angle == 45.0
    assert gradient.gradient.colors == ("#ff0000", "#0000ff")
    assert gradient.gradient.stops == (
        xdot.ColorStop(0.0, "#ff0000"),
        xdot.ColorStop(1.0, "#0000ff"),
    )

    assert isinstance(rgba, xdot.Color)
    assert rgba.color == "#aabbccff"
    assert rgba.gradient is None


def test_gradient_with_weight() -> None:
    """An optional trailing weight is kept on the gradient description."""
    token = "45:#ff0000:#0000ff:0.3"
    body = ("-" + token).encode("utf-8")
    drawing = xdot.parse_drawing(f"C {len(body)} -{token}-")

    assert drawing.ok
    gradient = drawing.operations[0]
    assert isinstance(gradient, xdot.Color)
    assert gradient.gradient is not None
    assert gradient.gradient.weight == 0.3


def test_gradient_fill_operation() -> None:
    """The ``K`` opcode decodes endpoints and colour stops."""
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))
    gradient_fill = drawing.operations[12]

    assert isinstance(gradient_fill, xdot.GradientFill)
    assert gradient_fill.p0 == xdot.Point(0.0, 18.0)
    assert gradient_fill.p1 == xdot.Point(54.0, 18.0)
    assert [stop.fraction for stop in gradient_fill.stops] == [0.0, 1.0]
    assert [stop.color for stop in gradient_fill.stops] == [
        "#ff0000ff",
        "#0000ffff",
    ]


def test_escaped_space_in_string() -> None:
    r"""Graphviz escapes spaces in arguments as ``\040``."""
    token = r"New\040Roman"
    body = ("-" + token).encode("utf-8")
    drawing = xdot.parse_drawing(f"F 14 {len(body)} -{token}-")

    assert drawing.ok
    font = drawing.operations[0]
    assert isinstance(font, xdot.Font)
    assert font.name == "New Roman"


@pytest.mark.parametrize(
    ("fixture_name", "expected_error", "surviving"),
    [
        ("bad_mid_utf8.xdot", XDotTruncatedError, ["E"]),
        ("bad_truncated_text.xdot", XDotTruncatedError, ["E"]),
        ("bad_negative_length.xdot", XDotNegativeLengthError, ["E"]),
        ("bad_unknown_op.xdot", XDotUnknownOpError, ["P"]),
        ("bad_point_count.xdot", XDotPointCountError, ["B"]),
    ],
)
def test_bad_instruction_keeps_following_instructions(
    fixture_name: str,
    expected_error: type[XDotError],
    surviving: list[str],
) -> None:
    """Each bad input is reported but never swallows later instructions."""
    drawing = xdot.parse_drawing(fixture(fixture_name))

    assert drawing.ok is False
    assert len(drawing.errors) == 1
    assert isinstance(drawing.errors[0], expected_error)
    assert [op.opcode for op in drawing.operations] == surviving


def test_bad_mid_utf8_length_is_byte_based() -> None:
    """A declared byte length splits a Chinese character in two.

    Lengths count encoded bytes, so declaring 4 of the 6 bytes of
    ``节点`` lands in the middle of the second character.
    """
    raw = fixture("bad_mid_utf8.xdot")
    full = "节点".encode()
    declared = 4

    assert len(full) == 6
    assert declared < len(full)
    with pytest.raises(UnicodeDecodeError):
        full[:declared].decode("utf-8")

    drawing = xdot.parse_drawing(raw)
    assert isinstance(drawing.errors[0], XDotTruncatedError)
    assert "UTF-8" in str(drawing.errors[0])


def test_strict_mode_raises() -> None:
    """Strict parsing raises the first malformed-instruction error."""
    with pytest.raises(XDotUnknownOpError):
        xdot.parse_drawing(fixture("bad_unknown_op.xdot"), strict=True)


def test_strict_mode_good_input() -> None:
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"), strict=True)
    assert drawing.ok


def test_error_positions_reported() -> None:
    """Every error carries the byte offset of the bad instruction."""
    drawing = xdot.parse_drawing(fixture("bad_point_count.xdot"))
    error = drawing.errors[0]
    assert error.position >= 0
    assert "at byte" in str(error)


def test_empty_input() -> None:
    assert xdot.parse_drawing("").operations == []
    assert xdot.parse_drawing(None).operations == []
    assert xdot.parse_drawing(b"").ok


def test_parse_operations_helper() -> None:
    operations = xdot.parse_operations("E 1 2 3 4")
    assert len(operations) == 1
    assert isinstance(operations[0], xdot.Ellipse)


def test_drawings_from_pydot_object_preserves_attribute() -> None:
    """Original ``_draw_`` attribute stays available unchanged."""
    value = fixture("full_draw.xdot").decode("utf-8")
    node = pydot.Node("n", **{"_draw_": value})

    drawing = xdot.drawings_from(node, "_draw_")

    assert drawing.ok
    assert len(drawing.operations) == 13
    assert node.get_attributes()["_draw_"] == value


def test_drawings_from_raw_string() -> None:
    drawing = xdot.drawings_from("e 1 2 3 4")
    assert isinstance(drawing.operations[0], xdot.Ellipse)
    assert drawing.operations[0].filled is True


def test_drawings_from_ldraw_attribute() -> None:
    """Edge/label drawings come from ``_ldraw_`` too."""
    edge = pydot.Edge("a", "b", **{"_ldraw_": "P 2 0 0 1 1"})
    drawing = xdot.drawings_from(edge, "_ldraw_")
    assert drawing.ok
    assert isinstance(drawing.operations[0], xdot.Polyline)


def test_bytes_input_accepted() -> None:
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))
    assert drawing.ok


def test_all_operations_are_xdot_ops() -> None:
    drawing = xdot.parse_drawing(fixture("full_draw.xdot"))
    assert all(isinstance(op, xdot.XDotOp) for op in drawing.operations)
