# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Parser for Graphviz XDOT drawing instructions.

The layout programs (``dot``, ``neato`` and so on) attach pre-rendered
drawing instructions to graphs, nodes and edges through the attributes
``_draw_``, ``_ldraw_``, ``_hdraw_``, ``_tdraw_`` and the corresponding
``*ldraw*`` variants.  This module turns such an instruction string into a
list of typed operations that can be replayed on an application-owned
canvas.

The subset implemented here is the one explicitly declared in the XDOT
specification:

* ``E``/``e`` ellipses,
* ``P``/``p`` polygons,
* ``L``/``l`` polylines,
* ``B``/``b`` (B)ezier splines,
* ``T`` text together with the ``F`` font-characteristics, ``S``
  font-name and ``s`` font-size/line-width controls,
* ``C``/``c`` colors (including color lists and ``L``/``G`` gradients),
* ``S`` pen styles, ``I``/``i`` images and ``K`` polygon interior points.

Length-prefixed values are read by their *encoded byte* length.  Malformed
input -- negative counts, truncated or mis-decoded text, unknown operations
or a point count that does not match the available data -- raises
:class:`XDotParseError`; the parser never silently skips ahead and swallows
instructions that follow the bad one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "Point",
    "Color",
    "ColorListEntry",
    "ColorList",
    "LinearGradient",
    "RadialGradient",
    "ColorSpec",
    "DrawOp",
    "Ellipse",
    "Polygon",
    "Polyline",
    "BSpline",
    "Text",
    "FontCharacteristics",
    "FontName",
    "PenStyle",
    "SetSize",
    "SetColor",
    "InteriorPoints",
    "Image",
    "XDotDrawing",
    "XDotParseError",
    "parse_xdot",
    "parse_drawings",
]


@dataclass(frozen=True)
class Point:
    """A single coordinate pair in Graphviz points (1/72 inch)."""

    x: float
    y: float


@dataclass(frozen=True)
class Ellipse:
    """Filled (``E``) or stroked (``e``) ellipse."""

    center: Point
    width: float
    height: float
    filled: bool


@dataclass(frozen=True)
class Polygon:
    """Filled (``P``) or stroked (``p``) polygon."""

    points: tuple[Point, ...]
    filled: bool


@dataclass(frozen=True)
class Polyline:
    """Filled (``L``) or stroked (``l``) polyline."""

    points: tuple[Point, ...]
    filled: bool


@dataclass(frozen=True)
class BSpline:
    """Filled (``B``) or stroked (``b``) cubic Bezier spline.

    The control polygon groups cubic segments of three points; stroked
    splines additionally carry explicit start and end points.
    """

    points: tuple[Point, ...]
    filled: bool
    start: Point | None = None
    end: Point | None = None


@dataclass(frozen=True)
class Text:
    """UTF-8 text positioned by the layout engine."""

    anchor: Point
    align: int
    width: float
    text: str


@dataclass(frozen=True)
class FontCharacteristics:
    """Bitmask of bold/italic/underline/strikethrough flags (``F``)."""

    mask: int


@dataclass(frozen=True)
class FontName:
    """Font selection: size in points and a font family name (``S``)."""

    size: float
    name: str


@dataclass(frozen=True)
class PenStyle:
    """Comma-separated pen style string (``S``), e.g. ``setlinewidth(2)``."""

    style: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class SetSize:
    """Line width (``s`` in draw context) or font size (text context)."""

    value: float
    line_width: bool


@dataclass(frozen=True)
class Image:
    """Filled (``I``) or stroked (``i``) external image reference."""

    corner: Point
    width: float
    height: float
    name: str
    filled: bool


@dataclass(frozen=True)
class InteriorPoints:
    """Interior points used for edge-label placement (``K``)."""

    points: tuple[Point, ...]


@dataclass(frozen=True)
class Color:
    """A plain color.

    ``r``, ``g``, ``b`` and ``a`` are normalized 0-255 channel values when
    the color was given numerically; named colors leave them ``None``.
    The original color string is always available as ``raw``.
    """

    raw: str
    r: int | None = None
    g: int | None = None
    b: int | None = None
    a: int | None = None


@dataclass(frozen=True)
class ColorListEntry:
    """One weighted color within a :class:`ColorList` or gradient."""

    color: ColorSpec
    weight: float | None = None


@dataclass(frozen=True)
class ColorList:
    """A ``C`` color list attribute value (e.g. layered fills)."""

    entries: tuple[ColorListEntry, ...]


@dataclass(frozen=True)
class LinearGradient:
    """An ``L`` linear gradient between ``x0``/``y0`` and ``x1``/``y1``."""

    entries: tuple[ColorListEntry, ...]
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class RadialGradient:
    """A ``G`` radial gradient between two circles."""

    entries: tuple[ColorListEntry, ...]
    x0: float
    y0: float
    r0: float
    x1: float
    y1: float
    r1: float


ColorSpec = Color | ColorList | LinearGradient | RadialGradient


@dataclass(frozen=True)
class SetColor:
    """Pen (``C``) or fill (``c``) color selection."""

    color: ColorSpec
    filled: bool


DrawOp = (
    Ellipse
    | Polygon
    | Polyline
    | BSpline
    | Text
    | FontCharacteristics
    | FontName
    | PenStyle
    | SetSize
    | SetColor
    | InteriorPoints
    | Image
)


@dataclass(frozen=True)
class XDotDrawing:
    """A parsed XDOT attribute together with the original string."""

    operations: tuple[DrawOp, ...]
    source: str

    def __iter__(self) -> Iterator[DrawOp]:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)


class XDotParseError(ValueError):
    """Raised when an XDOT instruction stream cannot be parsed.

    ``offset`` is the byte position at which parsing failed.  The stream is
    byte-oriented and length prefixed, so stopping here leaves the
    already-parsed operations intact; later instructions are not consumed.
    """

    def __init__(self, message: str, offset: int) -> None:
        self.offset = offset
        super().__init__(f"{message} at byte {offset}")


class _Scanner:
    """Byte-oriented scanner over one XDOT instruction stream."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0

    # -- low level helpers ------------------------------------------------

    def _skip_space(self) -> None:
        data = self._data
        pos = self.pos
        length = len(data)
        while pos < length and data[pos : pos + 1].isspace():
            pos += 1
        self.pos = pos

    def eof(self) -> bool:
        self._skip_space()
        return self.pos >= len(self._data)

    def peek_op(self) -> int:
        self._skip_space()
        if self.pos >= len(self._data):
            raise XDotParseError("unexpected end of instructions", self.pos)
        return self._data[self.pos]

    def read_op(self) -> str:
        op = self.peek_op()
        self.pos += 1
        return chr(op)

    def read_double(self) -> float:
        self._skip_space()
        start = self.pos
        data = self._data
        length = len(data)
        if start >= length:
            raise XDotParseError("expected number, got end of input", start)
        pos = start
        if data[pos] in (0x2B, 0x2D):  # '+' / '-'
            pos += 1
        seen_digit = False
        seen_dot = False
        seen_exp = False
        while pos < length:
            ch = data[pos]
            if 0x30 <= ch <= 0x39:
                seen_digit = True
                pos += 1
            elif ch == 0x2E and not seen_dot and not seen_exp:
                seen_dot = True
                pos += 1
            elif ch in (0x65, 0x45) and seen_digit and not seen_exp:
                exp_pos = pos + 1
                if exp_pos < length and data[exp_pos] in (0x2B, 0x2D):
                    exp_pos += 1
                if exp_pos < length and 0x30 <= data[exp_pos] <= 0x39:
                    seen_exp = True
                    seen_dot = True
                    pos = exp_pos
                else:
                    break
            else:
                break
        token = data[start:pos]
        if not seen_digit:
            raise XDotParseError(
                f"expected number, got {token[:16]!r}", start
            )
        self.pos = pos
        try:
            return float(token.decode("ascii"))
        except ValueError as exc:
            raise XDotParseError(
                f"invalid number {token[:16]!r}", start
            ) from exc

    def read_count(self) -> int:
        """Read a non-negative integer count (point counts, lengths)."""

        offset = self.pos
        value = self.read_double()
        if value != int(value):
            raise XDotParseError(
                f"expected integer count, got {value}", offset
            )
        integer = int(value)
        if integer < 0:
            raise XDotParseError(f"negative count {integer}", offset)
        return integer

    def read_point(self) -> Point:
        return Point(self.read_double(), self.read_double())

    def read_points(self, count: int) -> tuple[Point, ...]:
        return tuple(self.read_point() for _ in range(count))

    def read_encoded_string(self) -> str:
        """Read a ``n -<bytes>-`` length-prefixed UTF-8 string."""

        self._skip_space()
        length_start = self.pos
        count = self.read_count()
        self._skip_space()
        if self.pos >= len(self._data) or self._data[self.pos] != ord("-"):
            raise XDotParseError(
                "expected '-' before length-prefixed string", self.pos
            )
        begin = self.pos + 1
        end = begin + count
        if end >= len(self._data) or self._data[end] != ord("-"):
            raise XDotParseError(
                f"truncated string of length {count}", length_start
            )
        raw = self._data[begin:end]
        self.pos = end + 1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise XDotParseError(
                "length-prefixed string ends inside a UTF-8 sequence",
                length_start,
            ) from exc

    # -- color tokens -----------------------------------------------------

    def read_bare_token(self, *, keep_space_marker: bool = False) -> str:
        """Read the next whitespace-delimited ASCII token verbatim.

        Word-sized channel colors print a deliberate extra leading space
        (``" RRRRGGGGBBBBAAAA"``).  With ``keep_space_marker`` that space
        -- either at the body start or as the second space between
        tokens -- is kept as the first character of the returned token.
        """

        data = self._data
        length = len(data)
        # Find the end of the whitespace run that precedes the token.
        run_start = self.pos
        run_end = run_start
        while run_end < length and data[run_end : run_end + 1].isspace():
            run_end += 1
        run = data[run_start:run_end]
        marker_at_start = run_start == 0 and run[:1] == b" "
        marker_between = run.count(b" ") >= 2
        marked = keep_space_marker and (marker_at_start or marker_between)
        # First byte of the returned token: with a marker it is the marker
        # space itself; otherwise it is the first byte after the run.
        if marked:
            value_start = 0 if marker_at_start else run_end - 1
            value_end = value_start + 1  # skip the marker space
        else:
            value_start = run_end
            value_end = value_start
        while value_end < length and not data[
            value_end : value_end + 1
        ].isspace():
            value_end += 1
        if value_end == value_start:
            raise XDotParseError(
                "expected color token, got end of input", run_end
            )
        self.pos = value_end
        return data[value_start:value_end].decode("ascii")

    def next_token_is_number(self) -> bool:
        saved_pos = self.pos
        self._skip_space()
        if self.pos >= len(self._data):
            self.pos = saved_pos
            return False
        ch = self._data[self.pos]
        result = 0x30 <= ch <= 0x39 or ch in (0x2E, 0x2B, 0x2D)
        self.pos = saved_pos
        return result

    def at_font_name_selection(self) -> bool:
        """True for ``<size> <length> -<name>-`` (as opposed to a style).

        A pen style looks like ``<length> -<style>-`` (one number), while
        a font selection adds the size, giving two numbers before the dash.
        """

        probe = _Scanner(self._data[self.pos :])
        try:
            probe.read_double()
            probe.read_count()
            probe._skip_space()
        except XDotParseError:
            return False
        probe._skip_space()
        return probe.pos < len(probe._data) and probe._data[
            probe.pos
        ] == ord("-")


def _hex_color(raw: str, digits: str, word: bool, offset: int) -> Color:
    """Decode a numeric XDOT color token.

    Graphviz prints 8-bit channels as ``#RRGGBB[AA]`` and 16-bit channels
    as a space-prefixed ``RRRRGGGGBBBB[AAAA]`` (the leading space keeps it
    unambiguous from color names).
    """

    if not digits or not all(
        ch in "0123456789abcdefABCDEF" for ch in digits
    ):
        raise XDotParseError(f"invalid hexadecimal color {raw!r}", offset)
    if len(digits) == (16 if word else 8):
        width = 4
    elif not word and len(digits) == 6:
        width = 3
    else:
        raise XDotParseError(f"invalid hexadecimal color {raw!r}", offset)
    step = len(digits) // width
    channels = [
        int(digits[index * step : (index + 1) * step], 16)
        for index in range(width)
    ]
    max_value = (1 << (4 * step)) - 1
    channels = [round(value * 255 / max_value) for value in channels]
    if width == 3:
        channels.append(255)
    red, green, blue, alpha = channels
    return Color(raw=raw, r=red, g=green, b=blue, a=alpha)


class _ColorReader:
    """Parse the body of a ``C``/``c`` color operation."""

    def __init__(self, payload: str) -> None:
        self.scanner = _Scanner(payload.encode("utf-8"))

    def read_color(self) -> ColorSpec:
        offset = self.scanner.pos
        token = self.scanner.read_bare_token(keep_space_marker=True)
        first = token[0] if token else ""
        if first == "#":
            return _hex_color(token, token[1:], False, offset)
        if first == " ":
            return _hex_color(token, token[1:], True, offset)
        if first in "CLG":
            return self._read_object(first, offset)
        return Color(raw=token)

    def _read_object(self, kind: str, offset: int) -> ColorSpec:
        scanner = self.scanner
        extra_values = 4 if kind == "L" else 6 if kind == "G" else 0
        header = [scanner.read_bare_token()]
        header.extend(scanner.read_bare_token() for _ in range(extra_values))
        try:
            parts = [int(value) for value in header]
        except ValueError as exc:
            raise XDotParseError(
                f"invalid {kind!r} color header {header!r}", offset
            ) from exc
        count = parts[0] if parts else -1
        expected = 5 if kind == "L" else 7 if kind == "G" else 1
        if count < 0 or len(parts) != expected:
            raise XDotParseError(
                f"invalid {kind!r} color header {header!r}", offset
            )
        entries: list[ColorListEntry] = []
        for _ in range(count):
            color = self.read_color()
            weight: float | None = None
            if self.scanner.next_token_is_number():
                weight = self.scanner.read_double()
            entries.append(ColorListEntry(color=color, weight=weight))
        if kind == "C":
            return ColorList(entries=tuple(entries))
        coordinates = [value / 255.0 for value in parts[1:]]
        if kind == "L":
            x0, y0, x1, y1 = coordinates
            return LinearGradient(
                entries=tuple(entries), x0=x0, y0=y0, x1=x1, y1=y1
            )
        x0, y0, r0, x1, y1, r1 = coordinates
        return RadialGradient(
            entries=tuple(entries),
            x0=x0,
            y0=y0,
            r0=r0,
            x1=x1,
            y1=y1,
            r1=r1,
        )


def parse_xdot(
    source: str | bytes, *, text_context: bool = False
) -> list[DrawOp]:
    """Parse the XDOT instruction string ``source`` into typed operations.

    ``text_context`` selects how the ``s`` control is interpreted: pen
    width in draw operations, font size within a ``T`` text run.
    """

    data = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    scanner = _Scanner(data)
    operations: list[DrawOp] = []

    while not scanner.eof():
        op_offset = scanner.pos
        op = scanner.read_op()
        filled = op.isupper()

        if op in ("E", "e"):
            x = scanner.read_double()
            y = scanner.read_double()
            width = scanner.read_double()
            height = scanner.read_double()
            operations.append(
                Ellipse(
                    center=Point(x, y),
                    width=width,
                    height=height,
                    filled=filled,
                )
            )

        elif op in ("P", "p", "L", "l"):
            count = scanner.read_count()
            points = scanner.read_points(count)
            if op in ("P", "p"):
                operations.append(Polygon(points=points, filled=filled))
            else:
                operations.append(Polyline(points=points, filled=filled))

        elif op in ("B", "b"):
            count = scanner.read_count()
            points = scanner.read_points(count)
            start: Point | None = None
            end: Point | None = None
            if not filled:
                start = scanner.read_point()
                end = scanner.read_point()
            operations.append(
                BSpline(
                    points=points,
                    filled=filled,
                    start=start,
                    end=end,
                )
            )

        elif op in ("I", "i"):
            x = scanner.read_double()
            y = scanner.read_double()
            width = scanner.read_double()
            height = scanner.read_double()
            name = scanner.read_encoded_string()
            operations.append(
                Image(
                    corner=Point(x, y),
                    width=width,
                    height=height,
                    name=name,
                    filled=filled,
                )
            )

        elif op == "K":
            count = scanner.read_count()
            operations.append(
                InteriorPoints(points=scanner.read_points(count))
            )

        elif op == "T":
            x = scanner.read_double()
            y = scanner.read_double()
            align = int(scanner.read_double())
            width = scanner.read_double()
            text = scanner.read_encoded_string()
            operations.append(
                Text(anchor=Point(x, y), align=align, width=width, text=text)
            )

        elif op == "F":
            mask = scanner.read_count()
            operations.append(FontCharacteristics(mask=mask))

        elif op == "S":
            # The same opcode encodes either a pen style string
            # ("S 7 -dashed-") or a size+fontname selection
            # ("S 14 6 -Arial-").  A leading number distinguishes them.
            if scanner.at_font_name_selection():
                size = scanner.read_double()
                name = scanner.read_encoded_string()
                operations.append(FontName(size=size, name=name))
            else:
                style = scanner.read_encoded_string()
                parts = tuple(
                    part.strip()
                    for part in style.split(",")
                    if part.strip()
                )
                operations.append(PenStyle(style=style, parts=parts))

        elif op == "s":
            operations.append(
                SetSize(
                    value=scanner.read_double(),
                    line_width=not text_context,
                )
            )

        elif op in ("C", "c"):
            spec = scanner.read_encoded_string()
            reader = _ColorReader(spec)
            color = reader.read_color()
            if not reader.scanner.eof():
                raise XDotParseError(
                    "trailing data after color specification",
                    reader.scanner.pos,
                )
            operations.append(SetColor(color=color, filled=filled))

        else:
            raise XDotParseError(f"unknown XDOT operation {op!r}", op_offset)

    return operations


def parse_drawings(
    values: dict[str, object] | object,
    *names: str,
) -> list[XDotDrawing]:
    """Parse XDOT-bearing attributes from a mapping or graph-like object.

    Attribute names default to ``_draw_`` and ``_ldraw_``.  Missing
    attributes are ignored; the original strings are never modified.
    """

    if not names:
        names = ("_draw_", "_ldraw_")
    if hasattr(values, "obj_dict"):
        obj_dict = values.obj_dict  # type: ignore[attr-defined]
        mapping: dict[str, object] = obj_dict["attributes"]
    elif isinstance(values, dict):
        mapping = values
    elif hasattr(values, "get"):
        mapping = {
            name: values.get(name)  # type: ignore[call-arg]
            for name in names
        }
    else:
        raise TypeError(
            "parse_drawings expects a mapping or object with .get()"
        )

    drawings: list[XDotDrawing] = []
    for name in names:
        raw = mapping.get(name)
        if raw is None:
            continue
        source = raw if isinstance(raw, str) else str(raw)
        operations = parse_xdot(source)
        drawings.append(
            XDotDrawing(operations=tuple(operations), source=source)
        )
    return drawings
