# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Parser for the XDOT drawing language.

Graphviz emits machine-readable drawing instructions for each graph
element, in the ``_draw_`` and ``_ldraw_`` attributes (and the
``_tdraw_``/``_hldraw_`` variants), when a graph is written with
``-Txdot``.  The format is documented as the XDOT language in the
Graphviz sources (``xdot.7``).

Every instruction starts with a single ASCII opcode character followed
by space-separated arguments.  Text, colour and font-name arguments are
*length-prefixed* strings: a decimal byte length, a single space, the
exact number of bytes, and a trailing ``-``::

    T <x> <y> <align> <n> <text bytes>-
    C <n> -<colour bytes>-

This module turns such a blob into a typed list of operations.  A
malformed instruction is reported as an :class:`~pydot.exceptions.XDotError`
instead of silently consuming the following instructions.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydot.exceptions import (
    XDotError,
    XDotNegativeLengthError,
    XDotPointCountError,
    XDotTruncatedError,
    XDotUnknownOpError,
)

__all__ = [
    "Point",
    "ColorStop",
    "Gradient",
    "XDotOp",
    "Ellipse",
    "Polyline",
    "Polygon",
    "Bezier",
    "Text",
    "Color",
    "Font",
    "Style",
    "LineStyle",
    "GradientFill",
    "XDotDrawing",
    "XDotParser",
    "parse_drawing",
    "parse_operations",
    "drawings_from",
]

# Whitespace used to separate instruction fields.
_WHITESPACE = b" \t\n\r\f\v"

# Opcodes that start an XDOT instruction, used to resynchronise the
# parser after a malformed instruction.
_KNOWN_OPCODES = frozenset(b"EePpBbTCcFSLKk")

Number = int | float


@dataclass(frozen=True)
class Point:
    """A single drawing coordinate in XDOT units (72 points per inch)."""

    x: float
    y: float


@dataclass(frozen=True)
class ColorStop:
    """One stop of a gradient description."""

    fraction: float
    color: str


@dataclass(frozen=True)
class Gradient:
    """A gradient description parsed out of a colour string.

    Graphviz encodes linear gradients in colour values as
    ``"<angle>:<start colour>:<end colour>[:<weight>]"``.
    """

    angle: float | None
    colors: tuple[str, ...]
    weight: float | None
    stops: tuple[ColorStop, ...] = ()


class XDotOp:
    """Base class for all typed XDOT operations."""

    opcode: str = ""

    @property
    def filled(self) -> bool:
        """Whether the operation uses the lowercase (filled) opcode."""
        return self.opcode in ("e", "p", "b", "k")


@dataclass(frozen=True)
class Ellipse(XDotOp):
    """``E``/``e``: an ellipse inscribed in the given rectangle."""

    x: float
    y: float
    width: float
    height: float
    opcode: str = "E"


@dataclass(frozen=True)
class Polyline(XDotOp):
    """``P``: a stroked, open polyline through ``points``."""

    points: tuple[Point, ...]
    opcode: str = "P"


@dataclass(frozen=True)
class Polygon(XDotOp):
    """``p``: a filled, closed polygon through ``points``."""

    points: tuple[Point, ...]
    opcode: str = "p"


@dataclass(frozen=True)
class Bezier(XDotOp):
    """``B``/``b``: a stroked/filled cubic Bezier spline."""

    points: tuple[Point, ...]
    opcode: str = "B"

    @property
    def filled(self) -> bool:
        return self.opcode == "b"


@dataclass(frozen=True)
class Text(XDotOp):
    """``T``: a label string.

    ``align`` is ``-1`` (left), ``0`` (centred) or ``1`` (right).
    """

    x: float
    y: float
    align: int
    text: str
    opcode: str = "T"


@dataclass(frozen=True)
class Color(XDotOp):
    """``C``/``c``: set the pen/stroke (``C``) or fill (``c``) colour.

    The raw colour token is always kept verbatim in ``color``.  Plain
    colours (``#rrggbb[aa]``, named colours, HSV components) and
    gradient descriptions are recognised; the latter are additionally
    exposed in ``gradient``.
    """

    color: str
    gradient: Gradient | None
    fill: bool = False
    opcode: str = "C"


@dataclass(frozen=True)
class Font(XDotOp):
    """``F``: font size and family for following text operations."""

    size: float
    name: str
    opcode: str = "F"


@dataclass(frozen=True)
class Style(XDotOp):
    """``S``: comma-separated style flags (e.g. ``filled,dashed``)."""

    flags: tuple[str, ...]
    opcode: str = "S"


@dataclass(frozen=True)
class LineStyle(XDotOp):
    """``L``: line width and dash pattern.

    A ``0`` dash segment marks the end of the pattern (``solid`` lines
    carry no segments).
    """

    width: float
    dashes: tuple[float, ...]
    opcode: str = "L"


@dataclass(frozen=True)
class GradientFill(XDotOp):
    """``K``/``k``: a linear gradient fill between ``p0`` and ``p1``."""

    p0: Point
    p1: Point
    stops: tuple[ColorStop, ...]
    opcode: str = "K"

    @property
    def filled(self) -> bool:
        return self.opcode == "k"


@dataclass(frozen=True)
class XDotDrawing:
    """The operations decoded from one XDOT attribute value.

    ``operations`` holds every instruction that could be decoded in
    order.  ``errors`` reports malformed instructions; parsing always
    resynchronises at the next recognisable opcode so that a bad
    instruction cannot swallow the instructions following it.
    """

    operations: list[XDotOp] = field(default_factory=list)
    errors: list[XDotError] = field(default_factory=list)

    def __iter__(self) -> Iterator[XDotOp]:
        return iter(self.operations)

    def __len__(self) -> int:
        return len(self.operations)

    def __getitem__(self, index: int) -> XDotOp:
        return self.operations[index]

    @property
    def ok(self) -> bool:
        """``True`` when the whole input decoded without errors."""
        return not self.errors


class _Scanner:
    """Cursor over the bytes of one XDOT attribute value."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def skip_space(self) -> None:
        data = self.data
        n = len(data)
        pos = self.pos
        while pos < n and data[pos] in _WHITESPACE:
            pos += 1
        self.pos = pos

    def _token(self) -> tuple[int, bytes]:
        self.skip_space()
        data = self.data
        n = len(data)
        start = self.pos
        end = start
        while end < n and data[end] not in _WHITESPACE:
            end += 1
        return start, data[start:end]

    def read_number(self) -> Number:
        start, token = self._token()
        if not token:
            raise XDotTruncatedError(
                "expected a number but reached end of input", start
            )
        try:
            text = token.decode("ascii")
        except UnicodeDecodeError:
            raise XDotTruncatedError(
                "expected a number but found non-ASCII bytes", start
            ) from None
        try:
            value: Number = int(text)
        except ValueError:
            try:
                value = float(text)
            except ValueError:
                raise XDotTruncatedError(
                    f"expected a number but found {text!r}", start
                ) from None
        self.pos += len(token)
        return value

    def read_float(self) -> float:
        return float(self.read_number())

    def read_int(self) -> int:
        start, token = self._token()
        if not token:
            raise XDotTruncatedError(
                "expected an integer but reached end of input", start
            )
        try:
            text = token.decode("ascii")
        except UnicodeDecodeError:
            raise XDotTruncatedError(
                "expected an integer but found non-ASCII bytes", start
            ) from None
        try:
            value = int(text)
        except ValueError:
            raise XDotTruncatedError(
                f"expected an integer but found {text!r}", start
            ) from None
        self.pos += len(token)
        return value

    def read_point(self) -> Point:
        return Point(self.read_float(), self.read_float())

    def read_points(self, count: int) -> list[Point]:
        return [self.read_point() for _ in range(count)]

    def read_byte_slice(self, length: int) -> bytes:
        """Read exactly ``length`` encoded bytes for a counted string."""
        if length < 0:
            raise XDotNegativeLengthError(
                f"negative string length {length}", self.pos
            )
        self.skip_space()
        start = self.pos
        end = start + length
        if end > len(self.data):
            raise XDotTruncatedError(
                f"expected {length} bytes but only "
                f"{len(self.data) - start} remain",
                start,
            )
        raw = self.data[start:end]
        self.pos = end
        return raw

    def expect_dash(self) -> None:
        """Consume the ``-`` that terminates a counted string."""
        self.skip_space()
        if self.pos >= len(self.data):
            raise XDotTruncatedError(
                "expected '-' terminator but reached end of input",
                self.pos,
            )
        if self.data[self.pos] != ord("-"):
            raise XDotTruncatedError(
                "expected '-' terminator after counted string",
                self.pos,
            )
        self.pos += 1

    def read_text(self) -> str:
        """Decode a ``T`` text argument: ``<n> <bytes>-``."""
        length = self.read_int()
        raw = self.read_byte_slice(length)
        self.expect_dash()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as err:
            raise XDotTruncatedError(
                f"undecodable UTF-8 text ({err.reason})",
                self.pos - length,
            ) from None
        return _unescape(text)

    def read_dashed_string(self) -> str:
        """Decode a colour/font argument: ``<n> -<bytes>-``."""
        length = self.read_int()
        raw = self.read_byte_slice(length)
        self.expect_dash()
        if not raw or raw[0] != ord("-"):
            raise XDotTruncatedError(
                "expected '-' prefix before counted string",
                self.pos - length,
            )
        return _unescape(raw[1:].decode("utf-8"))


def _unescape(text: str) -> str:
    """Undo Graphviz' XDOT string escapes.

    Spaces are escaped as the octal escape ``\\040`` and a literal
    backslash is doubled.
    """
    return text.replace("\\040", " ").replace("\\\\", "\\")


def _parse_color(raw: str, fill: bool) -> Color:
    """Build a :class:`Color`, recognising gradient descriptions.

    A gradient description is ``"<angle>:<c0>:<c1>[:<weight>]"`` or a
    bare ``"<c0>:<c1>"``.  Anything else is kept as a plain colour,
    verbatim.
    """
    parts = raw.split(":")
    gradient: Gradient | None = None

    angle: float | None = None
    weight: float | None = None
    rest = parts

    if len(parts) >= 3:
        try:
            angle = float(parts[0])
        except ValueError:
            angle = None
        if angle is not None:
            rest = parts[1:]

    if len(rest) == 3 and angle is not None:
        try:
            weight = float(rest[-1])
            rest = rest[:-1]
        except ValueError:
            weight = None

    if len(rest) == 2 and raw != "none":
        colors = (rest[0], rest[1])
        stops = (
            ColorStop(0.0, colors[0]),
            ColorStop(1.0, colors[1]),
        )
        gradient = Gradient(angle, colors, weight, stops)

    return Color(
        color=raw,
        gradient=gradient,
        fill=fill,
        opcode="c" if fill else "C",
    )


class XDotParser:
    """Decode XDOT drawing instructions into typed operations."""

    def parse(self, data: str | bytes, strict: bool = False) -> XDotDrawing:
        """Parse all instructions in ``data``.

        With ``strict`` false (the default) a malformed instruction is
        recorded in ``XDotDrawing.errors`` and parsing resynchronises at
        the next recognisable opcode, so later instructions are never
        lost.  With ``strict`` true the first error is raised.
        """
        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = data

        scanner = _Scanner(payload)
        drawing = XDotDrawing()
        total = len(payload)

        while True:
            scanner.skip_space()
            if scanner.pos >= total:
                break
            start = scanner.pos
            try:
                op = self._read_operation(scanner)
            except (XDotError, UnicodeDecodeError, ValueError) as error:
                if not isinstance(error, XDotError):
                    error = XDotTruncatedError(str(error), start)
                drawing.errors.append(error)
                scanner.pos = self._resync(scanner, start)
                if scanner.pos >= total:
                    break
                continue
            drawing.operations.append(op)

        if strict and drawing.errors:
            raise drawing.errors[0]
        return drawing

    def _resync(self, scanner: _Scanner, at: int) -> int:
        """Find the offset of the next fully-parseable instruction."""
        data = scanner.data
        total = len(data)
        offset = at
        while offset < total:
            char = data[offset]
            if char in _KNOWN_OPCODES and (
                offset == 0 or data[offset - 1] in _WHITESPACE
            ):
                probe = _Scanner(data)
                probe.pos = offset
                try:
                    self._read_operation(probe)
                except (XDotError, UnicodeDecodeError, ValueError):
                    offset += 1
                    continue
                end = probe.pos
                if end >= total or data[end] in _WHITESPACE:
                    return offset
            offset += 1
        return total

    def _read_operation(self, sc: _Scanner) -> XDotOp:
        sc.skip_space()
        if sc.pos >= len(sc.data):
            raise XDotTruncatedError(
                "expected an opcode but reached end of input", sc.pos
            )
        code = chr(sc.data[sc.pos])
        if code not in "EePpBbTCcFSLKk":
            raise XDotUnknownOpError(f"unknown XDOT opcode {code!r}", sc.pos)
        sc.pos += 1

        if code in ("E", "e"):
            x = sc.read_float()
            y = sc.read_float()
            width = sc.read_float()
            height = sc.read_float()
            return Ellipse(x, y, width, height, code)

        if code in ("P", "p"):
            count = sc.read_int()
            if count < 0:
                raise XDotPointCountError(
                    f"negative point count {count}", sc.pos
                )
            try:
                points = tuple(sc.read_points(count))
            except XDotTruncatedError as error:
                raise XDotPointCountError(
                    f"opcode {code!r} declared {count} points",
                    error.position,
                ) from None
            if code == "P":
                return Polyline(points, code)
            return Polygon(points, code)

        if code in ("B", "b"):
            count = sc.read_int()
            if count < 0:
                raise XDotPointCountError(
                    f"negative point count {count}", sc.pos
                )
            try:
                points = tuple(sc.read_points(count))
            except XDotTruncatedError as error:
                raise XDotPointCountError(
                    f"opcode {code!r} declared {count} points",
                    error.position,
                ) from None
            return Bezier(points, code)

        if code == "T":
            x = sc.read_float()
            y = sc.read_float()
            align = sc.read_int()
            text = sc.read_text()
            return Text(x, y, align, text, code)

        if code in ("C", "c"):
            color = sc.read_dashed_string()
            return _parse_color(color, code == "c")

        if code == "F":
            size = sc.read_float()
            name = sc.read_dashed_string()
            return Font(size, name, code)

        if code == "S":
            raw_style = sc.read_dashed_string()
            flags = tuple(flag for flag in raw_style.split(",") if flag)
            return Style(flags, code)

        if code == "L":
            width = sc.read_float()
            dashes: list[float] = []
            while True:
                value = sc.read_number()
                if value == 0:
                    break
                dashes.append(float(value))
            return LineStyle(width, tuple(dashes), code)

        p0 = sc.read_point()
        p1 = sc.read_point()
        stop_count = sc.read_int()
        if stop_count < 0:
            raise XDotPointCountError(
                f"negative gradient stop count {stop_count}", sc.pos
            )
        stops: list[ColorStop] = []
        for _ in range(stop_count):
            fraction = sc.read_float()
            color = sc.read_dashed_string()
            stops.append(ColorStop(fraction, color))
        return GradientFill(p0, p1, tuple(stops), code)


def parse_drawing(
    data: str | bytes | None, strict: bool = False
) -> XDotDrawing:
    """Decode one XDOT attribute value into an :class:`XDotDrawing`."""
    if data is None or data == "":
        return XDotDrawing()
    return XDotParser().parse(data, strict=strict)


def parse_operations(
    data: str | bytes | None,
) -> list[XDotOp]:
    """Return just the typed operations decoded from ``data``."""
    return parse_drawing(data).operations


def drawings_from(obj: object, attribute: str = "_draw_") -> XDotDrawing:
    """Decode an XDOT attribute from a string/bytes or pydot object.

    Accepts a raw XDOT string/bytes value, or any object exposing an
    attribute mapping through ``get_attributes()`` (e.g.
    :class:`pydot.Dot`, :class:`pydot.Node`, :class:`pydot.Edge`).
    The original attribute is never modified.
    """
    if isinstance(obj, (str, bytes)):
        return parse_drawing(obj)

    try:
        attributes = obj.get_attributes()
    except AttributeError:
        raise TypeError(
            "expected an XDOT string or an object with get_attributes()"
        ) from None
    return parse_drawing(attributes.get(attribute))
