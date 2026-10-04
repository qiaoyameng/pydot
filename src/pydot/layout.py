"""Spatial hit testing for graphs containing Graphviz layout attributes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Union

from pydot.core import Dot, Edge, Node, Subgraph
from pydot.exceptions import LayoutError

POINT = 72.0
_NUMBER_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)$"
)
_LENGTH_FACTORS = {
    "": 1.0,
    "in": POINT,
    "inch": POINT,
    "inches": POINT,
    "pt": 1.0,
    "point": 1.0,
    "points": 1.0,
    "pc": 12.0,
    "pica": 12.0,
    "picas": 12.0,
    "cm": POINT / 2.54,
    "mm": POINT / 25.4,
}
_RECT_SHAPES = {
    "box",
    "box3d",
    "cds",
    "component",
    "cylinder",
    "folder",
    "note",
    "rect",
    "rectangle",
    "square",
    "msquare",
}
_ELLIPSE_SHAPES = {
    "ellipse",
    "oval",
    "circle",
    "mcircle",
}


def _unquote(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def intersects(
        self,
        x: float,
        y: float,
        tolerance: float,
    ) -> bool:
        return (
            self.min_x - tolerance <= x <= self.max_x + tolerance
            and self.min_y - tolerance <= y <= self.max_y + tolerance
        )


@dataclass(frozen=True)
class LayoutHit:
    """An object returned by a layout hit test."""

    kind: str
    object: Any
    distance: float
    order: int

    @property
    def node(self) -> Any | None:
        if self.kind != "node":
            return None
        return self.object

    @property
    def edge(self) -> Any | None:
        if self.kind != "edge":
            return None
        return self.object


@dataclass(frozen=True)
class _Entry:
    kind: str
    obj: Any
    order: int
    bounds: BoundingBox
    center: Point | None = None
    radii: Point | None = None
    segments: tuple[tuple[Point, Point, Point, Point], ...] | None = None
    half_width: float = 0.0
    shape: str = ""


def _pair_scale(
    scale: Union[float, tuple[float, float]],
) -> tuple[float, float]:
    if isinstance(scale, (int, float)):
        values = (float(scale), float(scale))
    elif isinstance(scale, (tuple, list)) and len(scale) == 2:
        values = (float(scale[0]), float(scale[1]))
    else:
        raise TypeError("scale must be a number or a pair of numbers")

    if not all(math.isfinite(value) and value != 0 for value in values):
        raise ValueError("scale values must be finite and non-zero")
    return values


def _pair_translate(
    translate: Union[float, tuple[float, float]],
) -> tuple[float, float]:
    if isinstance(translate, (int, float)):
        values = (float(translate), float(translate))
    elif isinstance(translate, (tuple, list)) and len(translate) == 2:
        values = (float(translate[0]), float(translate[1]))
    else:
        raise TypeError("translate must be a number or a pair of numbers")

    if not all(math.isfinite(value) for value in values):
        raise ValueError("translate values must be finite")
    return values


def _coerce_point(value: Any, context: str) -> Point:
    value = _unquote(value)
    if isinstance(value, (tuple, list)) and len(value) in (2, 3):
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError) as exc:
            raise LayoutError(
                f"{context} contains a non-numeric point"
            ) from exc
    elif isinstance(value, str):
        fields = value.rstrip("!").split(",")
        if len(fields) not in (2, 3):
            raise LayoutError(f"{context} must contain x,y coordinates")
        try:
            x = float(fields[0])
            y = float(fields[1])
        except ValueError as exc:
            raise LayoutError(
                f"{context} contains non-numeric coordinates"
            ) from exc
    else:
        raise LayoutError(f"{context} must be a string or coordinate pair")

    if not math.isfinite(x) or not math.isfinite(y):
        raise LayoutError(f"{context} contains a non-finite coordinate")
    return Point(x, y)


def _parse_length(
    value: Any,
    *,
    default_inches: float | None = None,
    unitless_is_inches: bool = False,
) -> float:
    if value is None:
        if default_inches is None:
            raise LayoutError("required length is missing")
        return default_inches * POINT

    value = _unquote(value)
    if isinstance(value, (int, float)):
        number = float(value)
        factor = POINT if unitless_is_inches else 1.0
    else:
        if not isinstance(value, str):
            raise LayoutError("length must be a number or string")
        match = _NUMBER_RE.match(value.strip())
        if match is None:
            raise LayoutError(f"invalid length {value!r}")
        number = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "" and unitless_is_inches:
            factor = POINT
        elif unit not in _LENGTH_FACTORS:
            raise LayoutError(f"unsupported length unit {unit!r}")
        else:
            factor = _LENGTH_FACTORS[unit]

    if not math.isfinite(number) or number <= 0:
        raise LayoutError("length must be finite and positive")
    return number * factor


_POINT_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_EDGE_TOKEN_RE = re.compile(
    rf"(?:(?P<marker>[esp])\s*,\s*)?"
    rf"(?P<x>{_POINT_NUMBER})\s*,\s*(?P<y>{_POINT_NUMBER})"
    rf"(?![^;,\s])"
)


def _parse_edge_position(
    value: Any,
) -> tuple[tuple[tuple[Point, Point, Point, Point], ...], BoundingBox]:
    value = _unquote(value)
    if not isinstance(value, str):
        raise LayoutError("edge pos must be a string")

    all_segments: list[tuple[Point, Point, Point, Point]] = []
    all_points: list[Point] = []

    for path_text in value.replace("\n", " ").split(";"):
        start: Point | None = None
        points: list[Point] = []
        cursor = 0
        path_text = path_text.strip()

        while cursor < len(path_text):
            match = _EDGE_TOKEN_RE.match(path_text, cursor)
            if match is None or match.start() != cursor:
                raise LayoutError("edge pos contains malformed coordinates")
            marker = match.group("marker")
            point = Point(float(match.group("x")), float(match.group("y")))
            if not math.isfinite(point.x) or not math.isfinite(point.y):
                raise LayoutError("edge pos contains a non-finite coordinate")

            if marker == "p":
                raise LayoutError("polyline edge positions are not supported")
            if marker == "s":
                start = point
            elif marker != "e":
                points.append(point)

            cursor = match.end()
            while cursor < len(path_text) and path_text[cursor].isspace():
                cursor += 1

        if not points:
            if start is not None:
                raise LayoutError(
                    "edge pos contains a start but no curve points"
                )
            continue

        if start is not None:
            if len(points) % 3 != 0:
                raise LayoutError(
                    "edge pos must contain one or more cubic point triplets"
                )
            path_start = start
            for offset in range(0, len(points), 3):
                control_1, control_2, end = points[offset : offset + 3]
                all_segments.append((path_start, control_1, control_2, end))
                path_start = end
            all_points.append(start)
            all_points.extend(points)
        else:
            if len(points) < 4 or (len(points) - 4) % 3 != 0:
                raise LayoutError(
                    "edge pos must contain 4, 7, 10, ... curve points"
                )
            for offset in range(0, len(points) - 1, 3):
                all_segments.append(
                    tuple(  # type: ignore[arg-type]
                        points[offset : offset + 4]
                    )
                )
            all_points.extend(points[::3])
            all_points.append(points[-1])

    if not all_segments:
        raise LayoutError("edge pos does not contain a cubic Bezier curve")

    min_x = min(point.x for point in all_points)
    max_x = max(point.x for point in all_points)
    min_y = min(point.y for point in all_points)
    max_y = max(point.y for point in all_points)
    return tuple(all_segments), BoundingBox(min_x, min_y, max_x, max_y)


def _distance_to_rectangle(point: Point, center: Point, radii: Point) -> float:
    dx = max(abs(point.x - center.x) - radii.x, 0.0)
    dy = max(abs(point.y - center.y) - radii.y, 0.0)
    return math.hypot(dx, dy)


def _ellipse_radius(angle: float, radii: Point) -> float:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return 1.0 / math.sqrt(
        (cosine / radii.x) ** 2 + (sine / radii.y) ** 2
    )


def _distance_to_ellipse(point: Point, center: Point, radii: Point) -> float:
    relative = Point(point.x - center.x, point.y - center.y)
    inside = (relative.x / radii.x) ** 2 + (relative.y / radii.y) ** 2 <= 1
    if inside:
        return 0.0

    angle = math.atan2(relative.y * radii.x, relative.x * radii.y)
    low = angle - math.pi / 2.0
    high = angle + math.pi / 2.0

    def boundary_distance(value: float) -> float:
        radius = _ellipse_radius(value, radii)
        return math.hypot(
            relative.x - radius * math.cos(value),
            relative.y - radius * math.sin(value),
        )

    for _ in range(60):
        middle_low = (2.0 * low + high) / 3.0
        middle_high = (low + 2.0 * high) / 3.0
        if boundary_distance(middle_low) < boundary_distance(middle_high):
            high = middle_high
        else:
            low = middle_low

    return boundary_distance((low + high) / 2.0)


def _bezier_bounds(
    segment: tuple[Point, Point, Point, Point],
) -> BoundingBox:
    xs = [point.x for point in segment]
    ys = [point.y for point in segment]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def _distance_to_bezier(
    point: Point,
    segment: tuple[Point, Point, Point, Point],
    tolerance: float,
) -> float:
    p0, p1, p2, p3 = segment
    b = (
        p0.x - point.x,
        p1.x - point.x,
        p2.x - point.x,
        p3.x - point.x,
    )
    d = (
        p0.y - point.y,
        p1.y - point.y,
        p2.y - point.y,
        p3.y - point.y,
    )

    x_powers = [
        b[0],
        3.0 * (b[1] - b[0]),
        3.0 * (b[0] - 2.0 * b[1] + b[2]),
        -b[0] + 3.0 * b[1] - 3.0 * b[2] + b[3],
    ]
    y_powers = [
        d[0],
        3.0 * (d[1] - d[0]),
        3.0 * (d[0] - 2.0 * d[1] + d[2]),
        -d[0] + 3.0 * d[1] - 3.0 * d[2] + d[3],
    ]

    derivative = [0.0] * 6
    for axis_powers in (x_powers, y_powers):
        for first_power, first_value in enumerate(axis_powers):
            for second_power, second_value in enumerate(axis_powers):
                power = first_power + second_power
                if power > 0:
                    derivative[power - 1] += (
                        power * first_value * second_value
                    )

    def evaluate(parameter: float) -> float:
        value = 0.0
        for coefficient in reversed(derivative):
            value = value * parameter + coefficient
        return value

    best = min(math.hypot(b[0], d[0]), math.hypot(b[3], d[3]))

    previous_t = 0.0
    previous_value = evaluate(0.0)
    for interval in range(1, 129):
        current_t = interval / 128.0
        current_value = evaluate(current_t)
        if previous_value == 0.0:
            root_t = previous_t
            root_point = _bezier_point(segment, root_t)
            best = min(
                best,
                math.hypot(point.x - root_point.x, point.y - root_point.y),
            )
        elif previous_value * current_value < 0.0:
            low = previous_t
            high = current_t
            low_value = previous_value
            for _ in range(50):
                middle = (low + high) / 2.0
                middle_value = evaluate(middle)
                if low_value * middle_value <= 0.0:
                    high = middle
                else:
                    low = middle
                    low_value = middle_value
            root_t = (low + high) / 2.0
            root_point = _bezier_point(segment, root_t)
            best = min(
                best,
                math.hypot(point.x - root_point.x, point.y - root_point.y),
            )
            if math.isfinite(tolerance) and best <= tolerance:
                return best
        previous_t = current_t
        previous_value = current_value

    return best


def _bezier_point(
    segment: tuple[Point, Point, Point, Point],
    t: float,
) -> Point:
    p0, p1, p2, p3 = segment
    u = 1.0 - t
    x = (
        u**3 * p0.x
        + 3.0 * u**2 * t * p1.x
        + 3.0 * u * t**2 * p2.x
        + t**3 * p3.x
    )
    y = (
        u**3 * p0.y
        + 3.0 * u**2 * t * p1.y
        + 3.0 * u * t**2 * p2.y
        + t**3 * p3.y
    )
    return Point(x, y)



class LayoutIndex:
    """Spatial index for laid-out nodes and cubic Bezier edges.

    Graphviz uses points for ``pos`` and inches for node ``width`` and
    ``height``. The index converts those units once. Coordinate values are
    transformed with ``x = scale_x * pos_x + translate_x`` and likewise for
    ``y``.
    """

    def __init__(
        self,
        graph: Dot,
        *,
        scale: Union[float, tuple[float, float]] = 1.0,
        translate: Union[float, tuple[float, float]] = (0.0, 0.0),
        cell_size: float = 128.0,
    ) -> None:
        self.scale_x, self.scale_y = _pair_scale(scale)
        self.translate_x, self.translate_y = _pair_translate(translate)
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be finite and positive")
        self.cell_size = float(cell_size)
        self.entries: tuple[_Entry, ...] = tuple(self._build_entries(graph))
        self.cells: dict[tuple[int, int], list[_Entry]] = {}
        for entry in self.entries:
            for key in self._cells_for(entry.bounds):
                self.cells.setdefault(key, []).append(entry)

    def hit_test(
        self,
        x: float,
        y: float,
        tolerance: float = 0.0,
    ) -> list[LayoutHit]:
        """Return all objects within ``tolerance``, nearest first."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("query coordinates must be finite")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")

        query = Point(float(x), float(y))
        hits: list[LayoutHit] = []
        for entry in self._candidates(query, tolerance):
            distance = self._distance(query, entry, tolerance)
            effective_tolerance = tolerance + entry.half_width
            if distance <= effective_tolerance:
                hits.append(
                    LayoutHit(entry.kind, entry.obj, distance, entry.order)
                )
        hits.sort(key=lambda hit: (hit.distance, hit.order))
        return hits

    def hit_at(
        self,
        x: float,
        y: float,
        tolerance: float = 0.0,
    ) -> Any | None:
        """Return the nearest object within ``tolerance``."""
        hits = self.hit_test(x, y, tolerance)
        if not hits:
            return None
        return hits[0].object

    def find(
        self,
        x: float,
        y: float,
        tolerance: float = 0.0,
    ) -> Any | None:
        """Alias for :meth:`hit_at`."""
        return self.hit_at(x, y, tolerance)

    def count_candidates(
        self,
        x: float,
        y: float,
        tolerance: float = 0.0,
    ) -> int:
        """Count unique objects visited after spatial filtering."""
        return len(self._candidates(Point(float(x), float(y)), tolerance))

    def _candidates(
        self,
        point: Point,
        tolerance: float,
    ) -> list[_Entry]:
        keys = self._cells_for(
            BoundingBox(
                point.x - tolerance,
                point.y - tolerance,
                point.x + tolerance,
                point.y + tolerance,
            ),
        )
        entries: dict[int, _Entry] = {}
        for key in keys:
            for entry in self.cells.get(key, ()):
                pad = tolerance + entry.half_width
                if entry.bounds.intersects(point.x, point.y, pad):
                    entries[id(entry)] = entry
        return sorted(entries.values(), key=lambda entry: entry.order)

    def _cells_for(self, bounds: BoundingBox) -> set[tuple[int, int]]:
        min_x = math.floor(bounds.min_x / self.cell_size)
        max_x = math.floor(bounds.max_x / self.cell_size)
        min_y = math.floor(bounds.min_y / self.cell_size)
        max_y = math.floor(bounds.max_y / self.cell_size)
        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1
        if span_x > 1024 or span_y > 1024:
            return set()
        return {
            (column, row)
            for column in range(min_x, max_x + 1)
            for row in range(min_y, max_y + 1)
        }

    def _distance(
        self,
        point: Point,
        entry: _Entry,
        tolerance: float,
    ) -> float:
        if entry.kind == "node":
            assert entry.center is not None
            assert entry.radii is not None
            if entry.shape == "rectangle":
                return _distance_to_rectangle(
                    point,
                    entry.center,
                    entry.radii,
                )
            return _distance_to_ellipse(point, entry.center, entry.radii)

        assert entry.segments is not None
        distance = math.inf
        cutoff = tolerance + entry.half_width
        for segment in entry.segments:
            bounds = _bezier_bounds(segment)
            if not bounds.intersects(point.x, point.y, min(distance, cutoff)):
                continue
            distance = min(
                distance,
                _distance_to_bezier(point, segment, min(distance, cutoff)),
            )
            if distance <= cutoff:
                break
        return distance

    def _transform(self, point: Point) -> Point:
        return Point(
            point.x * self.scale_x + self.translate_x,
            point.y * self.scale_y + self.translate_y,
        )

    def _build_entries(self, graph: Dot) -> list[_Entry]:
        entries: list[_Entry] = []

        def walk(container: Subgraph) -> None:
            children = []
            for obj_lists in container.obj_dict["nodes"].values():
                children.extend(obj_lists)
            for obj_lists in container.obj_dict["edges"].values():
                children.extend(obj_lists)
            for obj_lists in container.obj_dict["subgraphs"].values():
                children.extend(obj_lists)
            children.sort(key=lambda obj: obj.get("sequence") or 0)

            for child in children:
                kind = child.get("type")
                if kind == "node":
                    entry = self._node_entry(
                        Node(obj_dict=child), len(entries)
                    )
                    if entry is not None:
                        entries.append(entry)
                elif kind == "edge":
                    edge = Edge(obj_dict=child)
                    entries.append(self._edge_entry(edge, len(entries)))
                elif kind == "subgraph":
                    walk(Subgraph(obj_dict=child))

        walk(graph)
        return entries

    def _node_entry(self, node: Node, order: int) -> _Entry | None:
        name = str(_unquote(node.get_name()))
        if name in {"node", "edge", "graph"}:
            return None

        position = node.get("pos")
        if position is None:
            raise LayoutError(
                "node is missing pos",
                object_type="node",
                name=name,
                source=node,
            )

        shape = str(node.get("shape") or "ellipse").lower()
        if shape in _RECT_SHAPES:
            geometry = "rectangle"
        elif shape in _ELLIPSE_SHAPES:
            geometry = "ellipse"
        else:
            raise LayoutError(
                f"unsupported laid-out node shape {shape!r}",
                object_type="node",
                name=name,
                source=node,
            )

        def fail(message: str) -> LayoutError:
            return LayoutError(
                message,
                object_type="node",
                name=name,
                source=node,
            )

        try:
            center = self._transform(_coerce_point(position, "node pos"))
            width = _parse_length(
                node.get("width"),
                default_inches=0.75,
                unitless_is_inches=True,
            )
            height = _parse_length(
                node.get("height"),
                default_inches=0.5,
                unitless_is_inches=True,
            )
        except LayoutError as exc:
            if not exc.args:
                raise
            raise fail(str(exc)) from exc

        radii = Point(
            width * abs(self.scale_x) / 2.0,
            height * abs(self.scale_y) / 2.0,
        )
        bounds = BoundingBox(
            center.x - radii.x,
            center.y - radii.y,
            center.x + radii.x,
            center.y + radii.y,
        )
        return _Entry(
            "node",
            node,
            order,
            bounds,
            center=center,
            radii=radii,
            shape=geometry,
        )

    def _edge_entry(self, edge: Edge, order: int) -> _Entry:
        position = edge.get("pos")
        source = str(edge.get_source())
        destination = str(edge.get_destination())
        if position is None:
            raise LayoutError(
                "edge is missing pos",
                object_type="edge",
                name=f"{source} -> {destination}",
                source=edge,
            )

        try:
            segments, _ = _parse_edge_position(position)
            transformed_segments = tuple(
                tuple(  # type: ignore[arg-type]
                    self._transform(point) for point in segment
                )
                for segment in segments
            )
            penwidth = edge.get("penwidth")
            half_width = (
                _parse_length(penwidth) if penwidth is not None else 0.5
            )
            scale_average = (abs(self.scale_x) + abs(self.scale_y)) / 2.0
            half_width *= scale_average
        except LayoutError as exc:
            raise LayoutError(
                str(exc),
                object_type="edge",
                name=f"{source} -> {destination}",
                source=edge,
            ) from exc

        xs = [
            point.x
            for segment in transformed_segments
            for point in segment
        ]
        ys = [
            point.y
            for segment in transformed_segments
            for point in segment
        ]
        bounds = BoundingBox(
            min(xs) - half_width,
            min(ys) - half_width,
            max(xs) + half_width,
            max(ys) + half_width,
        )
        return _Entry(
            "edge",
            edge,
            order,
            bounds,
            segments=transformed_segments,
            half_width=half_width,
        )


def hit_test_layout(
    graph: Dot,
    x: float,
    y: float,
    *,
    tolerance: float = 0.0,
    scale: Union[float, tuple[float, float]] = 1.0,
    translate: Union[float, tuple[float, float]] = (0.0, 0.0),
) -> list[LayoutHit]:
    """Build a layout index and run one hit test."""
    return LayoutIndex(graph, scale=scale, translate=translate).hit_test(
        x,
        y,
        tolerance,
    )


def hit_at_layout(
    graph: Dot,
    x: float,
    y: float,
    *,
    tolerance: float = 0.0,
    scale: Union[float, tuple[float, float]] = 1.0,
    translate: Union[float, tuple[float, float]] = (0.0, 0.0),
) -> Any | None:
    """Build a layout index and return the nearest hit."""
    hits = hit_test_layout(
        graph,
        x,
        y,
        tolerance=tolerance,
        scale=scale,
        translate=translate,
    )
    return hits[0].object if hits else None
