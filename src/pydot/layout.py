# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Spatial hit queries for Graphviz layout attributes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from pydot.exceptions import LayoutHitError

if TYPE_CHECKING:
    from pydot.core import Dot, Edge, Graph, Node


Point = Tuple[float, float]
Bounds = Tuple[float, float, float, float]
Cubic = Tuple[Point, Point, Point, Point]
Scale = Tuple[float, float]
Translate = Tuple[float, float]

_RECTANGLE_SHAPES = frozenset(
    {
        "box",
        "box3d",
        "cds",
        "component",
        "cylinder",
        "folder",
        "insulator",
        "mcds",
        "msquare",
        "note",
        "rect",
        "rectangle",
        "square",
        "tab",
    }
)
_ELLIPSE_SHAPES = frozenset({"circle", "ellipse", "oval"})


@dataclass(frozen=True)
class Hit:
    """An object selected by a layout hit query."""

    target: Any
    kind: str
    name: str
    distance: float
    order: int


@dataclass(frozen=True)
class _Geometry:
    kind: str
    name: str
    target: Any
    order: int
    bounds: Bounds
    data: Any


class LayoutHitIndex:
    """Grid index for exact node and edge distance tests."""

    def __init__(
        self,
        graph: Dot,
        *,
        scale: float | Tuple[float, float] = 1.0,
        translate: Tuple[float, float] = (0.0, 0.0),
        cell_size: Optional[float] = None,
    ) -> None:
        self._scale = _parse_scale(scale)
        self._translate = _parse_pair(translate, "translate")
        if cell_size is not None and (
            cell_size <= 0 or not math.isfinite(cell_size)
        ):
            raise ValueError("cell_size must be positive")

        self.candidate_accesses = 0
        self._geometries: List[_Geometry] = []

        for order, obj in enumerate(_iter_layout_objects(graph)):
            if obj.obj_dict.get("type") == "node":
                self._geometries.append(
                    _node_geometry(obj, order, self._scale, self._translate)
                )
            else:
                self._geometries.append(
                    _edge_geometry(obj, order, self._scale, self._translate)
                )

        self.cell_size = (
            cell_size
            if cell_size is not None
            else _default_cell_size(self._geometries)
        )
        self._cells: Dict[Tuple[int, int], List[int]] = {}
        for index, geometry in enumerate(self._geometries):
            leaves = (
                _split_cubics(geometry.data, self.cell_size)
                if geometry.kind == "edge"
                else [(geometry.bounds, geometry.data)]
            )
            for bounds, _leaf in leaves:
                self._add_bounds(index, bounds)

    def reset_candidate_accesses(self) -> None:
        """Set the cumulative candidate counter back to zero."""

        self.candidate_accesses = 0

    def hits(
        self, x: float, y: float, *, tolerance: float = 0.0
    ) -> List[Hit]:
        """Return all objects within ``tolerance`` of a point."""

        point = (float(x), float(y))
        tolerance = _non_negative_float(tolerance, "tolerance")
        results: List[Hit] = []
        for geometry_index in self._candidates(point, tolerance):
            geometry = self._geometries[geometry_index]
            distance = _distance_to_geometry(point, geometry)
            if distance <= tolerance:
                results.append(
                    Hit(
                        geometry.target,
                        geometry.kind,
                        geometry.name,
                        distance,
                        geometry.order,
                    )
                )
        results.sort(key=lambda hit: (hit.distance, hit.order))
        return results

    def hit(
        self, x: float, y: float, *, tolerance: float = 0.0
    ) -> Optional[Hit]:
        """Return the closest object within ``tolerance``."""

        results = self.hits(x, y, tolerance=tolerance)
        return results[0] if results else None

    def nearest(self, x: float, y: float) -> Hit:
        """Return the closest laid-out object regardless of tolerance."""

        point = (float(x), float(y))
        nearest_hit: Optional[Hit] = None
        for geometry in self._geometries:
            distance = _distance_to_geometry(point, geometry)
            if nearest_hit is None or (
                distance,
                geometry.order,
            ) < (nearest_hit.distance, nearest_hit.order):
                nearest_hit = Hit(
                    geometry.target,
                    geometry.kind,
                    geometry.name,
                    distance,
                    geometry.order,
                )
        if nearest_hit is None:
            raise LayoutHitError("graph contains no indexed layout objects")
        return nearest_hit

    def _candidates(self, point: Point, tolerance: float) -> List[int]:
        min_col = math.floor((point[0] - tolerance) / self.cell_size)
        max_col = math.floor((point[0] + tolerance) / self.cell_size)
        min_row = math.floor((point[1] - tolerance) / self.cell_size)
        max_row = math.floor((point[1] + tolerance) / self.cell_size)

        found: Dict[int, None] = {}
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                for geometry_index in self._cells.get((col, row), ()):
                    found[geometry_index] = None

        candidates: List[int] = []
        for geometry_index in found:
            geometry = self._geometries[geometry_index]
            self.candidate_accesses += 1
            if _bounds_contains_expanded(geometry.bounds, point, tolerance):
                candidates.append(geometry_index)
        candidates.sort()
        return candidates

    def _add_bounds(self, geometry_index: int, bounds: Bounds) -> None:
        min_col = math.floor(bounds[0] / self.cell_size)
        max_col = math.floor(bounds[2] / self.cell_size)
        min_row = math.floor(bounds[1] / self.cell_size)
        max_row = math.floor(bounds[3] / self.cell_size)
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                self._cells.setdefault((col, row), []).append(geometry_index)


def _iter_layout_objects(graph: Graph) -> Iterable[Any]:
    def walk(current: Graph, seen: "set[int]") -> Iterable[Any]:
        entries = [
            (node.get_sequence() or 0, 0, index, node)
            for index, node in enumerate(current.get_node_list())
        ]
        entries.extend(
            (edge.get_sequence() or 0, 1, index, edge)
            for index, edge in enumerate(current.get_edge_list())
        )
        entries.extend(
            (subgraph.get_sequence() or 0, 2, index, subgraph)
            for index, subgraph in enumerate(current.get_subgraph_list())
        )
        for _, _, _, child in sorted(entries):
            if child.obj_dict.get("type") == "subgraph":
                yield from walk(child, seen)
                continue
            identity = id(child.obj_dict)
            is_default_statement = (
                child.obj_dict.get("type") == "node"
                and child.get_name() in ("node", "edge", "graph")
                and child.get("pos") is None
            )
            if identity not in seen and not is_default_statement:
                seen.add(identity)
                yield child

    yield from walk(graph, set())


def _node_geometry(
    node: Node, order: int, scale: Scale, translate: Translate
) -> _Geometry:
    name = node.get_name()
    if node.get("pos") is None:
        raise LayoutHitError(f"node {name!r}: missing pos", node)
    raw_shape = node.get("shape") or "ellipse"
    shape = str(raw_shape).strip().lower()
    if shape in _RECTANGLE_SHAPES:
        kind = "rectangle"
    elif shape in _ELLIPSE_SHAPES:
        kind = "ellipse"
    else:
        raise LayoutHitError(
            f"node {name!r}: unsupported shape {shape!r}", node
        )

    center = _parse_point(node.get("pos"), f"node {name!r}: pos", node)
    width = _parse_length(
        node.get("width"), f"node {name!r}: width", node, inches=True
    )
    height = _parse_length(
        node.get("height"), f"node {name!r}: height", node, inches=True
    )
    center = _transform_point(center, scale, translate)
    radii = (abs(width * scale[0]) / 2.0, abs(height * scale[1]) / 2.0)
    bounds = (
        center[0] - radii[0],
        center[1] - radii[1],
        center[0] + radii[0],
        center[1] + radii[1],
    )
    return _Geometry(kind, name, node, order, bounds, (center, radii))


def _edge_geometry(
    edge: Edge, order: int, scale: Scale, translate: Translate
) -> _Geometry:
    name = f"{edge.get_source()} -> {edge.get_destination()}"
    if edge.get("pos") is None:
        raise LayoutHitError(f"edge {name!r}: missing pos", edge)
    cubics = _parse_edge_pos(edge.get("pos"), name, edge)
    transformed = tuple(
        tuple(_transform_point(point, scale, translate) for point in cubic)
        for cubic in cubics
    )
    return _Geometry(
        "edge", name, edge, order, _cubics_bounds(transformed), transformed
    )


def _parse_edge_pos(
    value: Any, description: str, target: Edge
) -> Tuple[Cubic, ...]:
    if not isinstance(value, str) or not value.strip():
        raise LayoutHitError(f"{description}: malformed pos", target)

    start: Optional[Point] = None
    end: Optional[Point] = None
    body: List[Point] = []
    marker: Optional[str] = None

    for token in value.split():
        remainder = token
        if token in ("s", "e"):
            marker = token
            continue
        if token.startswith(("s,", "e,")):
            marker, remainder = token[0], token[2:]
        try:
            point = _parse_scalar_pair(remainder, description, target)
        except (TypeError, ValueError, IndexError) as exc:
            raise LayoutHitError(
                f"{description}: malformed pos", target
            ) from exc
        if marker == "s":
            if start is not None:
                raise LayoutHitError(f"{description}: duplicate start", target)
            start = point
        elif marker == "e":
            if end is not None:
                raise LayoutHitError(f"{description}: duplicate end", target)
            end = point
        else:
            body.append(point)
        marker = None

    if not body and start is not None and end is not None:
        return ((start, start, end, end),)
    if not body:
        raise LayoutHitError(f"{description}: malformed pos", target)

    if start is None and end is None:
        if (
            len(body) < 4
            or (len(body) - 4) % 3
        ):
            raise LayoutHitError(f"{description}: malformed pos", target)
        chunks = [tuple(body[:4])]
        chunks.extend(
            tuple(body[offset : offset + 3])
            for offset in range(4, len(body), 3)
        )
        previous: Optional[Point] = None
    elif start is None and end is not None:
        if len(body) < 4 or (len(body) - 4) % 3:
            raise LayoutHitError(f"{description}: malformed pos", target)
        chunks = [tuple(body[:4])]
        chunks.extend(
            tuple(body[offset : offset + 3])
            for offset in range(4, len(body), 3)
        )
        previous = None
    elif start is not None:
        if len(body) % 3:
            raise LayoutHitError(f"{description}: malformed pos", target)
        chunks = [
            tuple(body[offset : offset + 3])
            for offset in range(0, len(body), 3)
        ]
        previous = start
    elif end is not None and len(body) % 3 == 0:
        chunks = [
            tuple(body[offset : offset + 3])
            for offset in range(0, len(body), 3)
        ]
        previous = None
    else:
        raise LayoutHitError(f"{description}: malformed pos", target)

    cubics: List[Cubic] = []
    for chunk in chunks:
        if len(chunk) == 4 and previous is None:
            cubic = (chunk[0], chunk[1], chunk[2], chunk[3])
        elif len(chunk) == 3 and previous is not None:
            reflected = (
                2.0 * previous[0] - chunk[0][0],
                2.0 * previous[1] - chunk[0][1],
            )
            cubic = (previous, reflected, chunk[1], chunk[2])
        elif len(chunk) == 3 and previous is None and end is not None:
            cubic = (chunk[0], chunk[1], chunk[2], end)
        else:
            raise LayoutHitError(f"{description}: malformed pos", target)
        cubics.append(cubic)
        previous = cubic[3]
    return tuple(cubics)


def _parse_scale(
    value: Union[float, Tuple[float, float]]
) -> Scale:
    if isinstance(value, (int, float)):
        scale = (float(value), float(value))
    else:
        scale = _parse_pair(value, "scale")
    if not math.isfinite(scale[0]) or not math.isfinite(scale[1]):
        raise ValueError("scale must be finite")
    if scale[0] == 0.0 or scale[1] == 0.0:
        raise ValueError("scale must not be zero")
    return scale


def _parse_pair(
    value: Union[Tuple[float, float], List[float]], name: str
) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a pair of numbers")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a pair of numbers") from exc


def _non_negative_float(value: float, name: str) -> float:
    result = float(value)
    if result < 0.0 or not math.isfinite(result):
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _parse_scalar_pair(
    token: str, description: str, target: Any
) -> Point:
    parts = token.replace(",", " ").split()
    if len(parts) < 2:
        raise LayoutHitError(f"{description}: malformed coordinate", target)
    return float(parts[0]), float(parts[1].rstrip("!"))


def _parse_point(value: Any, description: str, target: Any) -> Point:
    if not isinstance(value, str):
        raise LayoutHitError(f"{description}: malformed coordinate", target)
    parts = value.strip().split(",")
    if len(parts) < 2:
        raise LayoutHitError(f"{description}: malformed coordinate", target)
    try:
        return float(parts[0]), float(parts[1].rstrip("!"))
    except ValueError as exc:
        raise LayoutHitError(
            f"{description}: malformed coordinate", target
        ) from exc


def _parse_length(
    value: Any, description: str, target: Any, *, inches: bool
) -> float:
    if value is None:
        raise LayoutHitError(f"{description}: missing layout length", target)
    try:
        text = str(value).strip().lower()
        if text.endswith("in"):
            number = text[:-2]
            multiplier = 72.0
        elif text.endswith("pt"):
            number = text[:-2]
            multiplier = 1.0
        elif text.endswith("px"):
            raise ValueError("px requires an explicit DPI conversion")
        else:
            number = text
            multiplier = 72.0 if inches else 1.0
        result = float(number) * multiplier
    except ValueError as exc:
        raise LayoutHitError(
            f"{description}: malformed length", target
        ) from exc
    if result < 0.0 or not math.isfinite(result):
        raise LayoutHitError(f"{description}: invalid length", target)
    return result


def _transform_point(
    point: Point, scale: Scale, translate: Translate
) -> Point:
    return (
        point[0] * scale[0] + translate[0],
        point[1] * scale[1] + translate[1],
    )


def _bounds_contains_expanded(
    bounds: Bounds, point: Point, tolerance: float
) -> bool:
    return (
        bounds[0] - tolerance <= point[0] <= bounds[2] + tolerance
        and bounds[1] - tolerance <= point[1] <= bounds[3] + tolerance
    )


def _cubics_bounds(cubics: Tuple[Cubic, ...]) -> Bounds:
    xs: List[float] = []
    ys: List[float] = []
    for source_cubic in cubics:
        for source_point in source_cubic:
            xs.append(source_point[0])
            ys.append(source_point[1])
    return min(xs), min(ys), max(xs), max(ys)


def _default_cell_size(geometries: List[_Geometry]) -> float:
    sizes: List[float] = []
    for geo in geometries:
        if geo.kind != "edge":
            sizes.append(geo.bounds[2] - geo.bounds[0])
            sizes.append(geo.bounds[3] - geo.bounds[1])
    positive = [value for value in sizes if value > 0.0]
    positive.sort()
    if not positive:
        return 64.0
    return min(256.0, max(8.0, positive[len(positive) // 2]))


def _split_cubics(
    cubics: Tuple[Cubic, ...], target: float
) -> List[Tuple[Bounds, Cubic]]:
    leaves: List[Tuple[Bounds, Cubic]] = []

    def split(cubic: Cubic, depth: int) -> None:
        bounds = _cubic_bounds(cubic)
        if (
            depth >= 24
            or (
                bounds[2] - bounds[0] <= target
                and bounds[3] - bounds[1] <= target
            )
        ):
            leaves.append((bounds, cubic))
            return
        left, right = _subdivide_cubic(cubic)
        split(left, depth + 1)
        split(right, depth + 1)

    for source_cubic in cubics:
        split(source_cubic, 0)
    return leaves


def _cubic_bounds(cubic: Cubic) -> Bounds:
    xs = [point[0] for point in cubic]
    ys = [point[1] for point in cubic]
    for dimension in range(2):
        p0, p1, p2, p3 = [point[dimension] for point in cubic]
        a = -3.0 * p0 + 9.0 * p1 - 9.0 * p2 + 3.0 * p3
        b = 6.0 * p0 - 12.0 * p1 + 6.0 * p2
        c = -3.0 * p0 + 3.0 * p1
        for root in _quadratic_roots(a, b, c):
            value = _cubic_value((p0, p1, p2, p3), root)
            if dimension == 0:
                xs.append(value)
            else:
                ys.append(value)
    return min(xs), min(ys), max(xs), max(ys)


def _quadratic_roots(a: float, b: float, c: float) -> Iterable[float]:
    if abs(a) < 1e-15:
        if abs(b) >= 1e-15:
            root = -c / b
            if 0.0 <= root <= 1.0:
                yield root
        return
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return
    square_root = math.sqrt(discriminant)
    roots = (
        (-b - square_root) / (2.0 * a),
        (-b + square_root) / (2.0 * a),
    )
    for root in roots:
        if 0.0 <= root <= 1.0:
            yield root


def _cubic_value(values: Tuple[float, float, float, float], t: float) -> float:
    inverse = 1.0 - t
    return (
        values[0] * inverse**3
        + 3.0 * values[1] * inverse**2 * t
        + 3.0 * values[2] * inverse * t**2
        + values[3] * t**3
    )


def _subdivide_cubic(cubic: Cubic) -> Tuple[Cubic, Cubic]:
    p0, p1, p2, p3 = cubic
    q1 = _midpoint(p0, p1)
    q2 = _midpoint(p1, p2)
    q3 = _midpoint(p2, p3)
    r1 = _midpoint(q1, q2)
    r2 = _midpoint(q2, q3)
    midpoint = _midpoint(r1, r2)
    return (p0, q1, r1, midpoint), (midpoint, r2, q3, p3)


def _midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _distance_to_geometry(point: Point, geometry: _Geometry) -> float:
    if geometry.kind == "edge":
        return min(
            _distance_to_cubic(point, cubic)
            for cubic in geometry.data
        )
    center, radii = geometry.data
    if geometry.kind == "rectangle":
        return _distance_to_rectangle(point, center, radii)
    return _distance_to_ellipse(point, center, radii)


def _distance_to_rectangle(
    point: Point, center: Point, radii: Point
) -> float:
    dx = abs(point[0] - center[0])
    dy = abs(point[1] - center[1])
    if dx <= radii[0] and dy <= radii[1]:
        return 0.0
    return math.hypot(max(0.0, dx - radii[0]), max(0.0, dy - radii[1]))


def _distance_to_ellipse(point: Point, center: Point, radii: Point) -> float:
    x = abs(point[0] - center[0])
    y = abs(point[1] - center[1])
    rx, ry = radii
    if rx <= 0.0 or ry <= 0.0:
        return math.hypot(x - rx, y - ry)
    if x * x / (rx * rx) + y * y / (ry * ry) <= 1.0:
        return 0.0

    def ellipse_distance(parameter: float) -> float:
        return math.hypot(
            rx * math.cos(parameter) - x,
            ry * math.sin(parameter) - y,
        )

    best_distance = float("inf")
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for window in range(32):
        left_t = window * math.pi / 64.0
        right_t = min(
            math.pi / 2.0,
            (window + 1) * math.pi / 64.0,
        )
        inner_left = right_t - golden * (right_t - left_t)
        inner_right = left_t + golden * (right_t - left_t)
        distance_left = ellipse_distance(inner_left)
        distance_right = ellipse_distance(inner_right)
        for _ in range(30):
            if distance_left < distance_right:
                right_t = inner_right
                inner_right = inner_left
                distance_right = distance_left
                inner_left = right_t - golden * (
                    right_t - left_t
                )
                distance_left = ellipse_distance(inner_left)
            else:
                left_t = inner_left
                inner_left = inner_right
                distance_left = distance_right
                inner_right = left_t + golden * (
                    right_t - left_t
                )
                distance_right = ellipse_distance(inner_right)
        best_distance = min(
            best_distance,
            distance_left,
            distance_right,
            ellipse_distance(left_t),
            ellipse_distance(right_t),
        )

    return best_distance


def _distance_to_cubic(point: Point, cubic: Cubic) -> float:
    best_squared = min(
        _distance_squared_between(_cubic_at(cubic, parameter), point)
        for parameter in (
            sample / 32.0 for sample in range(33)
        )
    )
    stack = [(cubic, 0, 0.0, 1.0)]
    while stack:
        current, depth, t0, t1 = stack.pop()
        current_bounds = _cubic_bounds(current)
        lower_bound = _point_bounds_distance_squared(
            point, current_bounds
        )
        if lower_bound > best_squared:
            continue
        if depth >= 5:
            for initial_t in (
                t0,
                (3.0 * t0 + t1) / 4.0,
                (t0 + t1) / 2.0,
                (t0 + 3.0 * t1) / 4.0,
                t1,
            ):
                candidate_t = _newton_cubic_parameter(
                    cubic, point, initial_t, t0, t1
                )
                if candidate_t is not None:
                    candidate_point = _cubic_at(cubic, candidate_t)
                    candidate_squared = _distance_squared_between(
                        candidate_point, point
                    )
                    if candidate_squared < best_squared:
                        best_squared = candidate_squared
            continue
        midpoint_t = (t0 + t1) / 2.0
        left, right = _subdivide_cubic(current)
        stack.append((left, depth + 1, t0, midpoint_t))
        stack.append((right, depth + 1, midpoint_t, t1))
    return math.sqrt(best_squared)


def _newton_cubic_parameter(
    cubic: Cubic,
    point: Point,
    initial_t: float,
    lower_t: float,
    upper_t: float,
) -> Optional[float]:
    t = initial_t
    for _ in range(12):
        curve, first_derivative, second_derivative = _cubic_derivatives(
            cubic, t
        )
        delta = (curve[0] - point[0], curve[1] - point[1])
        function_value = (
            first_derivative[0] * delta[0]
            + first_derivative[1] * delta[1]
        )
        derivative_value = (
            second_derivative[0] * delta[0]
            + second_derivative[1] * delta[1]
            + first_derivative[0] ** 2
            + first_derivative[1] ** 2
        )
        if abs(derivative_value) < 1e-14:
            break
        next_t = min(
            upper_t,
            max(lower_t, t - function_value / derivative_value),
        )
        if (
            abs(next_t - t) < 1e-12
            and lower_t < next_t < upper_t
        ):
            return next_t
        t = next_t
    return t if lower_t <= t <= upper_t else None


def _cubic_derivatives(
    cubic: Cubic, t: float
) -> Tuple[Point, Point, Point]:
    p0, p1, p2, p3 = cubic
    inverse = 1.0 - t
    curve = (
        _cubic_value(tuple(point[0] for point in cubic), t),
        _cubic_value(tuple(point[1] for point in cubic), t),
    )
    first = (
        3.0
        * (
            inverse**2 * (p1[0] - p0[0])
            + 2.0 * inverse * t * (p2[0] - p1[0])
            + t**2 * (p3[0] - p2[0])
        ),
        3.0
        * (
            inverse**2 * (p1[1] - p0[1])
            + 2.0 * inverse * t * (p2[1] - p1[1])
            + t**2 * (p3[1] - p2[1])
        ),
    )
    second = (
        6.0
        * (
            inverse * (p2[0] - 2.0 * p1[0] + p0[0])
            + t * (p3[0] - 2.0 * p2[0] + p1[0])
        ),
        6.0
        * (
            inverse * (p2[1] - 2.0 * p1[1] + p0[1])
            + t * (p3[1] - 2.0 * p2[1] + p1[1])
        ),
    )
    return curve, first, second


def _point_bounds_distance_squared(point: Point, bounds: Bounds) -> float:
    dx = max(bounds[0] - point[0], 0.0, point[0] - bounds[2])
    dy = max(bounds[1] - point[1], 0.0, point[1] - bounds[3])
    return dx * dx + dy * dy


def _distance_squared_between(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _cubic_at(cubic: Cubic, t: float) -> Point:
    return (
        _cubic_value(tuple(point[0] for point in cubic), t),
        _cubic_value(tuple(point[1] for point in cubic), t),
    )
