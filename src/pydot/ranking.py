# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Finite constraint checking for rank layout.

Before a layout is submitted, the rank semantics requested through DOT
attributes can be checked independently of a layout engine. This module
currently supports, for directed graphs only:

- ``rank="same"`` groups, which force all members onto one layer;
- edge ``minlen`` constraints with non-negative integer values, applied
  only when the edge's ``constraint`` attribute is enabled;
- edges declared with ``constraint=false`` are ignored, as are the
  ``node``/``edge``/``graph`` default-attribute statements as such
  (though ``edge [...]`` defaults still apply to the following edges);

Every layer constraint states a minimum difference between integer
layer numbers: an enabled edge from ``u`` to ``v`` with ``minlen=n``
requires ``layer(v) - layer(u) >= n``. A ``rank="same"`` group with
members ``m1, m2, ...`` requires the members to share a layer, which is
modelled as ``layer(a) == layer(b)`` (differences of zero in both
directions).

When the constraints are feasible, any satisfying integer assignment is
returned; it does not try to reproduce Graphviz's visual output. When
they are infeasible, the positive-weight cycles of the associated
difference-constraint graph are reported. Each conflict is a chain
whose steps can be independently checked against the constraints that
caused it; parallel edges keep their own identity (every declared edge
gets its own index and may carry a DOT ``id``).

Rank semantics other than ``same`` (``min``, ``source``, ``max``,
``sink``), undirected graphs, and edges routed through logical
``lhead``/``ltail`` endpoints are reported as unsupported instead of
being guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from pydot.classes import AttributeDict, FrozenDict

if TYPE_CHECKING:
    from pydot.core import Graph


_RESERVED_NODE_NAMES = frozenset(("node", "edge", "graph"))
_SUPPORTED_RANK = "same"


class ConstraintStatus(str, Enum):
    """Outcome of a rank constraint check.

    FEASIBLE
        The requested constraints are consistent; ``layers`` holds a
        satisfying integer assignment.
    INFEASIBLE
        The constraints contradict each other; ``conflicts`` holds the
        independently verifiable positive-weight cycles.
    UNSUPPORTED
        The graph uses rank semantics this checker does not implement;
        ``unsupported`` describes them. No assignment is produced and
        nothing is guessed.
    """

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class LayerConstraint:
    """One minimum layer difference from a declared edge.

    ``tail`` and ``head`` name nodes before same-rank merging; the
    solver merges them. An edge into an anonymous subgraph endpoint is
    expanded into one constraint per member; such constraints share an
    ``edge_index``, but every declared edge keeps an identity of its
    own.

    Attributes:
        tail: name of the edge's source node.
        head: name of the edge's destination node.
        minlen: required minimum ``layer(head) - layer(tail)``.
        edge_index: 1-based index of the declaring edge in declaration
            order across the whole graph and its subgraphs.
        edge_id: value of the edge's DOT ``id`` attribute, if any.
        expanded: True when this constraint was produced by expanding
            an anonymous subgraph endpoint; ``tail``/``head`` then name
            an expansion pair rather than the literal endpoints.
        source: literal source endpoint as declared.
        destination: literal destination endpoint as declared.
    """

    tail: str
    head: str
    minlen: int
    edge_index: int
    edge_id: Optional[str] = None
    expanded: bool = False
    source: Optional[str] = None
    destination: Optional[str] = None


@dataclass(frozen=True)
class SameRankGroup:
    """One ``rank="same"`` group.

    Attributes:
        index: 1-based index in the order the group was encountered.
        members: member node names in first-declaration order.
        location: human-readable location of the group statement.
    """

    index: int
    members: Tuple[str, ...]
    location: str


@dataclass(frozen=True)
class UnsupportedRank:
    """One piece of rank semantics this checker does not implement.

    Attributes:
        kind: machine-readable reason, one of ``"undirected_graph"``,
            ``"rank_value"``, ``"rank_on_graph"``, ``"logical_endpoint"``.
        detail: human-readable explanation.
        location: where the unsupported construct was encountered.
    """

    kind: str
    detail: str
    location: str


@dataclass(frozen=True)
class ConflictStep:
    """One edge in a conflict chain.

    A step is an arc ``tail -> head`` carrying a required minimum layer
    difference. It originates either from a declared edge constraint or
    from one direction of a ``rank="same"`` equality.

    Attributes:
        tail: node the step starts from.
        head: node the step leads to.
        min_difference: required minimum ``layer(head) - layer(tail)``.
        kind: ``"edge"`` or ``"same_rank"``.
        edge_index: declaring edge index for edge steps, else None.
        edge_id: declaring edge DOT ``id`` for edge steps, if present.
        group_index: rank group index for same-rank steps, else None.
        expanded: whether an edge step resulted from subgraph endpoint
            expansion.
        description: short human-readable description of the source.
    """

    tail: str
    head: str
    min_difference: int
    kind: str
    edge_index: Optional[int] = None
    edge_id: Optional[str] = None
    group_index: Optional[int] = None
    expanded: bool = False
    description: str = ""


@dataclass(frozen=True)
class ConflictChain:
    """A contradiction given as a positive-weight cycle.

    Following the steps in order returns to ``start_node``. Summing the
    steps' minimum differences gives ``total_difference``, which is
    strictly positive; applying them around the cycle would require
    ``layer(start_node) >= layer(start_node) + total_difference``,
    which no integer layer assignment can satisfy.

    Attributes:
        start_node: node the chain starts and ends at.
        steps: the constraints that jointly cause the contradiction.
        total_difference: sum of the steps' minimum differences.
    """

    start_node: str
    steps: Tuple[ConflictStep, ...]
    total_difference: int


@dataclass(frozen=True)
class RankAssignmentResult:
    """Result of :func:`check_rank_constraints`.

    Attributes:
        status: outcome of the check.
        layers: integer layer per node name when feasible, else None.
        constraints: minimum layer-difference constraints derived from
            enabled edges, in declaration order.
        same_rank_groups: the ``rank="same"`` groups found.
        conflicts: positive-weight cycles proving infeasibility.
        unsupported: rank semantics this checker does not implement.
    """

    status: ConstraintStatus
    layers: Optional[Dict[str, int]] = None
    constraints: Tuple[LayerConstraint, ...] = ()
    same_rank_groups: Tuple[SameRankGroup, ...] = ()
    conflicts: Tuple[ConflictChain, ...] = ()
    unsupported: Tuple[UnsupportedRank, ...] = ()


# ---------------------------------------------------------------------------
# DOT attribute interpretation
# ---------------------------------------------------------------------------


def _strip_port(name: str) -> str:
    """Strip a trailing ``:port`` compass suffix like Node creation does."""

    if name.startswith('"'):
        return name
    idx = name.find(":")
    if idx > 0 and idx + 1 < len(name):
        return name[:idx]
    return name


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _parse_bool(value: Any, what: str, where: str) -> bool:
    """Interpret a DOT boolean attribute value."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = _unquote(value).strip().lower()
        if text in ("true", "yes", "1", "on"):
            return True
        if text in ("false", "no", "0", "off"):
            return False
    raise ValueError(
        f"Invalid boolean value {value!r} for {what} at {where}; "
        'expected "true" or "false".'
    )


def _parse_minlen(value: Any, where: str) -> int:
    """Interpret a non-negative integer ``minlen`` attribute value."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"Invalid minlen value {value!r} at {where}; "
            "expected a non-negative integer."
        )
    text = value if isinstance(value, int) else _unquote(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        parsed = -1
    if parsed < 0:
        raise ValueError(
            f"Invalid minlen value {value!r} at {where}; "
            "expected a non-negative integer."
        )
    return parsed


# ---------------------------------------------------------------------------
# Model collection
# ---------------------------------------------------------------------------


def _container_label(obj: Dict[str, Any]) -> str:
    """Stable, human-readable label for a graph/subgraph obj_dict."""

    ctype = obj.get("type", "subgraph")
    name = obj.get("name")
    sequence = obj.get("sequence")
    if ctype in ("graph", "digraph"):
        return f"{ctype} {name}".rstrip()
    if name:
        return f"subgraph {name}"
    return f"anonymous subgraph (sequence {sequence})"


def _sorted_objs(store: Dict[Any, Any]) -> Tuple[Dict[str, Any], ...]:
    """Flatten an obj_dict node/edge/subgraph store, by sequence."""

    objs: List[Dict[str, Any]] = []
    for obj_list in store.values():
        objs.extend(obj_list)
    objs.sort(key=lambda obj: obj.get("sequence", 0))
    return tuple(objs)


def _endpoint_node(endpoint: Any) -> Optional[str]:
    """Return the node name of a literal endpoint.

    Anonymous subgraph endpoints (``FrozenDict``) return None; their
    members are resolved separately.
    """

    if isinstance(endpoint, FrozenDict):
        return None
    if isinstance(endpoint, (int, float)):
        endpoint = str(endpoint)
    return _strip_port(endpoint)


def _collect_members(
    container: Dict[str, Any], expand_edges_within: bool
) -> Tuple[str, ...]:
    """Node names belonging to a (sub)graph tree.

    Used in two contexts:

    - the members of a ``rank`` group, or the nodes an anonymous edge
      endpoint expands to: declared nodes and, when
      ``expand_edges_within`` is True, the endpoints of edges within
      the tree;
    - a nested anonymous endpoint found through such an edge: only its
      declared nodes (``expand_edges_within=False``), since its own
      edges are visited as separate graph edges instead.
    """

    members: List[str] = []
    seen = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            members.append(name)

    def walk(obj: Dict[str, Any], edges_within: bool) -> None:
        for node_obj in _sorted_objs(obj["nodes"]):
            name = node_obj.get("name")
            if isinstance(name, str) and name not in _RESERVED_NODE_NAMES:
                add(name)
        if edges_within:
            for edge_obj in _sorted_objs(obj["edges"]):
                for point in edge_obj["points"]:
                    name = _endpoint_node(point)
                    if name is not None:
                        add(name)
                    else:
                        for member in _collect_members(point, False):
                            add(member)
        for sub_obj in _sorted_objs(obj["subgraphs"]):
            walk(sub_obj, edges_within)

    walk(container, expand_edges_within)
    return tuple(members)


class _Collector:
    """Walks the pydot object model and gathers rank constructs."""

    def __init__(self) -> None:
        self.nodes: List[str] = []
        self.node_seen: set[str] = set()
        self.constraints: List[LayerConstraint] = []
        self.groups: List[SameRankGroup] = []
        self.unsupported: List[UnsupportedRank] = []
        self._edge_index = 0
        self._edge_defaults: List[Dict[str, Any]] = []

    def note_node(self, name: str) -> None:
        if name not in self.node_seen:
            self.node_seen.add(name)
            self.nodes.append(name)

    def note_rank(self, container: Dict[str, Any], root: bool) -> None:
        attrs = container.get("attributes", {})
        if "rank" not in attrs:
            return
        raw = attrs["rank"]
        value = (
            _unquote(raw).strip().lower()
            if isinstance(raw, str)
            else raw
        )
        label = _container_label(container)
        if root:
            self.unsupported.append(
                UnsupportedRank(
                    "rank_on_graph",
                    "The rank attribute is only interpreted on "
                    f"subgraphs; found rank={raw!r} on the graph itself.",
                    label,
                )
            )
            return
        if value != _SUPPORTED_RANK:
            self.unsupported.append(
                UnsupportedRank(
                    "rank_value",
                    f"rank={raw!r} is not supported; only "
                    f'rank="{_SUPPORTED_RANK}" is implemented. '
                    "Other rank semantics are not guessed at.",
                    label,
                )
            )

    def collect_edge(self, edge_obj: Dict[str, Any]) -> None:
        # Edges inside anonymous endpoint subgraphs are part of the
        # graph and precede the edge that names the endpoint.
        for point in edge_obj["points"]:
            if isinstance(point, FrozenDict):
                self.visit(point)

        self._edge_index += 1
        index = self._edge_index
        attrs = self._effective_edge_attrs(edge_obj.get("attributes", {}))
        points = edge_obj["points"]
        where = f"edge #{index}"

        source_ep, destination_ep = points[0], points[1]
        source = _endpoint_node(source_ep)
        destination = _endpoint_node(destination_ep)

        raw_constraint = attrs.get("constraint", "true")
        enabled = _parse_bool(raw_constraint, "constraint", where)
        minlen = _parse_minlen(attrs.get("minlen", 1), where)
        edge_id = attrs.get("id")
        if edge_id is not None:
            edge_id = _unquote(str(edge_id))

        if (
            attrs.get("lhead") is not None
            or attrs.get("ltail") is not None
        ):
            self.unsupported.append(
                UnsupportedRank(
                    "logical_endpoint",
                    "Edges routed with lhead/ltail do not express a "
                    "direct layer difference; their rank effect is not "
                    "implemented.",
                    where,
                )
            )
            return

        src_members = (
            (source,)
            if source is not None
            else _collect_members(source_ep, True)
        )
        dst_members = (
            (destination,)
            if destination is not None
            else _collect_members(destination_ep, True)
        )
        for name in src_members:
            self.note_node(name)
        for name in dst_members:
            self.note_node(name)

        if not enabled:
            return

        literal_source = source if source is not None else "{...}"
        literal_destination = (
            destination if destination is not None else "{...}"
        )
        expanded = source is None or destination is None
        for tail in src_members:
            for head in dst_members:
                self.constraints.append(
                    LayerConstraint(
                        tail=tail,
                        head=head,
                        minlen=minlen,
                        edge_index=index,
                        edge_id=edge_id,
                        expanded=expanded,
                        source=literal_source,
                        destination=literal_destination,
                    )
                )

    def _effective_edge_attrs(
        self, own: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for defaults in self._edge_defaults:
            merged.update(defaults)
        merged.update(own)
        return merged

    def visit(self, container: Dict[str, Any], root: bool = False) -> None:
        """Depth-first visit in declaration order.

        Children of every container are ordered by their declaration
        sequence, which also fixes the edge indices. Edges inside
        anonymous subgraphs used as edge endpoints are visited when the
        containing edge is encountered, mirroring DOT semantics.
        """

        self.note_rank(container, root)
        rank_value = container.get("attributes", {}).get("rank")
        same_rank_here = (
            not root
            and rank_value is not None
            and _unquote(str(rank_value)).strip().lower() == _SUPPORTED_RANK
        )

        # Everything is visited in declaration (sequence) order so that
        # edge indices and group indices are stable.
        children = (
            _sorted_objs(container["nodes"])
            + _sorted_objs(container["edges"])
            + _sorted_objs(container["subgraphs"])
        )

        defaults_before = len(self._edge_defaults)
        for obj in children:
            obj_type = obj.get("type")
            if obj_type == "node":
                name = obj.get("name")
                if name == "edge":
                    self._edge_defaults.append(
                        dict(obj.get("attributes", {}))
                    )
                elif (
                    isinstance(name, str)
                    and name not in _RESERVED_NODE_NAMES
                ):
                    self.note_node(name)
            elif obj_type == "edge":
                self.collect_edge(obj)
            else:
                self.visit(obj)

        del self._edge_defaults[defaults_before:]

        if same_rank_here:
            members = _collect_members(container, True)
            for name in members:
                self.note_node(name)
            self.groups.append(
                SameRankGroup(
                    index=len(self.groups) + 1,
                    members=members,
                    location=_container_label(container),
                )
            )


# ---------------------------------------------------------------------------
# Difference-constraint solver
# ---------------------------------------------------------------------------


def _solve(
    nodes: Sequence[str],
    constraints: Sequence[LayerConstraint],
    groups: Sequence[SameRankGroup],
) -> Tuple[Optional[Dict[str, int]], Tuple[ConflictChain, ...]]:
    """Solve ``layer(v) - layer(u) >= w`` plus same-rank equalities.

    Returns ``(layers, ())`` when feasible or ``(None, cycles)`` when a
    positive-weight cycle makes the system unsatisfiable.
    """

    # Every arc gets a unique serial, so parallel declarations and the
    # two directions of a same-rank equality stay distinguishable. The
    # serial also fixes a deterministic iteration order.
    arcs: List[Tuple[str, str, int, int]] = []
    provenance: Dict[int, Any] = {}

    def add_arc(
        tail: str, head: str, weight: int, origin: Any
    ) -> None:
        serial = len(arcs)
        arcs.append((tail, head, weight, serial))
        provenance[serial] = origin

    for constraint in constraints:
        add_arc(
            constraint.tail,
            constraint.head,
            constraint.minlen,
            ("edge", constraint),
        )
    for group in groups:
        members = group.members
        for other in members[1:]:
            add_arc(members[0], other, 0, ("same", group, 1))
            add_arc(other, members[0], 0, ("same", group, -1))

    distance: Dict[str, int] = {name: 0 for name in nodes}
    predecessor: Dict[str, Tuple[str, str, int, int]] = {}

    count = len(nodes)
    # At most count-1 relaxation rounds are needed to reach the longest
    # paths when no positive-weight cycle exists.
    for _ in range(max(count - 1, 0)):
        changed = False
        for tail, head, weight, identity in arcs:
            candidate = distance[tail] + weight
            if candidate > distance[head]:
                distance[head] = candidate
                predecessor[head] = (tail, head, weight, identity)
                changed = True
        if not changed:
            break

    # An extra dry sweep finds arcs that can still improve. Such arcs
    # reach a positive-weight cycle; distances and predecessors are
    # deliberately left untouched so the cycle can be reconstructed.
    relaxing = [
        arc
        for arc in arcs
        if distance[arc[0]] + arc[2] > distance[arc[1]]
    ]

    if not relaxing:
        return dict(distance), ()

    conflicts = _extract_cycles(nodes, relaxing, predecessor, provenance)
    return None, tuple(conflicts)


def _extract_cycles(
    nodes: Sequence[str],
    relaxing: Sequence[Tuple[str, str, int, int]],
    predecessor: Dict[str, Tuple[str, str, int, int]],
    provenance: Dict[int, Any],
) -> List[ConflictChain]:
    """Reconstruct positive-weight cycles reachable from relaxing arcs.

    Following predecessor pointers from the head of each relaxing arc
    must repeat a node within ``len(nodes)`` hops; the repeated segment
    is a positive-weight cycle.
    """

    seen_cycles: set[frozenset[int]] = set()
    chains: List[ConflictChain] = []

    for relaxed in sorted(relaxing, key=lambda arc: arc[3]):
        # A relaxing self-loop is itself a positive-weight cycle.
        if relaxed[0] == relaxed[1]:
            identities = frozenset((relaxed[3],))
            if identities not in seen_cycles:
                seen_cycles.add(identities)
                steps = (
                    _step_from_arc(
                        relaxed, provenance[relaxed[3]]
                    ),
                )
                chains.append(
                    ConflictChain(
                        start_node=relaxed[0],
                        steps=steps,
                        total_difference=relaxed[2],
                    )
                )
            continue

        walk: List[Tuple[str, str, int, int]] = [relaxed]
        # Node -> its position as a walk endpoint: the head of walk[0]
        # is position 0, the tail of walk[k] is position k.
        positions: Dict[str, int] = {relaxed[1]: 0}
        current = relaxed[0]

        repeat_at: Optional[int] = None
        while len(walk) <= len(nodes) and current in predecessor:
            arc = predecessor[current]
            walk.append(arc)
            tail = arc[0]
            tail_position = len(walk) - 1
            if tail in positions:
                repeat_at = positions[tail]
                break
            positions[tail] = tail_position
            current = tail

        if repeat_at is None:
            # No predecessor chain on a cycle; another relaxing arc
            # will reach it.
            continue

        # The repeated node is the endpoint at repeat_at. The cycle
        # follows walk arcs from repeat_at+1 onwards; when repeat_at is
        # 0 the relaxing edge itself closes the loop.
        forward = list(reversed(walk[repeat_at + 1:]))
        if repeat_at == 0:
            forward.append(walk[0])
        start_node = forward[0][0]
        identities = frozenset(arc[3] for arc in forward)
        if identities in seen_cycles:
            continue
        seen_cycles.add(identities)

        steps = tuple(
            _step_from_arc(arc, provenance[arc[3]]) for arc in forward
        )
        total = sum(step.min_difference for step in steps)
        chains.append(
            ConflictChain(
                start_node=start_node,
                steps=steps,
                total_difference=total,
            )
        )

    chains.sort(key=lambda chain: _chain_key(chain))
    return chains


def _step_from_arc(
    arc: Tuple[str, str, int, int], origin: Any
) -> ConflictStep:
    tail, head, weight = arc[0], arc[1], arc[2]
    if origin[0] == "edge":
        constraint = origin[1]
        return ConflictStep(
            tail=tail,
            head=head,
            min_difference=weight,
            kind="edge",
            edge_index=constraint.edge_index,
            edge_id=constraint.edge_id,
            expanded=constraint.expanded,
            description=(
                f"edge #{constraint.edge_index}: "
                f"{constraint.source} -> {constraint.destination} "
                f"(minlen={constraint.minlen})"
            ),
        )
    group = origin[1]
    return ConflictStep(
        tail=tail,
        head=head,
        min_difference=weight,
        kind="same_rank",
        group_index=group.index,
        description=(
            f"rank=same group #{group.index} ({group.location}): "
            f"layers of {tail} and "
            f"{head} are equal"
        ),
    )


def _chain_key(chain: ConflictChain) -> Tuple[Any, ...]:
    edge_indices = tuple(
        step.edge_index
        for step in chain.steps
        if step.edge_index is not None
    )
    group_indices = tuple(
        step.group_index
        for step in chain.steps
        if step.group_index is not None
    )
    return (
        min(edge_indices) if edge_indices else float("inf"),
        edge_indices,
        group_indices,
        chain.start_node,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_rank_constraints(graph: "Graph") -> RankAssignmentResult:
    """Check finite rank constraints on a directed graph.

    Merges every ``rank="same"`` group and turns each edge with
    ``constraint`` enabled into a minimum layer difference of
    ``minlen`` (default 1, which must be a non-negative integer).
    Parallel edges are kept as separate constraints. Undeclared edge
    endpoints are handled as implicit nodes.

    Only a limited, finite subset of the rank semantics is checked:

    - the graph must be a directed graph (``digraph``);
    - only ``rank="same"`` is interpreted (``min``, ``source``, ``max``
      and ``sink`` are reported as unsupported);
    - edges using ``lhead``/``ltail`` logical endpoints are reported as
      unsupported.

    Args:
        graph: a ``pydot.Dot`` or ``pydot.Graph`` of type ``digraph``.

    Returns:
        A :class:`RankAssignmentResult`. When feasible it contains a
        dictionary of integer layer numbers keyed by node name; when
        infeasible it contains conflict chains that can be checked
        independently; when unsupported it describes the semantics the
        checker does not implement.

    Raises:
        TypeError: if ``graph`` is not a pydot graph object.
        ValueError: if an attribute value is malformed (for example a
            negative or non-integer ``minlen``).
    """

    from pydot.core import Graph as _Graph

    if not isinstance(graph, _Graph):
        raise TypeError(
            "check_rank_constraints() expects a pydot.Graph, got "
            f"{type(graph).__name__}."
        )

    graph_type = graph.get_graph_type()
    if graph_type != "digraph":
        return RankAssignmentResult(
            status=ConstraintStatus.UNSUPPORTED,
            unsupported=(
                UnsupportedRank(
                    kind="undirected_graph",
                    detail=(
                        "Rank layering is a directed-graph concept; "
                        f'this graph has type "{graph_type}". The '
                        "checker does not infer an orientation."
                    ),
                    location=_container_label(graph.obj_dict),
                ),
            ),
        )

    collector = _Collector()
    collector.visit(graph.obj_dict, root=True)

    if collector.unsupported:
        return RankAssignmentResult(
            status=ConstraintStatus.UNSUPPORTED,
            constraints=tuple(collector.constraints),
            same_rank_groups=tuple(collector.groups),
            unsupported=tuple(collector.unsupported),
        )

    layers, conflicts = _solve(
        tuple(collector.nodes),
        tuple(collector.constraints),
        tuple(collector.groups),
    )

    if conflicts:
        return RankAssignmentResult(
            status=ConstraintStatus.INFEASIBLE,
            constraints=tuple(collector.constraints),
            same_rank_groups=tuple(collector.groups),
            conflicts=conflicts,
        )

    return RankAssignmentResult(
        status=ConstraintStatus.FEASIBLE,
        layers=layers,
        constraints=tuple(collector.constraints),
        same_rank_groups=tuple(collector.groups),
    )
