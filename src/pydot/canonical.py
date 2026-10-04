# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Canonical representation and digest for finite flattened graphs.

Two graphs that are isomorphic -- allowing for node renaming and
reordering of nodes, edges and declared attributes -- produce the same
canonical representation string and digest. Only the node and edge
attributes explicitly declared as semantic participate in the
comparison; node identifiers never do.

The implementation is an individualization/refinement (IR) search over
ordered partitions, in the style of nauty/Bliss:

1. Nodes start in one colour class per exact node-attribute token.
2. Deterministic colour refinement (ordered-partition 1-WL) splits
   classes using the cell positions of neighbours until stable.
3. A non-discrete partition is individualized: each member of one
   ambiguous cell is, in turn, fixed into a singleton cell, and
   refinement plus backtracking continue.
4. Every discrete leaf induces a node permutation; the graph is
   serialized under that permutation and the lexicographically smallest
   serialization is the canonical representation.

Plain neighbour hashing cannot separate symmetric structures; the
backtracking search explores the alternative labelings instead. Every
refinement iteration and every individualized branch is charged
against a search budget. Exhausting the budget yields an *incomplete*
result -- never a result falsely claiming canonicity.

Parallel edges and self-loops are first-class elements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import pydot

__all__ = [
    "CanonicalForm",
    "canonicalize",
    "canonicalize_graph",
    "encode_attrs",
]

SCHEMA_TAG: Final = "pydot-canonical-v1"
# A present attribute with value ``None`` must not collide with a
# missing attribute or with the string "null".
_NONE_TAG: Final = "\x00missing"
_SCALAR_TAGS: Final = {
    bool: "b",
    int: "i",
    float: "f",
    str: "s",
    type(None): "z",
}


def _stable(value: Any) -> Any:
    """Return a deterministically ordered form of *value*."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "surrogateescape")
    if isinstance(value, Mapping):
        return {
            str(key): _stable(value[key]) for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_stable(item) for item in sorted(value, key=encode_attrs)]
    return repr(value)


def encode_attrs(
    attrs: Mapping[str, Any] | None,
    semantic: Iterable[str] = (),
) -> str:
    """Encode selected attributes into an order-free token.

    Only names in *semantic* participate; their declaration order is
    irrelevant. A missing name is encoded distinctly from one present
    with value ``None``. Mappings and sets are sorted deterministically
    before JSON encoding.
    """
    selected = attrs if attrs is not None else {}
    payload: list[tuple[str, str, Any]] = []
    for name in sorted(set(semantic)):
        if name in selected:
            value = selected[name]
            kind = _SCALAR_TAGS.get(type(value), "j")
            if kind == "j":
                value = _stable(value)
            payload.append((name, kind, value))
        else:
            payload.append((name, _NONE_TAG, None))
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=repr,
    )


@dataclass(frozen=True)
class CanonicalForm:
    """Outcome of a canonicalization run.

    When ``complete`` is false, ``representation``, ``digest`` and
    ``mapping`` are empty. Do not cache on an incomplete digest; rerun
    with a larger ``budget`` or with more distinguishing attributes.
    """

    complete: bool
    representation: str | None = None
    digest: str | None = None
    node_count: int = 0
    edge_count: int = 0
    directed: bool = True
    budget: int | None = None
    budget_used: int = 0
    mapping: Mapping[str, int] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """``"complete"`` or ``"incomplete"``."""
        return "complete" if self.complete else "incomplete"


_Partition = tuple[tuple[int, ...], ...]
_Edge = tuple[int, int, str]


def _node_signatures(
    size: int,
    colours: Sequence[str],
    cell_of: Sequence[int],
    edges: Sequence[_Edge],
    directed: bool,
) -> list[str]:
    """Ordered-partition refinement signature per node.

    Each incident edge contributes a token carrying its direction, the
    neighbour's cell position and the edge's semantic token. A node's
    signature is its colour plus the sorted multiset of those tokens.
    Cell positions (rather than neighbour colours) make an
    individualized singleton propagate through the partition.
    """
    buckets: list[list[str]] = [[] for _ in range(size)]
    for source, target, token in edges:
        buckets[source].append(f">{cell_of[target]}:{token}")
        if directed:
            buckets[target].append(f"<{cell_of[source]}:{token}")
        elif source != target:
            buckets[target].append(f">{cell_of[source]}:{token}")
    return [
        colours[node] + "|" + "".join(sorted(buckets[node]))
        for node in range(size)
    ]


def _refine(
    partition: _Partition,
    colours: Sequence[str],
    edges: Sequence[_Edge],
    directed: bool,
) -> tuple[_Partition, list[str]]:
    size = len(colours)
    cell_of = [0] * size
    for position, cell in enumerate(partition):
        for node in cell:
            cell_of[node] = position
    signatures = _node_signatures(
        size, colours, cell_of, edges, directed
    )

    next_partition: list[tuple[int, ...]] = []
    next_colours: list[str] = [""] * size
    for cell in partition:
        groups: dict[str, list[int]] = {}
        for node in cell:
            groups.setdefault(signatures[node], []).append(node)
        for signature in sorted(groups):
            members = tuple(sorted(groups[signature]))
            next_partition.append(members)
            for node in members:
                next_colours[node] = signature
    return tuple(next_partition), next_colours


def _stabilize(
    partition: _Partition,
    colours: Sequence[str],
    edges: Sequence[_Edge],
    directed: bool,
    spend: Callable[[], bool],
) -> tuple[_Partition, list[str], bool]:
    """Refine until stable; last element flags budget exhaustion."""
    current = partition
    current_colours = list(colours)
    previous: _Partition | None = None
    while previous != current:
        if spend():
            return current, current_colours, True
        previous = current
        current, current_colours = _refine(
            current, current_colours, edges, directed
        )
    return current, current_colours, False


def _individualize(
    partition: _Partition, cell: tuple[int, ...], marker: int
) -> _Partition:
    result: list[tuple[int, ...]] = []
    for current in partition:
        if current is cell:
            result.append((marker,))
            rest = tuple(node for node in cell if node != marker)
            if rest:
                result.append(rest)
        else:
            result.append(current)
    return tuple(result)


def _canonical_text(
    order: Sequence[int],
    node_tokens: Sequence[str],
    edges: Sequence[_Edge],
    directed: bool,
) -> str:
    position = {node: index for index, node in enumerate(order)}
    lines = [
        SCHEMA_TAG,
        f"directed={int(directed)}",
        f"nodes={len(order)}",
        f"edges={len(edges)}",
    ]
    lines.extend(
        f"n{position[node]}={node_tokens[node]}"
        for node in sorted(position, key=lambda n: position[n])
    )
    remapped = [
        (position[source], position[target], token)
        for source, target, token in edges
    ]
    if not directed:
        remapped = [
            (a, b, token) if a <= b else (b, a, token)
            for a, b, token in remapped
        ]
    lines.extend(
        f"e{source}>{target}={token}"
        for source, target, token in sorted(remapped)
    )
    return "\n".join(lines)


def canonicalize(
    nodes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[tuple[Any, ...]],
    *,
    node_attrs: Iterable[str] = (),
    edge_attrs: Iterable[str] = (),
    directed: bool = True,
    budget: int | None = 10000,
) -> CanonicalForm:
    """Build a canonical form of a finite, flattened graph.

    Parameters
    ----------
    nodes:
        Mapping of node identifier to its attribute mapping.
    edges:
        ``(source, target)`` or ``(source, target, attributes)``
        tuples. Parallel edges and self-loops are supported; endpoints
        must be keys of *nodes*.
    node_attrs, edge_attrs:
        Attribute names that carry semantic meaning. Declaration order
        is ignored; unlisted attributes and node identifiers are
        ignored.
    directed:
        Whether edge endpoints are ordered.
    budget:
        Maximum charged operations (refinement iterations plus
        individualized branches); ``None`` means unlimited. Exceeding
        the budget returns an incomplete :class:`CanonicalForm`.
    """
    names = sorted(nodes)
    index = {name: i for i, name in enumerate(names)}
    size = len(names)
    node_attr_names = tuple(sorted(set(node_attrs)))
    edge_attr_names = tuple(sorted(set(edge_attrs)))

    node_tokens = [
        encode_attrs(nodes[name], node_attr_names) for name in names
    ]

    internal_edges: list[_Edge] = []
    for edge in edges:
        if len(edge) == 2:
            source_name, target_name = edge
            edge_attributes: Mapping[str, Any] = {}
        elif len(edge) == 3:
            source_name, target_name, edge_attributes = edge
        else:
            raise ValueError("edges must be (source, target[, attrs])")
        if source_name not in index or target_name not in index:
            raise ValueError(f"edge {edge!r} references an unknown node")
        internal_edges.append(
            (
                index[source_name],
                index[target_name],
                encode_attrs(edge_attributes, edge_attr_names),
            )
        )

    # The empty graph has a unique canonical form without any search,
    # regardless of the budget.
    if size == 0:
        text = _canonical_text((), (), internal_edges, directed)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return CanonicalForm(
            complete=True,
            representation=text,
            digest=digest,
            node_count=0,
            edge_count=len(internal_edges),
            directed=directed,
            budget=budget,
            budget_used=0,
        )

    used = 0

    def spend() -> bool:
        nonlocal used
        used += 1
        return budget is not None and used > budget

    def incomplete() -> CanonicalForm:
        return CanonicalForm(
            complete=False,
            node_count=size,
            edge_count=len(internal_edges),
            directed=directed,
            budget=budget,
            budget_used=used,
        )

    seed: dict[str, list[int]] = {}
    for node, token in enumerate(node_tokens):
        seed.setdefault(token, []).append(node)
    partition: _Partition = tuple(
        tuple(sorted(members))
        for _, members in sorted(seed.items())
    )
    colours = list(node_tokens)

    partition, colours, exhausted = _stabilize(
        partition, colours, internal_edges, directed, spend
    )
    if exhausted:
        return incomplete()

    best_text: str | None = None
    best_order: tuple[int, ...] | None = None

    # Stack entries are (partition, colours, already-stable).
    stack: list[tuple[_Partition, list[str], bool]] = [
        (partition, colours, True)
    ]
    while stack:
        current, current_colours, already_stable = stack.pop()
        if already_stable:
            stabilized, stabilized_colours = current, current_colours
        else:
            stabilized, stabilized_colours, exhausted = _stabilize(
                current,
                current_colours,
                internal_edges,
                directed,
                spend,
            )
            if exhausted:
                return incomplete()

        if all(len(cell) == 1 for cell in stabilized):
            order = tuple(cell[0] for cell in stabilized)
            text = _canonical_text(
                order, node_tokens, internal_edges, directed
            )
            if best_text is None or text < best_text:
                best_text = text
                best_order = order
            continue

        ambiguous = max(
            (cell for cell in stabilized if len(cell) > 1),
            key=len,
        )
        # Orbit pruning: if two candidate markers refine to the same
        # ordered partition they are in the same automorphism orbit and
        # lead to isomorphic sub-searches; explore only one of them.
        seen_orbits: set[tuple[int, ...]] = set()
        # Push in reverse so the lowest original index is explored
        # first; order never affects the result, only the trace.
        for marker in reversed(sorted(ambiguous)):
            if spend():
                return incomplete()
            individualized = _individualize(
                stabilized, ambiguous, marker
            )
            refined, refined_colours, exhausted = _stabilize(
                individualized,
                list(stabilized_colours),
                internal_edges,
                directed,
                spend,
            )
            if exhausted:
                return incomplete()
            # Cell colours are identity-free strings (they encode
            # neighbour cell positions and edge tokens), so two markers
            # whose refinement yields the same ordered cell shape are in
            # one automorphism orbit; trying both would repeat the same
            # sub-search. The marker singleton sits at the ambiguous
            # cell's position for every candidate, so ordered cell
            # comparison is valid here.
            orbit_key = tuple(
                (len(cell), refined_colours[cell[0]])
                for cell in refined
            )
            if orbit_key in seen_orbits:
                continue
            seen_orbits.add(orbit_key)
            stack.append(
                (individualized, list(stabilized_colours), False)
            )

    assert best_text is not None
    assert best_order is not None
    digest = hashlib.sha256(
        best_text.encode("utf-8", "surrogateescape")
    ).hexdigest()
    mapping = {
        names[node]: position
        for position, node in enumerate(best_order)
    }
    return CanonicalForm(
        complete=True,
        representation=best_text,
        digest=digest,
        node_count=size,
        edge_count=len(internal_edges),
        directed=directed,
        budget=budget,
        budget_used=used,
        mapping=mapping,
    )


def _endpoint_name(endpoint: Any) -> str:
    if isinstance(endpoint, str):
        # Strip an optional compass point ("node:port"); the finite
        # graph model identifies nodes by their DOT name.
        if not endpoint.startswith('"'):
            colon = endpoint.find(":")
            if colon > 0:
                return endpoint[:colon]
        return endpoint
    if isinstance(endpoint, bool):
        raise ValueError("boolean edge endpoints are not supported")
    if isinstance(endpoint, (int, float)):
        return str(endpoint)
    raise ValueError(
        f"unsupported edge endpoint {endpoint!r}; canonicalization "
        "expects an expanded graph whose edges connect named nodes"
    )


def canonicalize_graph(
    graph: pydot.Graph,
    *,
    node_attrs: Iterable[str] = (),
    edge_attrs: Iterable[str] = (),
    budget: int | None = 10000,
) -> CanonicalForm:
    """Canonicalize an expanded pydot graph.

    Node identifiers (DOT ``name``) are treated as non-semantic labels.
    Default-attribute holders (the pseudo-nodes ``node`` and ``edge``)
    are ignored, as are graph attributes. Edge endpoints that name no
    declared node contribute an implicit node with empty attributes,
    matching DOT semantics. Ports on endpoints are stripped. Subgraphs
    must be flattened by the caller (edges nested inside subgraphs are
    not part of the graph's direct edge list).
    """
    directed = graph.get_top_graph_type() != "graph"

    node_attributes: dict[str, dict[str, Any]] = {}
    for node in graph.get_nodes():
        name = node.get_name()
        if name in ("node", "edge", "graph"):
            continue
        node_attributes.setdefault(str(name), {}).update(
            node.get_attributes()
        )

    plain_edges: list[tuple[str, str, Mapping[str, Any]]] = []
    for edge in graph.get_edges():
        source = _endpoint_name(edge.get_source())
        target = _endpoint_name(edge.get_destination())
        node_attributes.setdefault(source, {})
        node_attributes.setdefault(target, {})
        plain_edges.append(
            (source, target, dict(edge.get_attributes()))
        )

    return canonicalize(
        node_attributes,
        plain_edges,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        directed=directed,
        budget=budget,
    )
