# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Canonical representations for finite, expanded graphs."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pydot

DEFAULT_SEARCH_BUDGET = 10_000
_DIGEST_VERSION = "pydot-canonical-graph-v1"

AttributeNames = Union[str, Iterable[str]]
_UNDECLARED = object()


@dataclass(frozen=True)
class CanonicalizationResult:
    """Result of a bounded canonicalization attempt.

    ``status`` is ``"complete"`` only when the complete symmetry search
    fit in the configured budget. An incomplete result never contains a
    digest, so callers cannot accidentally cache an unverified form.
    """

    status: str
    representation: Optional[Dict[str, object]]
    digest: Optional[str]
    budget: int
    searched_candidates: int

    @property
    def complete(self) -> bool:
        """Return whether the canonicalization was fully verified."""
        return self.status == "complete"


@dataclass(frozen=True)
class _GraphData:
    directed: bool
    node_ids: List[Any]
    node_attributes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, Dict[str, Any]]]


def canonicalize_graph(
    graph: Any,
    node_attributes: AttributeNames = (),
    edge_attributes: AttributeNames = (),
    search_budget: int = DEFAULT_SEARCH_BUDGET,
) -> CanonicalizationResult:
    """Return a canonical representation and SHA-256 digest.

    Node identifiers and declaration order are not semantic. Only the
    named node and edge attributes participate in comparison. Parallel
    edges and self-loops are retained.

    The graph must already be expanded to nodes and directed edges.
    Subgraph endpoints are rejected because they represent an unexpanded
    structure.
    """

    node_names = _attribute_names(node_attributes, "node_attributes")
    edge_names = _attribute_names(edge_attributes, "edge_attributes")

    if not isinstance(search_budget, int) or isinstance(
        search_budget, bool
    ):
        raise TypeError("search_budget must be an integer")
    if search_budget < 0:
        raise ValueError("search_budget must be non-negative")

    data = _as_graph_data(graph, node_names)
    return _canonical_data(data, node_names, edge_names, search_budget)


def _canonical_data(
    data: _GraphData,
    node_names: Sequence[str],
    edge_names: Sequence[str],
    search_budget: int,
) -> CanonicalizationResult:
    node_attrs = [
        _selected_attributes(attrs, node_names)
        for attrs in data.node_attributes
    ]
    edges = [
        (source, target, _selected_attributes(attrs, edge_names))
        for source, target, attrs in data.edges
    ]

    colors = _refine_colors(node_attrs, edges, data.directed)
    blocks = _color_blocks(colors)

    total_candidates = 1
    for block in blocks:
        for factor in range(2, len(block) + 1):
            total_candidates *= factor
            if total_candidates > max(search_budget, 1):
                return CanonicalizationResult(
                    "incomplete", None, None, search_budget, 0
                )

    node_count = len(data.node_ids)

    position_ranges: List[List[int]] = []
    local_indices: Dict[int, int] = {}
    position = 0
    for block_index, block in enumerate(blocks):
        positions = list(range(position, position + len(block)))
        position_ranges.append(positions)
        position += len(block)
        for local_index, vertex in enumerate(block):
            local_indices[vertex] = local_index

    best_json: Optional[str] = None
    searched = 0
    block_permutations = (
        itertools.permutations(positions)
        for positions in position_ranges
    )

    for choices in itertools.product(*block_permutations):
        mapping = [0] * node_count
        for block_index, block in enumerate(blocks):
            for vertex in block:
                mapping[vertex] = choices[block_index][
                    local_indices[vertex]
                ]

        canonical_nodes: List[Dict[str, object]] = [
            {} for _ in range(node_count)
        ]
        for vertex, canonical_id in enumerate(mapping):
            canonical_nodes[canonical_id] = {
                "id": canonical_id,
                "attributes": node_attrs[vertex],
            }

        canonical_edges = []
        for source, target, attributes in edges:
            canonical_source = mapping[source]
            canonical_target = mapping[target]
            if not data.directed:
                canonical_source, canonical_target = sorted(
                    (canonical_source, canonical_target)
                )
            canonical_edges.append(
                {
                    "source": canonical_source,
                    "target": canonical_target,
                    "attributes": attributes,
                }
            )
        canonical_edges.sort(
            key=lambda edge: json_dumps(
                edge, sort_keys=True, separators=(",", ":")
            )
        )

        representation = {
            "directed": data.directed,
            "nodes": canonical_nodes,
            "edges": canonical_edges,
        }
        candidate_json = json_dumps(
            representation, sort_keys=True, separators=(",", ":")
        )
        if best_json is None or candidate_json < best_json:
            best_json = candidate_json
        searched += 1

    assert best_json is not None
    representation = json.loads(best_json)
    digest_payload = {
        "semantics": {
            "node_attributes": list(node_names),
            "edge_attributes": list(edge_names),
        },
        "representation": representation,
    }
    digest_input = (
        _DIGEST_VERSION
        + "\0"
        + json_dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    return CanonicalizationResult(
        "complete",
        representation,
        digest,
        search_budget,
        searched,
    )


def _refine_colors(
    node_attributes: Sequence[Dict[str, Any]],
    edges: Sequence[Tuple[int, int, Dict[str, Any]]],
    directed: bool,
) -> List[int]:
    outgoing: List[List[Tuple[int, str]]] = [
        [] for _ in node_attributes
    ]
    incoming: List[List[Tuple[int, str]]] = [
        [] for _ in node_attributes
    ]
    edge_codes = [_attribute_code(attrs) for _, _, attrs in edges]

    for edge_index, (source, target, _) in enumerate(edges):
        edge_code = edge_codes[edge_index]
        if directed:
            outgoing[source].append((target, edge_code))
            incoming[target].append((source, edge_code))
        elif source == target:
            outgoing[source].append((target, edge_code))
            incoming[source].append((source, edge_code))
        else:
            neighbors = (
                (source, target),
                (target, source),
            )
            for vertex, neighbor in neighbors:
                outgoing[vertex].append((neighbor, edge_code))
                incoming[vertex].append((neighbor, edge_code))

    signatures = [
        json_dumps(_stable_token(attributes), sort_keys=True)
        for attributes in node_attributes
    ]
    colors = _assign_colors(signatures)

    for _ in range(len(node_attributes) + 1):
        next_signatures = []
        for vertex, signature in enumerate(signatures):
            in_signature = sorted(
                (colors[neighbor], edge_code)
                for neighbor, edge_code in incoming[vertex]
            )
            out_signature = sorted(
                (colors[neighbor], edge_code)
                for neighbor, edge_code in outgoing[vertex]
            )
            next_signatures.append(
                signature
                + "\0"
                + json_dumps(
                    {
                        "in": in_signature,
                        "out": out_signature,
                    },
                    sort_keys=True,
                )
            )

        next_colors = _assign_colors(next_signatures)
        if next_colors == colors:
            return next_colors
        colors = next_colors
        signatures = next_signatures

    return colors  # pragma: no cover


def _assign_colors(signatures: Sequence[str]) -> List[int]:
    identifiers = {
        signature: index
        for index, signature in enumerate(sorted(set(signatures)))
    }
    return [identifiers[signature] for signature in signatures]


def _color_blocks(colors: Sequence[int]) -> List[List[int]]:
    grouped: Dict[int, List[int]] = {}
    for vertex, color in enumerate(colors):
        grouped.setdefault(color, []).append(vertex)
    return [grouped[color] for color in sorted(grouped)]


def _as_graph_data(
    graph: Any,
    node_names: Sequence[str],
) -> _GraphData:
    if isinstance(graph, pydot.Graph):
        return _pydot_graph_data(graph, node_names)
    if isinstance(graph, Mapping):
        return _mapping_graph_data(graph, node_names)
    raise TypeError("graph must be a pydot.Graph or a mapping")


def _pydot_graph_data(
    graph: pydot.Graph, node_names: Sequence[str]
) -> _GraphData:
    if graph.get_subgraph_list():
        raise ValueError("subgraphs must be expanded before canonicalization")

    node_index: Dict[Any, int] = {}
    node_ids: List[Any] = []
    node_attributes: List[Dict[str, Any]] = []

    def vertex(endpoint: Any) -> int:
        if isinstance(endpoint, Mapping):
            raise ValueError(
                "subgraph edge endpoints must be expanded "
                "before canonicalization"
            )
        if not isinstance(endpoint, (str, int, float)) or isinstance(
            endpoint, bool
        ):
            raise TypeError("edge endpoints must identify expanded nodes")
        node_id = _pydot_endpoint_id(endpoint)
        if node_id not in node_index:
            node_index[node_id] = len(node_ids)
            node_ids.append(node_id)
            node_attributes.append({name: _UNDECLARED for name in node_names})
        return node_index[node_id]

    edges: List[Tuple[int, int, Dict[str, Any]]] = []
    for edge in graph.get_edge_list():
        source = vertex(edge.get_source())
        target = vertex(edge.get_destination())
        edges.append((source, target, dict(edge.get_attributes())))

    default_nodes = {"graph", "node", "edge"}
    for node in graph.get_node_list():
        node_id = _pydot_endpoint_id(node.get_name())
        if node_id in default_nodes and node_id not in node_index:
            continue
        if node_id not in node_index:
            node_index[node_id] = len(node_ids)
            node_ids.append(node_id)
            selected = {name: _UNDECLARED for name in node_names}
            _merge_selected_attributes(
                selected, node.get_attributes(), node_names, node_id
            )
            node_attributes.append(selected)
        else:
            _merge_selected_attributes(
                node_attributes[node_index[node_id]],
                node.get_attributes(),
                node_names,
                node_id,
            )

    directed = graph.get_graph_type() != "graph"
    return _GraphData(directed, node_ids, node_attributes, edges)


def _pydot_endpoint_id(endpoint: Any) -> str:
    value = str(endpoint)
    if (
        not value.startswith('"')
        and ":" in value
        and not value.endswith('"')
    ):
        value = value.split(":", 1)[0]
    return value


def _mapping_graph_data(
    graph: Mapping[Any, Any], node_names: Sequence[str]
) -> _GraphData:
    node_index: Dict[Any, int] = {}
    node_ids: List[Any] = []
    node_attributes: List[Dict[str, Any]] = []

    def vertex(endpoint: Any) -> int:
        _ensure_node_identifier(endpoint)
        if endpoint not in node_index:
            node_index[endpoint] = len(node_ids)
            node_ids.append(endpoint)
            node_attributes.append(
                {name: _UNDECLARED for name in node_names}
            )
        return node_index[endpoint]

    raw_nodes = graph.get("nodes", ())
    if not isinstance(raw_nodes, Iterable):
        raise TypeError("graph['nodes'] must be iterable")
    for item in raw_nodes:
        if isinstance(item, Mapping):
            if "id" not in item:
                raise ValueError("node mappings require an 'id'")
            node_id = item["id"]
            attributes = _mapping_item_attributes(
                item, ("id", "attributes", "attrs")
            )
        elif isinstance(item, tuple) and len(item) == 2:
            node_id, attributes = item
        else:
            node_id, attributes = item, {}
        _ensure_node_identifier(node_id)
        if not isinstance(attributes, Mapping):
            raise TypeError("node attributes must be a mapping")
        if node_id not in node_index:
            node_index[node_id] = len(node_ids)
            node_ids.append(node_id)
            selected = {name: _UNDECLARED for name in node_names}
            _merge_selected_attributes(
                selected, attributes, node_names, node_id
            )
            node_attributes.append(selected)
        else:
            _merge_selected_attributes(
                node_attributes[node_index[node_id]],
                attributes,
                node_names,
                node_id,
            )

    raw_edges = graph.get("edges", ())
    if not isinstance(raw_edges, Iterable):
        raise TypeError("graph['edges'] must be iterable")

    edges: List[Tuple[int, int, Dict[str, Any]]] = []
    for item in raw_edges:
        if isinstance(item, Mapping):
            if "source" not in item or "target" not in item:
                raise ValueError("edge mappings require source and target")
            source = vertex(item["source"])
            target = vertex(item["target"])
            attributes = _mapping_item_attributes(
                item,
                ("source", "target", "attributes", "attrs"),
            )
        elif isinstance(item, tuple) and len(item) == 3:
            source_id, target_id, attributes = item
            source = vertex(source_id)
            target = vertex(target_id)
        else:
            raise ValueError("edges must be mappings or 3-tuples")
        if not isinstance(attributes, Mapping):
            raise TypeError("edge attributes must be a mapping")
        edges.append((source, target, dict(attributes)))

    directed = bool(graph.get("directed", True))
    return _GraphData(directed, node_ids, node_attributes, edges)


def _ensure_node_identifier(value: Any) -> None:
    if isinstance(value, (Mapping, list, bytearray)):
        raise TypeError("node identifiers must be hashable values")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("node identifiers must be hashable values") from exc


def _attribute_names(value: AttributeNames, argument: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, Iterable):
        names = tuple(value)
    else:
        raise TypeError(
            f"{argument} must be a string or iterable of strings"
        )
    if any(not isinstance(name, str) for name in names):
        raise TypeError(f"{argument} must contain only strings")
    return tuple(sorted(set(names)))


def _mapping_item_attributes(
    item: Mapping[Any, Any], reserved: Tuple[str, ...]
) -> Dict[str, Any]:
    if "attributes" in item:
        attributes = item["attributes"]
    elif "attrs" in item:
        attributes = item["attrs"]
    else:
        return {
            key: value
            for key, value in item.items()
            if key not in reserved
        }
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    return dict(attributes)


def _merge_selected_attributes(
    target: Dict[str, Any],
    source: Mapping[Any, Any],
    names: Sequence[str],
    node_id: Any,
) -> None:
    for name in names:
        if name not in source or source[name] is _UNDECLARED:
            continue
        value = _stable_token(source[name])
        if (
            target.get(name, _UNDECLARED) is not _UNDECLARED
            and _stable_token(target[name]) != value
        ):
            raise ValueError(
                f"conflicting declarations for node {node_id!r}.{name}"
            )
        target[name] = source[name]


def _selected_attributes(
    attributes: Mapping[Any, Any], names: Sequence[str]
) -> Dict[str, Any]:
    return {
        name: _stable_token(attributes[name])
        for name in names
        if name in attributes and attributes[name] is not _UNDECLARED
    }


def _attribute_code(attributes: Mapping[str, Any]) -> str:
    return json_dumps(_stable_token(attributes), sort_keys=True)


def _stable_token(value: Any) -> Any:
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {
                key: _stable_token(item_value)
                for key, item_value in sorted(value.items())
            }
        return [
            [_stable_token(key), _stable_token(item_value)]
            for key, item_value in sorted(
                value.items(),
                key=lambda item: _json_key(_stable_token(item[0])),
            )
        ]
    if isinstance(value, (list, tuple)):
        return [_stable_token(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_stable_token(item) for item in value),
            key=_json_key,
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite floats are not supported")
        return value
    raise TypeError(f"canonicalization does not support {type(value)!r}")


def _json_key(value: Any) -> str:
    return json_dumps(value)


def json_dumps(value: Any, **kwargs: Any) -> str:
    """JSON serialization that retains non-ASCII text."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        **kwargs,
    )
