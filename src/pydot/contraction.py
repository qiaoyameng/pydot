# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Explicit node contraction for pydot graphs.

Node contraction folds groups of nodes (for example, the instances of
one service in a dependency graph) into single summary nodes, while
keeping a full provenance record of how every original edge was
routed.

The main entry point is :func:`contract_nodes`. It takes a graph and
a set of non-overlapping node groups, and returns a
:class:`ContractionResult` holding the contracted graph together with
the source mapping: every original edge is accounted for either in
the ``origins`` of exactly one contracted edge or in the explicit
``dropped`` records.

Policies are explicit and chosen by the caller:

- ``self_loops``: whether edges that become self-loops after
  contraction are kept (``"keep"``, the default) or deleted
  (``"drop"``). Deleted self-loops are reported in
  :attr:`ContractionResult.dropped`.
- ``merge_parallel``: whether parallel edges between the same pair of
  contracted nodes are merged into one (``True``) or kept as
  individual edges (``False``, the default).
- ``attribute_reducer``: conflicts between attribute values (of
  contracted member nodes or of merged parallel edges) are never
  resolved by silently picking one value. If the values differ and no
  reducer is provided, :class:`~pydot.exceptions.AttributeConflictError`
  is raised. The reducer is called as ``reducer(name, values)`` with
  the attribute name and the conflicting values in encounter order.
- ``port_map``: edges whose endpoints carry a port (``node:port``)
  cannot have their port name blindly attached to the contracted
  node. Each ``(node, port)`` combination used by an edge touching a
  contracted group must have an entry in ``port_map`` mapping it to
  the port on the contracted node (or to ``None``/``""`` to drop the
  port), otherwise :class:`~pydot.exceptions.PortMappingError` is
  raised and no contraction happens.

Nested cluster membership is validated before anything is built: if
the members of one group are declared in different clusters (or a
member is declared in more than one place),
:class:`~pydot.exceptions.ClusterConflictError` is raised with
per-member cluster paths so the conflict can be located. When all
members share one cluster, the contracted node is placed in that same
(possibly nested) cluster.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping

from pydot.classes import AttributeDict, EdgeEndpoint, FrozenDict
from pydot.core import Edge, Graph, Node
from pydot.exceptions import (
    AttributeConflictError,
    ClusterConflictError,
    ContractionError,
    OverlappingGroupsError,
    PortMappingError,
    UnknownNodeError,
)

__all__ = [
    "contract_nodes",
    "ContractionResult",
    "ContractedEdge",
    "DroppedEdge",
    "EdgeOrigin",
]

SELF_LOOP_POLICIES = ("keep", "drop")

AttributeReducer = Callable[[str, list[Any]], Any]


@dataclass
class EdgeOrigin:
    """Provenance record for one original edge.

    ``index`` is a global sequence number assigned to every edge of
    the original graph in encounter order (container by container,
    top-level graph first). ``location`` is the path of subgraph
    names (outermost first) of the container the edge was declared
    in; an empty tuple means the top-level graph.
    """

    index: int
    source: str
    destination: str
    attributes: dict[str, Any]
    location: tuple[str, ...]


@dataclass
class ContractedEdge:
    """One edge of the contracted graph and where it came from."""

    source: str
    destination: str
    attributes: dict[str, Any]
    origins: list[EdgeOrigin] = field(default_factory=list)


@dataclass
class DroppedEdge:
    """An original edge that was deliberately deleted.

    ``reason`` is currently always ``"self_loop"``: the edge became a
    self-loop after contraction and the policy was ``"drop"``.
    """

    origin: EdgeOrigin
    reason: str


@dataclass
class ContractionResult:
    """The outcome of :func:`contract_nodes`.

    ``graph`` is the new, contracted graph (the input graph is not
    modified). ``node_map`` maps every original node name to its name
    in the contracted graph. ``groups`` maps each contracted node
    name to its member node names. ``edges`` and ``dropped`` together
    account for every original edge exactly once.
    """

    graph: Graph
    node_map: dict[str, str]
    groups: dict[str, tuple[str, ...]]
    edges: list[ContractedEdge] = field(default_factory=list)
    dropped: list[DroppedEdge] = field(default_factory=list)

    def all_origins(self) -> list[EdgeOrigin]:
        """Return every original edge, kept or dropped, exactly once."""
        origins: list[EdgeOrigin] = []
        for edge in self.edges:
            origins.extend(edge.origins)
        origins.extend(dropped.origin for dropped in self.dropped)
        return sorted(origins, key=lambda origin: origin.index)


def _split_endpoint(endpoint: EdgeEndpoint) -> tuple[Any, str | None]:
    """Split an edge endpoint into (node name, port).

    The port, if present, is returned without the leading colon.
    Non-string endpoints are returned as-is with no port.
    """
    if not isinstance(endpoint, str) or not endpoint:
        return endpoint, None
    if endpoint.startswith('"'):
        end = endpoint.find('"', 1)
        if end > 0:
            name = endpoint[1:end]
            rest = endpoint[end + 1 :]
            if rest.startswith(":") and len(rest) > 1:
                return name, rest[1:]
            return name, None
    idx = endpoint.find(":")
    if idx > 0 and idx + 1 < len(endpoint):
        return endpoint[:idx], endpoint[idx + 1 :]
    return endpoint, None


def _walk_containers(
    obj_dict: AttributeDict, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], AttributeDict]]:
    """Yield (path, obj_dict) for a graph and all nested subgraphs."""
    yield path, obj_dict
    for name in obj_dict["subgraphs"]:
        for sub_obj_dict in obj_dict["subgraphs"][name]:
            yield from _walk_containers(sub_obj_dict, path + (name,))


def _reduce_attributes(
    attr_dicts: list[dict[str, Any]],
    context: str,
    reducer: AttributeReducer | None,
) -> dict[str, Any]:
    """Combine attribute dicts, routing conflicts through ``reducer``."""
    values_by_key: dict[str, list[Any]] = {}
    for attrs in attr_dicts:
        for key, value in attrs.items():
            values_by_key.setdefault(key, []).append(value)
    reduced: dict[str, Any] = {}
    for key, values in values_by_key.items():
        first = values[0]
        if all(bool(value == first) for value in values[1:]):
            reduced[key] = first
        elif reducer is not None:
            reduced[key] = reducer(key, list(values))
        else:
            raise AttributeConflictError(key, list(values), context)
    return reduced


def _validate_groups(
    groups: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Normalize and validate the contraction groups."""
    if not groups:
        raise ContractionError(
            "contract_nodes() requires at least one contraction group."
        )
    normalized: dict[str, tuple[str, ...]] = {}
    for name, members in groups.items():
        if not isinstance(name, str) or not name:
            raise ContractionError(
                f"Invalid contracted node name {name!r}; "
                "names must be non-empty strings."
            )
        member_list = list(members)
        if not member_list or not all(
            isinstance(member, str) and member for member in member_list
        ):
            raise ContractionError(
                f"Invalid members {member_list!r} for group {name!r}; "
                "members must be non-empty node name strings."
            )
        normalized[name] = tuple(dict.fromkeys(member_list))
    overlapping: dict[str, list[str]] = {}
    for name, members in normalized.items():
        for member in members:
            for other_name, other_members in normalized.items():
                if other_name != name and member in other_members:
                    overlapping.setdefault(member, [])
                    for group_name in (name, other_name):
                        if group_name not in overlapping[member]:
                            overlapping[member].append(group_name)
    if overlapping:
        raise OverlappingGroupsError(
            {node: sorted(names) for node, names in overlapping.items()}
        )
    return normalized


def contract_nodes(
    graph: Graph,
    groups: Mapping[str, Iterable[str]],
    *,
    self_loops: str = "keep",
    merge_parallel: bool = False,
    attribute_reducer: AttributeReducer | None = None,
    port_map: Mapping[tuple[str, str], str | None] | None = None,
) -> ContractionResult:
    """Contract non-overlapping node groups into single nodes.

    ``groups`` maps each new (contracted) node name to the node names
    it absorbs. Groups must not overlap. The input graph is left
    untouched; a :class:`ContractionResult` with the new graph and the
    full source mapping is returned.

    See the module docstring for the exact ``self_loops``,
    ``merge_parallel``, ``attribute_reducer`` and ``port_map``
    policies.
    """
    if self_loops not in SELF_LOOP_POLICIES:
        raise ValueError(
            f"Invalid self_loops policy {self_loops!r}; "
            f"expected one of {SELF_LOOP_POLICIES}."
        )
    normalized = _validate_groups(groups)
    node_to_group = {
        member: name
        for name, members in normalized.items()
        for member in members
    }

    # Pre-pass over the original graph: collect known node names,
    # cluster memberships and member attribute dicts.
    known_nodes: set[str] = set()
    membership: dict[str, set[tuple[str, ...]]] = {}
    member_attr_dicts: dict[str, list[dict[str, Any]]] = {
        name: [] for name in normalized
    }
    for path, obj_dict in _walk_containers(graph.obj_dict, ()):
        for node_name, node_list in obj_dict["nodes"].items():
            known_nodes.add(node_name)
            if path:
                membership.setdefault(node_name, set()).add(path)
            group = node_to_group.get(node_name)
            if group is not None:
                for node_obj_dict in node_list:
                    member_attr_dicts[group].append(
                        dict(node_obj_dict["attributes"])
                    )
        for points, edge_list in obj_dict["edges"].items():
            for endpoint in points:
                name, _port = _split_endpoint(endpoint)
                if isinstance(name, str):
                    known_nodes.add(name)

    unknown = sorted(
        member for member in node_to_group if member not in known_nodes
    )
    if unknown:
        raise UnknownNodeError(unknown)

    for name in normalized:
        if name in known_nodes and name not in normalized[name]:
            raise ContractionError(
                f"Contracted node name {name!r} collides with an "
                "existing node that is not a member of its own group."
            )

    # Cluster membership: every member of a group must be declared in
    # exactly one identical place, otherwise the contracted node could
    # not be placed unambiguously.
    placement: dict[str, tuple[str, ...]] = {}
    for name, members in normalized.items():
        memberships: dict[str, list[tuple[str, ...]]] = {}
        paths: set[tuple[str, ...]] = set()
        for member in members:
            member_paths = membership.get(member) or {()}
            memberships[member] = sorted(member_paths)
            paths.update(member_paths)
        if len(paths) != 1 or any(len(p) != 1 for p in memberships.values()):
            raise ClusterConflictError(name, memberships)
        placement[name] = next(iter(paths))

    directed = graph.get_type() != "graph"

    def remap_endpoint(
        endpoint: EdgeEndpoint, edge: tuple[EdgeEndpoint, EdgeEndpoint]
    ) -> EdgeEndpoint:
        if isinstance(endpoint, FrozenDict):
            raise ContractionError(
                "Cannot contract edges with subgraph endpoints; "
                "expand the subgraph endpoint into plain node "
                "endpoints first."
            )
        name, port = _split_endpoint(endpoint)
        if not isinstance(name, str):
            return endpoint
        new_name = node_to_group.get(name)
        if new_name is None:
            return endpoint
        if port is None:
            return new_name
        mapped: str | None = None
        if port_map is not None and (name, port) in port_map:
            mapped = port_map[(name, port)]
        else:
            raise PortMappingError(name, port, (str(edge[0]), str(edge[1])))
        if mapped:
            return f"{new_name}:{mapped}"
        return new_name

    # Transform pass on a deep copy.
    new_graph = copy.deepcopy(graph)
    result = ContractionResult(
        graph=new_graph,
        node_map={
            node: node_to_group.get(node, node) for node in known_nodes
        },
        groups=normalized,
    )

    edge_index = 0

    for path, obj_dict in _walk_containers(new_graph.obj_dict, ()):
        container = Graph(obj_dict=obj_dict)

        # Remove absorbed nodes, place contracted nodes.
        for node_name in list(obj_dict["nodes"]):
            if node_name in node_to_group:
                del obj_dict["nodes"][node_name]
        for group_name, group_path in placement.items():
            if group_path == path:
                attrs = _reduce_attributes(
                    member_attr_dicts[group_name],
                    f"node {group_name!r}",
                    attribute_reducer,
                )
                container.add_node(Node(group_name, **attrs))

        # Transform edges declared in this container.
        survivors: list[ContractedEdge] = []
        merge_index: dict[tuple[Any, ...], int] = {}
        for points, edge_list in list(obj_dict["edges"].items()):
            for edge_obj_dict in edge_list:
                src, dst = edge_obj_dict["points"]
                origin = EdgeOrigin(
                    index=edge_index,
                    source=str(src),
                    destination=str(dst),
                    attributes=dict(edge_obj_dict["attributes"]),
                    location=path,
                )
                edge_index += 1
                new_src = remap_endpoint(src, (src, dst))
                new_dst = remap_endpoint(dst, (src, dst))
                if new_src == new_dst and self_loops == "drop":
                    result.dropped.append(
                        DroppedEdge(origin=origin, reason="self_loop")
                    )
                    continue
                key: tuple[Any, ...]
                if directed:
                    key = (new_src, new_dst)
                else:
                    key = tuple(sorted((str(new_src), str(new_dst))))
                if merge_parallel and key in merge_index:
                    survivors[merge_index[key]].origins.append(origin)
                    continue
                if merge_parallel:
                    merge_index[key] = len(survivors)
                survivors.append(
                    ContractedEdge(
                        source=str(new_src),
                        destination=str(new_dst),
                        attributes={},
                        origins=[origin],
                    )
                )

        obj_dict["edges"] = {}
        for survivor in survivors:
            survivor.attributes = _reduce_attributes(
                [origin.attributes for origin in survivor.origins],
                f"edges {survivor.source!r} -> {survivor.destination!r}",
                attribute_reducer,
            )
            container.add_edge(
                Edge(
                    survivor.source,
                    survivor.destination,
                    **survivor.attributes,
                )
            )
            result.edges.append(survivor)

    return result
