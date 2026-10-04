from __future__ import annotations

import ast
import copy
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydot.classes import FrozenDict
from pydot.core import Edge, Graph, Node, quote_id_if_necessary
from pydot.exceptions import ContractionConflict, ContractionError

__all__ = [
    "ContractionResult",
    "EdgeOrigin",
    "DeletedEdge",
    "PortMappingContext",
    "contract_nodes",
]

AttributeReducer = Callable[..., Any]
_SENTINEL = object()
_RESERVED_NAMES = {"graph", "node", "edge"}
_QUOTED_ID = re.compile(r'^"(?:\\.|[^"\\])*"')
_HTML_ID = re.compile(r"^<(?:\\.|[^>\\])*>")


@dataclass(frozen=True)
class EdgeOrigin:
    new_edge_id: int
    original_edge_ids: tuple[int, ...]
    path: tuple[str, ...]
    source: Any
    destination: Any
    sources: tuple[Any, ...]
    destinations: tuple[Any, ...]


@dataclass(frozen=True)
class DeletedEdge:
    original_edge_id: int
    path: tuple[str, ...]
    source: Any
    destination: Any
    reason: str


@dataclass(frozen=True)
class ContractionResult:
    graph: Graph
    node_sources: Mapping[str, tuple[str, ...]]
    edge_origins: tuple[EdgeOrigin, ...]
    deleted_edges: tuple[DeletedEdge, ...]

    @property
    def edge_sources(self) -> Mapping[int, tuple[int, ...]]:
        return {
            origin.new_edge_id: origin.original_edge_ids
            for origin in self.edge_origins
        }

    @property
    def source_mapping(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            "nodes": self.node_sources,
            "edges": self.edge_sources,
        }


@dataclass(frozen=True)
class PortMappingContext:
    group: str
    member: str
    port: str
    edge_id: int
    role: str


@dataclass(frozen=True)
class _GroupSpec:
    name: str
    members: tuple[str, ...]
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class _NodeOccurrence:
    name: str
    path: tuple[str, ...]
    container_index: int
    sequence: int
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class _Container:
    graph: Graph
    path: tuple[str, ...]
    index: int
    children: tuple[int, ...]


@dataclass(frozen=True)
class _OriginalEdge:
    edge_id: int
    path: tuple[str, ...]
    container_index: int
    sequence: int
    source: Any
    destination: Any
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class _ReductionRequest:
    kind: str
    name: str
    values: tuple[Any, ...]
    reducer: AttributeReducer
    group: str | None = None
    edge_ids: tuple[int, ...] = ()
    container_index: int | None = None


@dataclass(frozen=True)
class _ConflictRequest:
    kind: str
    name: str
    values: tuple[Any, ...]
    group: str | None = None
    edge_ids: tuple[int, ...] = ()
    container_index: int | None = None


@dataclass
class _EdgePlan:
    edge_id: int
    path: tuple[str, ...]
    container_index: int
    sequence: int
    source: Any
    destination: Any
    attributes: dict[str, Any]
    source_edge_ids: tuple[int, ...]
    original_sources: tuple[Any, ...]
    original_destinations: tuple[Any, ...]
    effective_source_port: str | None
    effective_destination_port: str | None
    internal_self_loop: bool = False


@dataclass
class _PlanGroup:
    container_index: int
    key: tuple[Any, ...]
    plans: list[_EdgePlan]
    attributes: dict[str, Any]

    @property
    def first(self) -> _EdgePlan:
        return self.plans[0]


def _canonical_id(value: Any) -> str:
    if isinstance(value, Node):
        return quote_id_if_necessary(value.get_name())
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("Contraction group names and members must be nodes.")
    return quote_id_if_necessary(str(value))


def _logical_id(value: str) -> str:
    quoted = _QUOTED_ID.match(value)
    if quoted:
        text = quoted.group(0)
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text[1:-1]
        return parsed if isinstance(parsed, str) else text
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    return value


def _split_endpoint(endpoint: Any) -> tuple[Any, str | None]:
    if isinstance(endpoint, FrozenDict):
        if endpoint.get("type") == "node":
            return endpoint.get("name"), None
        return endpoint, None
    if isinstance(endpoint, bool) or not isinstance(
        endpoint, (str, int, float)
    ):
        return endpoint, None
    if isinstance(endpoint, (int, float)):
        return str(endpoint), None

    prefix_len: int | None = None
    if endpoint.startswith('"'):
        match = _QUOTED_ID.match(endpoint)
        prefix_len = match.end() if match else None
    elif endpoint.startswith("<"):
        match = _HTML_ID.match(endpoint)
        prefix_len = match.end() if match else None
    else:
        position = endpoint.find(":")
        prefix_len = None if position < 0 else position

    if prefix_len is None or prefix_len >= len(endpoint):
        return endpoint, None
    if endpoint[prefix_len] != ":":
        return endpoint, None
    return endpoint[:prefix_len], endpoint[prefix_len + 1 :]


def _normalise_groups(groups: Any) -> tuple[_GroupSpec, ...]:
    if not isinstance(groups, Mapping):
        raise TypeError("groups must map contracted node names to members.")

    result: list[_GroupSpec] = []
    seen_names: set[str] = set()
    seen_members: dict[str, str] = {}

    for raw_name, raw_value in groups.items():
        name = _canonical_id(raw_name)
        if name in seen_names:
            raise ValueError(f"Duplicate contracted node name {name}.")
        seen_names.add(name)
        if name in _RESERVED_NAMES:
            raise ValueError(f"Cannot use reserved node name {name}.")

        attributes: Mapping[str, Any] = {}
        if isinstance(raw_value, Mapping):
            raw_members = raw_value.get("members")
            attributes = raw_value.get("attributes", {})
            if not isinstance(attributes, Mapping):
                raise TypeError(f"Attributes for {name} must be a mapping.")
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value, str
        ):
            raw_members = raw_value
        else:
            raise TypeError(f"Members for {name} must be a sequence.")

        if not isinstance(raw_members, Sequence) or isinstance(
            raw_members, (str, bytes)
        ):
            raise TypeError(f"Members for {name} must be a sequence.")
        if not raw_members:
            raise ValueError(f"Contraction group {name} is empty.")

        members: list[str] = []
        for raw_member in raw_members:
            member = _canonical_id(raw_member)
            if member in seen_members:
                raise ValueError(
                    f"Node {member} is in both {seen_members[member]} "
                    f"and {name}."
                )
            if member not in members:
                seen_members[member] = name
                members.append(member)

        result.append(
            _GroupSpec(
                name=name,
                members=tuple(members),
                attributes=attributes,
            )
        )

    return tuple(result)


def _traverse(
    graph: Graph
) -> tuple[
    tuple[_Container, ...],
    tuple[_NodeOccurrence, ...],
    tuple[_OriginalEdge, ...],
]:
    containers: list[_Container] = []
    nodes: list[_NodeOccurrence] = []
    edges: list[_OriginalEdge] = []

    def visit(current: Graph, path: tuple[str, ...], index: int) -> None:
        child_indices: list[int] = []

        def collect() -> tuple[int, ...]:
            for node in current.get_node_list():
                name = node.get_name()
                if name not in _RESERVED_NAMES:
                    nodes.append(
                        _NodeOccurrence(
                            name=name,
                            path=path,
                            container_index=index,
                            sequence=node.get_sequence() or 0,
                            attributes=node.get_attributes(),
                        )
                    )

            for edge in current.get_edge_list():
                edge_id = len(edges)
                edges.append(
                    _OriginalEdge(
                        edge_id=edge_id,
                        path=path,
                        container_index=index,
                        sequence=edge.get_sequence() or 0,
                        source=edge.get_source(),
                        destination=edge.get_destination(),
                        attributes=edge.get_attributes(),
                    )
                )

            for subgraph in current.get_subgraph_list():
                child_index = len(containers)
                child_indices.append(child_index)
                visit(subgraph, path + (subgraph.get_name(),), child_index)

            return tuple(child_indices)

        containers.append(_Container(current, path, index, ()))
        children = collect()
        containers[index] = _Container(current, path, index, children)

    visit(graph, (), 0)
    return tuple(containers), tuple(nodes), tuple(edges)


def _identity_maps(
    nodes: Sequence[_NodeOccurrence], edges: Sequence[_OriginalEdge]
) -> tuple[set[str], dict[str, str]]:
    raw: set[str] = set()
    logical_to_raw: dict[str, str] = {}

    def add(value: Any) -> None:
        base, _ = _split_endpoint(value)
        if isinstance(base, str) and not isinstance(base, FrozenDict):
            raw.add(base)
            logical_to_raw.setdefault(_logical_id(base), base)

    for occurrence in nodes:
        add(occurrence.name)
    for edge in edges:
        add(edge.source)
        add(edge.destination)
    return raw, logical_to_raw


def _resolve_identity(
    value: str, raw_ids: set[str], logical_to_raw: dict[str, str]
) -> str | None:
    if value in raw_ids:
        return value
    return logical_to_raw.get(_logical_id(value))


def _validate_and_locate(
    groups: Sequence[_GroupSpec],
    nodes: Sequence[_NodeOccurrence],
    edges: Sequence[_OriginalEdge],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, int],
    dict[str, set[int]],
    dict[str, _NodeOccurrence],
]:
    raw_ids, logical_to_raw = _identity_maps(nodes, edges)
    member_group: dict[str, str] = {}
    logical_member_group: dict[str, str] = {}

    for group in groups:
        existing = _resolve_identity(group.name, raw_ids, logical_to_raw)
        if existing is not None and existing not in group.members:
            raise ValueError(f"Graph already contains node {group.name}.")

        for member in group.members:
            actual = _resolve_identity(member, raw_ids, logical_to_raw)
            if actual is None:
                raise ValueError(f"Unknown node {member} in {group.name}.")
            member_group[actual] = group.name
            logical_member_group[_logical_id(actual)] = group.name

    occurrences_by_name: dict[str, list[_NodeOccurrence]] = defaultdict(list)
    for occurrence in nodes:
        occurrences_by_name[occurrence.name].append(occurrence)

    edge_containers_by_name: dict[str, set[int]] = defaultdict(set)
    for edge in edges:
        for endpoint in (edge.source, edge.destination):
            base, _ = _split_endpoint(endpoint)
            if isinstance(base, str):
                edge_containers_by_name[base].add(edge.container_index)

    targets: dict[str, int] = {}
    group_targets: dict[str, set[int]] = defaultdict(set)
    representative: dict[str, _NodeOccurrence] = {}
    conflicts: list[ContractionConflict] = []
    container_paths: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    container_paths[0].add(())
    for occurrence in nodes:
        container_paths[occurrence.container_index].add(occurrence.path)
    for edge in edges:
        container_paths[edge.container_index].add(edge.path)

    for group in groups:
        for member in group.members:
            actual = _resolve_identity(member, raw_ids, logical_to_raw)
            assert actual is not None
            occurrences = occurrences_by_name.get(actual, [])
            locations: dict[
                tuple[tuple[str, ...], int], list[_NodeOccurrence]
            ] = defaultdict(list)
            for occurrence in occurrences:
                locations[
                    (occurrence.path, occurrence.container_index)
                ].append(occurrence)

            if not occurrences:
                inferred = edge_containers_by_name.get(actual, set())
                if not inferred:
                    logical = _logical_id(actual)
                    for endpoint_name, indices in (
                        edge_containers_by_name.items()
                    ):
                        if _logical_id(endpoint_name) == logical:
                            inferred.update(indices)
                if len(inferred) != 1:
                    inferred_paths = tuple(
                        path
                        for index in inferred
                        for path in container_paths.get(index, ())
                    )
                    conflicts.append(
                        ContractionConflict(
                            kind="cluster_membership_conflict",
                            message=(
                                f"Node {member} appears in nested clusters "
                                + ", ".join(map(str, inferred_paths))
                                + "."
                            ),
                            group=group.name,
                            node=member,
                            container_indices=tuple(inferred),
                            paths=inferred_paths,
                        )
                    )
                    continue
                target_index = next(iter(inferred))
                targets[actual] = target_index
                group_targets[group.name].add(target_index)
                continue

            if len(locations) > 1:
                first_per_path = [
                    path_nodes[0] for path_nodes in locations.values()
                ]
                conflicts.append(
                    ContractionConflict(
                        kind="cluster_membership_conflict",
                        message=(
                            f"Node {member} is assigned to nested clusters "
                            + ", ".join(str(path) for path in paths)
                            + "."
                        ),
                        group=group.name,
                        node=member,
                        container_indices=tuple(
                            item.container_index for item in first_per_path
                        ),
                        paths=tuple(item.path for item in first_per_path),
                    )
                )
                continue

            first = occurrences[0]
            targets[actual] = first.container_index
            group_targets[group.name].add(first.container_index)
            representative[actual] = first

    for group in groups:
        indices = group_targets.get(group.name, set())
        if len(indices) > 1:
            paths = tuple(
                path
                for index in indices
                for path in container_paths.get(index, ())
            )
            conflicts.append(
                ContractionConflict(
                    kind="cluster_group_membership_conflict",
                    message=(
                        f"Members of {group.name} are assigned to different "
                        "nested clusters."
                    ),
                    group=group.name,
                    container_indices=tuple(indices),
                    paths=paths,
                )
            )

    if conflicts:
        raise ContractionError(conflicts)
    return (
        member_group,
        logical_member_group,
        targets,
        group_targets,
        representative,
    )


def _normalise_self_loops(self_loops: str | bool) -> str:
    if self_loops is True:
        return "keep"
    if self_loops is False:
        return "drop"
    if self_loops not in {"keep", "drop", "merge"}:
        raise ValueError("self_loops must be 'keep', 'drop', or 'merge'.")
    return self_loops


def _map_port(
    port_mapping: Any,
    group: str,
    member: str,
    port: str,
    edge_id: int,
    role: str,
    conflicts: list[ContractionConflict],
) -> str | None:
    context = PortMappingContext(
        group=group,
        member=member,
        port=port,
        edge_id=edge_id,
        role=role,
    )

    if callable(port_mapping):
        mapped = port_mapping(context)
    elif isinstance(port_mapping, Mapping):
        candidates = (
            (group, member, port, role),
            (group, member, port),
            (group, port, role),
            (group, port),
            (member, port, role),
            (member, port),
            (port, role),
            port,
        )
        mapped = _SENTINEL
        for candidate in candidates:
            if candidate in port_mapping:
                mapped = port_mapping[candidate]
                break
    else:
        mapped = _SENTINEL

    if mapped is _SENTINEL:
        conflicts.append(
            ContractionConflict(
                kind="missing_port_mapping",
                message=(
                    f"No port mapping for {member}:{port} on edge "
                    f"{edge_id} ({role})."
                ),
                group=group,
                node=member,
                edge_id=edge_id,
                endpoint=f"{member}:{port}",
            )
        )
        return port

    if mapped is None or mapped == "":
        return None
    return str(mapped)


def _attach_port(base: Any, port: str | None) -> Any:
    if port is None or port == "":
        return base
    return f"{base}:{port}"


def _rewrite_endpoint(
    endpoint: Any,
    role: str,
    edge_id: int,
    member_group: Mapping[str, str],
    logical_member_group: Mapping[str, str],
    port_mapping: Any,
    attributes: dict[str, Any],
    conflicts: list[ContractionConflict],
) -> tuple[Any, str | None, str | None]:
    base, explicit_port = _split_endpoint(endpoint)
    if not isinstance(base, str):
        return endpoint, None, None

    group = member_group.get(base) or logical_member_group.get(
        _logical_id(base)
    )
    if group is None:
        return endpoint, explicit_port, attributes.get(
            "tailport" if role == "source" else "headport"
        )

    resolved_explicit = None
    if explicit_port is not None:
        resolved_explicit = _map_port(
            port_mapping,
            group,
            base,
            explicit_port,
            edge_id,
            role,
            conflicts,
        )

    attr_name = "tailport" if role == "source" else "headport"
    attr_port = attributes.get(attr_name)
    resolved_attr = None
    if attr_port is not None:
        resolved_attr = _map_port(
            port_mapping,
            group,
            base,
            str(attr_port),
            edge_id,
            role,
            conflicts,
        )
        if resolved_attr is None:
            attributes.pop(attr_name, None)
        else:
            attributes[attr_name] = resolved_attr

    return (
        _attach_port(group, resolved_explicit),
        resolved_explicit,
        resolved_attr,
    )


def _node_attribute_requests(
    groups: Sequence[_GroupSpec],
    nodes: Sequence[_NodeOccurrence],
    reducer: AttributeReducer | None,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[_ReductionRequest, ...],
    tuple[_ConflictRequest, ...],
]:
    reductions: list[_ReductionRequest] = []
    conflicts: list[_ConflictRequest] = []
    result: dict[str, dict[str, Any]] = {}

    for group in groups:
        values_by_attribute: dict[str, list[Any]] = defaultdict(list)
        for member in group.members:
            for occurrence in nodes:
                if occurrence.name != member:
                    continue
                for key, value in occurrence.attributes.items():
                    values_by_attribute[key].append(value)

        attributes = dict(group.attributes)
        for key, values in values_by_attribute.items():
            if key in attributes:
                continue
            distinct: list[Any] = []
            for value in values:
                if not any(value == existing for existing in distinct):
                    distinct.append(value)
            if len(distinct) > 1:
                if reducer is None:
                    conflicts.append(
                        _ConflictRequest(
                            kind="node",
                            name=key,
                            values=tuple(values),
                            group=group.name,
                        )
                    )
                else:
                    reductions.append(
                        _ReductionRequest(
                            kind="node",
                            name=key,
                            values=tuple(values),
                            reducer=reducer,
                            group=group.name,
                        )
                    )
            else:
                attributes[key] = distinct[0]

        result[group.name] = attributes

    return result, tuple(reductions), tuple(conflicts)


def _edge_attribute_request(
    plans: Sequence[_EdgePlan],
    reducer: AttributeReducer | None,
) -> tuple[
    dict[str, Any],
    tuple[_ReductionRequest, ...],
    tuple[_ConflictRequest, ...],
]:
    values_by_attribute: dict[str, list[Any]] = defaultdict(list)
    for plan in plans:
        for key, value in plan.attributes.items():
            values_by_attribute[key].append(value)

    attributes: dict[str, Any] = {}
    reductions: list[_ReductionRequest] = []
    conflicts: list[_ConflictRequest] = []
    edge_ids = tuple(plan.edge_id for plan in plans)
    container_index = plans[0].container_index

    for key, values in values_by_attribute.items():
        distinct: list[Any] = []
        for value in values:
            if not any(value == existing for existing in distinct):
                distinct.append(value)
        if len(distinct) > 1:
            attributes[key] = distinct[0]
            request_data: dict[str, Any] = {
                "kind": "edge",
                "name": key,
                "values": tuple(values),
                "edge_ids": edge_ids,
                "container_index": container_index,
            }
            if reducer is None:
                conflicts.append(_ConflictRequest(**request_data))
            else:
                reductions.append(
                    _ReductionRequest(reducer=reducer, **request_data)
                )
        else:
            attributes[key] = distinct[0]

    return attributes, tuple(reductions), tuple(conflicts)


def _raise_reduction_conflicts(
    requests: Sequence[_ConflictRequest],
) -> None:
    if requests:
        conflicts = [
            ContractionConflict(
                kind="attribute_conflict",
                message=(
                    f"Conflicting {request.kind} attribute {request.name} "
                    "requires an attribute reducer."
                ),
                group=request.group,
                attribute=request.name,
                values=request.values,
                edge_ids=request.edge_ids,
            )
            for request in requests
        ]
        raise ContractionError(conflicts)


def _build_plans(
    edges: Sequence[_OriginalEdge],
    member_group: Mapping[str, str],
    logical_member_group: Mapping[str, str],
    port_mapping: Any,
    attribute_reducer: AttributeReducer | None,
    merge_parallel_edges: bool,
    self_loops: str,
) -> tuple[tuple[_EdgePlan, ...], tuple[DeletedEdge, ...]]:
    plans: list[_EdgePlan] = []
    deleted: list[DeletedEdge] = []
    port_conflicts: list[ContractionConflict] = []

    for edge in edges:
        attributes = dict(edge.attributes)
        source, source_port, source_attr_port = _rewrite_endpoint(
            edge.source,
            "source",
            edge.edge_id,
            member_group,
            logical_member_group,
            port_mapping,
            attributes,
            port_conflicts,
        )
        destination, destination_port, destination_attr_port = (
            _rewrite_endpoint(
            edge.destination,
            "destination",
            edge.edge_id,
            member_group,
            logical_member_group,
            port_mapping,
            attributes,
            port_conflicts,
            )
        )

        source_base, _ = _split_endpoint(source)
        destination_base, _ = _split_endpoint(destination)
        contracted_names = set(member_group.values())
        internal_self_loop = (
            source_base == destination_base and source_base in contracted_names
        )
        if internal_self_loop and self_loops == "drop":
            deleted.append(
                DeletedEdge(
                    original_edge_id=edge.edge_id,
                    path=edge.path,
                    source=edge.source,
                    destination=edge.destination,
                    reason="self_loop",
                )
            )
            continue

        plans.append(
            _EdgePlan(
                edge_id=edge.edge_id,
                path=edge.path,
                container_index=edge.container_index,
                sequence=edge.sequence,
                source=source,
                destination=destination,
                attributes=attributes,
                source_edge_ids=(edge.edge_id,),
                original_sources=(edge.source,),
                original_destinations=(edge.destination,),
                effective_source_port=(
                    source_port
                    if source_port is not None
                    else source_attr_port
                ),
                effective_destination_port=(
                    destination_port
                    if destination_port is not None
                    else destination_attr_port
                ),
                internal_self_loop=internal_self_loop,
            )
        )

    if port_conflicts:
        raise ContractionError(port_conflicts)

    grouped: dict[tuple[int, tuple[Any, ...]], _PlanGroup] = {}
    for plan in plans:
        key = (
            plan.container_index,
            (
                plan.source,
                plan.destination,
                plan.effective_source_port,
                plan.effective_destination_port,
                plan.attributes.get("tailport"),
                plan.attributes.get("headport"),
                plan.attributes.get("ltail"),
                plan.attributes.get("lhead"),
            ),
        )
        should_merge = merge_parallel_edges or (
            self_loops == "merge" and plan.internal_self_loop
        )
        group_key = (
            (plan.container_index, plan.edge_id) if not should_merge else key
        )
        if group_key in grouped:
            grouped[group_key].plans.append(plan)
        else:
            grouped[group_key] = _PlanGroup(
                container_index=plan.container_index,
                key=key[1],
                plans=[plan],
                attributes=dict(plan.attributes),
            )

    ordered = sorted(
        grouped.values(),
        key=lambda group: min(plan.edge_id for plan in group.plans),
    )
    ordered.sort(key=lambda group: group.container_index)

    output: list[_EdgePlan] = []
    reduction_requests: list[_ReductionRequest] = []
    reduction_conflicts: list[_ConflictRequest] = []
    for group in ordered:
        if len(group.plans) == 1:
            output.append(group.plans[0])
            continue

        attributes, requests, conflicts = _edge_attribute_request(
            group.plans, attribute_reducer
        )
        reduction_requests.extend(requests)
        reduction_conflicts.extend(conflicts)
        first = group.plans[0]
        output.append(
            _EdgePlan(
                edge_id=min(plan.edge_id for plan in group.plans),
                path=first.path,
                container_index=first.container_index,
                sequence=min(plan.sequence for plan in group.plans),
                source=first.source,
                destination=first.destination,
                attributes=attributes,
                source_edge_ids=tuple(
                    sorted(plan.edge_id for plan in group.plans)
                ),
                original_sources=tuple(
                    original_source
                    for plan in sorted(
                        group.plans, key=lambda item: item.edge_id
                    )
                    for original_source in plan.original_sources
                ),
                original_destinations=tuple(
                    original_destination
                    for plan in sorted(
                        group.plans, key=lambda item: item.edge_id
                    )
                    for original_destination in plan.original_destinations
                ),
                effective_source_port=first.effective_source_port,
                effective_destination_port=(
                    first.effective_destination_port
                ),
            )
        )

    if reduction_conflicts:
        _raise_reduction_conflicts(reduction_conflicts)
    for request in reduction_requests:
        request_value = request.reducer(request.name, request.values)
        for plan in output:
            if plan.container_index != request.container_index:
                continue
            if request.edge_ids == plan.source_edge_ids:
                plan.attributes[request.name] = request_value

    output.sort(key=lambda plan: (plan.container_index, plan.edge_id))
    return tuple(output), tuple(deleted)


def _rebuild_graph(
    original: Graph,
    containers: Sequence[_Container],
    nodes: Sequence[_NodeOccurrence],
    plans: Sequence[_EdgePlan],
    groups: Sequence[_GroupSpec],
    member_group: Mapping[str, str],
    group_targets: Mapping[str, set[int]],
    node_attributes: Mapping[str, Mapping[str, Any]],
) -> Graph:
    result = copy.deepcopy(original)
    result_containers, _, _ = _traverse(result)

    nodes_by_container: dict[int, list[_NodeOccurrence]] = defaultdict(list)
    plans_by_container: dict[int, list[_EdgePlan]] = defaultdict(list)
    for occurrence in nodes:
        nodes_by_container[occurrence.container_index].append(occurrence)
    for plan in plans:
        plans_by_container[plan.container_index].append(plan)

    for container_index in range(len(containers) - 1, -1, -1):
        current = containers[container_index].graph
        rebuilt = result_containers[container_index].graph
        retained_nodes = [
            occurrence
            for occurrence in nodes_by_container.get(container_index, [])
            if occurrence.name not in member_group
        ]
        current_events: list[tuple[int, int, Any]] = []

        for occurrence in retained_nodes:
            current_events.append((occurrence.sequence, 0, occurrence))

        for group in groups:
            if container_index in group_targets.get(group.name, set()):
                member_sequences = [
                    occurrence.sequence
                    for occurrence in nodes_by_container.get(
                        container_index, []
                    )
                    if member_group.get(occurrence.name) == group.name
                ]
                member_edges = [
                    plan.edge_id
                    for plan in plans
                    if plan.container_index == container_index
                    and (
                        group.name in {
                            _split_endpoint(plan.source)[0],
                            _split_endpoint(plan.destination)[0],
                        }
                    )
                ]
                sequence_values = list(member_sequences)
                if member_edges:
                    sequence_values.append(min(member_edges))
                sequence = min(sequence_values if sequence_values else [0])
                current_events.append((sequence, 1, group))

        for occurrence in current.get_node_list():
            if occurrence.get_name() in _RESERVED_NAMES:
                current_events.append(
                    (occurrence.get_sequence() or 0, 0, occurrence)
                )

        for plan in plans_by_container.get(container_index, []):
            current_events.append((plan.sequence, 2, plan))

        for child_index, subgraph in enumerate(current.get_subgraph_list()):
            current_events.append(
                (subgraph.get_sequence() or 0, 3, child_index)
            )

        rebuilt.obj_dict["nodes"] = {}
        rebuilt.obj_dict["edges"] = {}
        rebuilt.obj_dict["subgraphs"] = {}
        rebuilt.obj_dict["current_child_sequence"] = 1

        current_events.sort(key=lambda event: (event[0], event[1]))
        for _, event_type, event in current_events:
            if event_type == 0:
                if isinstance(event, _NodeOccurrence):
                    node = Node(
                        event.name,
                        **copy.deepcopy(dict(event.attributes)),
                    )
                else:
                    node = Node(
                        event.get_name(),
                        **copy.deepcopy(event.get_attributes()),
                    )
                rebuilt.add_node(node)
            elif event_type == 1:
                rebuilt.add_node(
                    Node(
                        event.name,
                        **copy.deepcopy(dict(node_attributes[event.name])),
                    )
                )
            elif event_type == 2:
                rebuilt.add_edge(
                    Edge(
                        copy.deepcopy(event.source),
                        copy.deepcopy(event.destination),
                        **copy.deepcopy(event.attributes),
                    )
                )
            else:
                child_index = event
                rebuilt.add_subgraph(
                    result_containers[
                        containers[container_index].children[child_index]
                    ].graph
                )

    result.set_parent_graph(result)
    return result


def contract_nodes(
    graph: Graph,
    groups: Any,
    attribute_reducer: AttributeReducer | None = None,
    merge_parallel_edges: bool = False,
    self_loops: str | bool = "keep",
    port_mapping: Any = None,
) -> ContractionResult:
    """Contract disjoint node groups in a directed multigraph.

    ``groups`` maps new node names to member-name sequences. Each new node
    may also map to ``{"members": [...], "attributes": {...}}``.

    Every original edge is either represented by one ``EdgeOrigin`` or one
    ``DeletedEdge``. Parallel edges are kept unless ``merge_parallel_edges``
    is true; conflicting merged attributes call ``attribute_reducer``.

    Ports on contracted endpoints require ``port_mapping``. It may be a
    mapping or callable receiving a :class:`PortMappingContext`.
    """

    if not isinstance(graph, Graph) or graph.get_type() != "digraph":
        raise ValueError("Only directed multigraphs can be contracted.")
    if graph.get_strict():
        raise ValueError("Strict directed graphs are not multigraphs.")

    self_loop_policy = _normalise_self_loops(self_loops)
    normalised_groups = _normalise_groups(groups)
    containers, nodes, edges = _traverse(graph)
    (
        member_group,
        logical_member_group,
        _,
        group_targets,
        _,
    ) = _validate_and_locate(normalised_groups, nodes, edges)

    node_attributes, node_reductions, node_conflicts = (
        _node_attribute_requests(
            normalised_groups,
            nodes,
            attribute_reducer,
        )
    )
    plans, deleted = _build_plans(
        edges,
        member_group,
        logical_member_group,
        port_mapping,
        attribute_reducer,
        merge_parallel_edges,
        self_loop_policy,
    )

    if node_conflicts:
        _raise_reduction_conflicts(node_conflicts)
    for request in node_reductions:
        assert request.group is not None
        value = request.reducer(request.name, request.values)
        node_attributes[request.group][request.name] = value

    rebuilt = _rebuild_graph(
        graph,
        containers,
        nodes,
        plans,
        normalised_groups,
        member_group,
        group_targets,
        node_attributes,
    )

    origins: list[EdgeOrigin] = []
    for new_edge_id, plan in enumerate(plans):
        origins.append(
            EdgeOrigin(
                new_edge_id=new_edge_id,
                original_edge_ids=plan.source_edge_ids,
                path=plan.path,
                source=plan.source,
                destination=plan.destination,
                sources=plan.original_sources,
                destinations=plan.original_destinations,
            )
        )

    accounted_for = {
        edge_id
        for origin in origins
        for edge_id in origin.original_edge_ids
    } | {deleted_edge.original_edge_id for deleted_edge in deleted}
    assert accounted_for == {edge.edge_id for edge in edges}

    node_sources = {
        group.name: group.members for group in normalised_groups
    }
    return ContractionResult(
        graph=rebuilt,
        node_sources=node_sources,
        edge_origins=tuple(origins),
        deleted_edges=deleted,
    )
