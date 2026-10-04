# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Effective graph view.

The parsed object tree preserves the DOT source as written: default
statements (``node [...]`` / ``edge [...]`` / ``graph [...]``) are kept
as pseudo-nodes and implicit nodes created by edge statements are not
materialized.  Reading ``Node.get_attributes()`` therefore misses the
attributes an object receives by default at the point where it is
declared.

``effective_graph`` walks that tree in DOT declaration order and
computes the objects as Graphviz sees them:

* graph, node and edge defaults are scoped to the graph/subgraph in
  which they are declared (child scopes inherit a snapshot of the
  parent defaults, later changes never propagate back or back in
  time);
* edge statements implicitly create their endpoint nodes;
* a node restated (also across subgraphs) keeps its identity and
  receives the new explicit attributes;
* subgraphs used as edge endpoints expand to the cartesian product of
  their member nodes;
* strict graphs merge duplicate edges (unordered endpoints in
  undirected graphs), while ordinary graphs keep parallel edges.

The original declaration tree is never modified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import pydot.core
from pydot.classes import FrozenDict

if TYPE_CHECKING:
    from pydot.core import Common, Edge, Graph, Node



def _node_id(endpoint: Any) -> str:
    """Return the node identity of an edge endpoint reference.

    Mirrors ``Node.__init__``: an unquoted ``name:port`` reference
    identifies the node without its port, while quoted names keep
    their literal form.
    """
    if isinstance(endpoint, str):
        if not endpoint.startswith('"'):
            idx = endpoint.find(":")
            if 0 < idx + 1 < len(endpoint):
                return endpoint[:idx]
        return endpoint
    return str(endpoint)


def _is_keyword(name: Any, keyword: str) -> bool:
    """Whether `name` denotes the pseudo-object keyword (not a quoted ID)."""
    return (
        isinstance(name, str)
        and not name.startswith('"')
        and name.lower() == keyword
    )


class EffectiveGraph:
    """Resolved view of a graph as declared in DOT source.

    Returned by :func:`effective_graph` and ``Graph.effective_graph()``.
    Nodes are exposed in first-declaration order; subviews mirror the
    declared subgraph tree.
    """

    def __init__(
        self,
        builder: "_Builder",
        obj_dict: Any,
        top_type: str,
        strict: bool,
        is_root: bool = False,
    ) -> None:
        self._builder = builder
        self._is_root = is_root
        self._name = obj_dict.get("name")
        self._type = obj_dict.get("type", top_type)
        self._top_type = top_type
        self._strict = strict
        self.attributes: dict[str, Any] = dict(
            obj_dict.get("attributes", {}) or {}
        )
        self._member_nodes: list[str] = []
        self._declared_edges: list[Edge] = []
        self._subgraphs: list[EffectiveGraph] = []

    def get_name(self) -> Any:
        """Get the (sub)graph name."""
        return self._name

    def get_type(self) -> str:
        """Get the declared type: 'graph', 'digraph' or 'subgraph'."""
        return self._type  # type: ignore[no-any-return]

    def get_graph_type(self) -> str:
        """Get the effective graph type ('graph' or 'digraph')."""
        return self._top_type

    def get_strict(self) -> bool:
        """Whether the top-level graph is strict."""
        return self._strict

    def get_attributes(self) -> dict[str, Any]:
        """Get the effective graph/subgraph attributes."""
        return self.attributes

    def get_node(self, name: str) -> list[Node]:
        """Get the effective node with `name` as a one-element list.

        Returns an empty list if no such node exists or, for a subgraph
        view, if the node is not a member of that subgraph.
        """
        identity = _node_id(name)
        if identity not in self.get_node_name_list():
            return []
        node = self._builder.nodes.get(identity)
        return [node] if node is not None else []

    def get_node_name_list(self) -> list[str]:
        """Names of the effective nodes visible from this (sub)graph."""
        if self._is_root:
            return list(self._builder.node_order)
        return list(self._member_nodes)

    def get_nodes(self) -> list[Node]:
        """Get all effective nodes visible from this (sub)graph, in order."""
        return self.get_node_list()

    def get_node_list(self) -> list[Node]:
        """Get all effective nodes visible from this (sub)graph, in order."""
        return [
            self._builder.nodes[name]
            for name in self.get_node_name_list()
            if name in self._builder.nodes
        ]

    def get_edge(
        self, src: Any = None, dst: Any = None
    ) -> list[Edge]:
        """Get effective edges, optionally filtered by endpoint pair.

        In undirected graphs the endpoint order is irrelevant.
        """
        edges = self.get_edge_list()
        if src is None and dst is None:
            return list(edges)

        wanted = self._builder.edge_key(src, dst)
        return [edge for edge in edges if self._edge_key(edge) == wanted]

    def get_edges(self) -> list[Edge]:
        """Get effective edges declared in this (sub)graph."""
        return self.get_edge_list()

    def get_edge_list(self) -> list[Edge]:
        """Get effective edges declared in this (sub)graph."""
        if self._is_root:
            return list(self._builder.edges)
        return list(self._declared_edges)

    def get_subgraph(self, name: str) -> list[EffectiveGraph]:
        """Get direct subgraph views carrying `name` (a list, 0..n)."""
        return [sg for sg in self._subgraphs if sg.get_name() == name]

    def get_subgraphs(self) -> list[EffectiveGraph]:
        """Get direct subgraph views in declaration order."""
        return self.get_subgraph_list()

    def get_subgraph_list(self) -> list[EffectiveGraph]:
        """Get direct subgraph views in declaration order."""
        return list(self._subgraphs)

    def _edge_key(self, edge: Edge) -> tuple[str, ...]:
        return self._builder.edge_key(
            edge.get_source(), edge.get_destination()
        )

    def __iter__(self) -> Iterator[Node]:
        return iter(self.get_node_list())

    def __len__(self) -> int:
        return len(self.get_node_list())

    def __repr__(self) -> str:
        return (
            f"EffectiveGraph(name={self._name!r}, type={self._type!r}, "
            f"nodes={len(self._builder.node_order)}, "
            f"edges={len(self._builder.edges)})"
        )


class _Scope:
    """Declaration scope of one graph or subgraph body."""

    def __init__(
        self,
        view: EffectiveGraph,
        parent: _Scope | None,
    ) -> None:
        self.view = view
        self.parent = parent
        if parent is None:
            self.node_defaults: dict[str, Any] = {}
            self.edge_defaults: dict[str, Any] = {}
        else:
            # Snapshot of the enclosing defaults at entry time.
            self.node_defaults = dict(parent.node_defaults)
            self.edge_defaults = dict(parent.edge_defaults)
            parent.children.append(self)
        self.children: list[_Scope] = []
        self.own_members: list[str] = []
        self._all_members: list[str] | None = None

    def touch(self, node_name: str) -> None:
        if node_name not in self.own_members:
            self.own_members.append(node_name)

    def all_members(self) -> list[str]:
        """Member nodes of this scope, including nested subgraphs."""
        if self._all_members is not None:
            return self._all_members
        members = list(self.own_members)
        for child in self.children:
            for name in child.all_members():
                if name not in members:
                    members.append(name)
        self._all_members = members
        return members


class _Builder:
    """Walks the parsed tree in declaration order once."""

    def __init__(self, graph: Graph) -> None:
        top = graph
        parent = graph.get_parent_graph()
        while parent is not None and parent is not top:
            top = parent
            parent = top.get_parent_graph()

        self.top_graph = top
        self.top_type = str(top.obj_dict.get("type", "graph"))
        if self.top_type == "subgraph":
            self.top_type = "graph"
        self.strict = bool(top.obj_dict.get("strict", False))

        self.nodes: dict[str, Node] = {}
        self.node_order: list[str] = []
        self.edges: list[Edge] = []
        self._edge_index: dict[tuple[str, ...], Edge] = {}
        self.root_view: EffectiveGraph | None = None

    def edge_key(self, src: Any, dst: Any) -> tuple[str, ...]:
        src_id, dst_id = _node_id(src), _node_id(dst)
        if self.top_type == "graph":
            return tuple(sorted((src_id, dst_id)))
        return (src_id, dst_id)

    def build(self, root: Graph) -> EffectiveGraph:
        view = EffectiveGraph(
            self, root.obj_dict, self.top_type, self.strict, is_root=True
        )
        self.root_view = view
        scope = _Scope(view, None)
        self._process(root.obj_dict, scope)
        view._member_nodes = list(self.node_order)
        return view

    def _sorted_statements(
        self, obj_dict: Any
    ) -> list[tuple[str, dict[str, Any]]]:
        statements: list[tuple[Any, str, dict[str, Any]]] = []
        for obj_list in obj_dict.get("nodes", {}).values():
            for od in obj_list:
                statements.append((od.get("sequence"), "node", od))
        for obj_list in obj_dict.get("edges", {}).values():
            for od in obj_list:
                statements.append((od.get("sequence"), "edge", od))
        for obj_list in obj_dict.get("subgraphs", {}).values():
            for od in obj_list:
                statements.append((od.get("sequence"), "subgraph", od))
        # Stable sort: equal sequences keep insertion order and any
        # statement without a sequence is placed last.
        statements.sort(key=lambda item: (item[0] is None, item[0]))
        return [(kind, od) for _, kind, od in statements]

    def _process(self, obj_dict: Any, scope: _Scope) -> None:
        for kind, od in self._sorted_statements(obj_dict):
            if kind == "node":
                self._process_node_stmt(od, scope)
            elif kind == "edge":
                self._process_edge_stmt(od, scope)
            else:
                self._process_subgraph_stmt(od, scope)

    def _process_node_stmt(
        self, obj_dict: dict[str, Any], scope: _Scope
    ) -> None:
        name = obj_dict.get("name")
        attrs = dict(obj_dict.get("attributes", {}) or {})

        if _is_keyword(name, "graph"):
            scope.view.attributes.update(attrs)
        elif _is_keyword(name, "node"):
            scope.node_defaults.update(attrs)
        elif _is_keyword(name, "edge"):
            scope.edge_defaults.update(attrs)
        else:
            self._ensure_node(scope, name, attrs)

    def _process_subgraph_stmt(
        self, obj_dict: dict[str, Any], scope: _Scope
    ) -> None:
        view = EffectiveGraph(self, obj_dict, self.top_type, self.strict)
        scope.view._subgraphs.append(view)
        child_scope = _Scope(view, scope)
        self._process(obj_dict, child_scope)
        view._member_nodes = child_scope.all_members()

    def _process_edge_stmt(
        self, obj_dict: dict[str, Any], scope: _Scope
    ) -> None:
        points = obj_dict.get("points", ())
        if len(points) < 2:  # pragma: no cover - parser always provides two
            return
        explicit_attrs = dict(obj_dict.get("attributes", {}) or {})

        src_raw, dst_raw = points[0], points[1]
        tails = self._resolve_endpoint(src_raw, scope)
        heads = self._resolve_endpoint(dst_raw, scope)

        # Plain (string) endpoints are nodes of the current scope. Their
        # first-touch order is tails first, heads afterwards. The node
        # identity strips the port, but the raw reference (which may
        # carry a port) is kept on the resulting edge.
        if isinstance(src_raw, str):
            self._ensure_node(scope, src_raw, {})
        if isinstance(dst_raw, str):
            self._ensure_node(scope, dst_raw, {})

        base_attrs = dict(scope.edge_defaults)
        for tail in tails:
            edge_tail = src_raw if isinstance(src_raw, str) else tail
            for head in heads:
                edge_head = dst_raw if isinstance(dst_raw, str) else head
                attrs = dict(base_attrs)
                attrs.update(explicit_attrs)
                edge = self._make_edge(
                    edge_tail, edge_head, attrs, explicit_attrs
                )
                scope.view._declared_edges.append(edge)

    def _resolve_endpoint(
        self, endpoint: Any, scope: _Scope
    ) -> list[str]:
        """Expand an edge endpoint into the node names it connects."""
        if isinstance(endpoint, FrozenDict):
            view = EffectiveGraph(
                self, endpoint, self.top_type, self.strict
            )
            endpoint_scope = _Scope(view, scope)
            self._process(endpoint, endpoint_scope)
            return endpoint_scope.all_members()
        return [_node_id(endpoint)]

    def _ensure_node(
        self,
        scope: _Scope,
        endpoint: Any,
        explicit_attrs: dict[str, Any],
    ) -> Node:
        identity = _node_id(endpoint)
        node = self.nodes.get(identity)
        if node is None:
            attrs = dict(scope.node_defaults)
            attrs.update(explicit_attrs)
            node = pydot.core.Node(str(endpoint), **attrs)
            node.set_parent_graph(self.top_graph)
            self.nodes[identity] = node
            self.node_order.append(identity)
        elif explicit_attrs:
            node.obj_dict["attributes"].update(explicit_attrs)
        scope.touch(identity)
        return node

    def _make_edge(
        self,
        tail: str,
        head: str,
        attrs: dict[str, Any],
        explicit_attrs: dict[str, Any],
    ) -> Edge:
        key = self.edge_key(tail, head)
        existing = self._edge_index.get(key)
        if self.strict and existing is not None:
            # Merge only this statement's explicit attributes. Edge
            # defaults in effect later are not applied retroactively to
            # the edge already created by the first statement.
            existing.obj_dict["attributes"].update(explicit_attrs)
            return existing

        edge = pydot.core.Edge(tail, head, **attrs)
        edge.set_parent_graph(self.top_graph)
        self.edges.append(edge)
        self._edge_index[key] = edge
        return edge


def effective_graph(graph: Graph | Common) -> EffectiveGraph:
    """Compute the effective graph view of a parsed/built `graph`.

    The returned :class:`EffectiveGraph` resolves default attributes,
    implicit node creation, subgraph endpoint expansion and strict
    duplicate-edge merging in DOT declaration order. The input graph
    is left untouched.
    """
    if not isinstance(graph, pydot.core.Graph):
        raise TypeError(
            "effective_graph() expects a pydot Graph, got: "
            + type(graph).__name__
        )
    return _Builder(graph).build(graph)
