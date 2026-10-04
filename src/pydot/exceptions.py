# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Exception classes for pydot."""

from __future__ import annotations


class PydotException(Exception):
    """Base class for exceptions in Pydot.

    This base class will not be raised directly.

    Catch this base class to catch all derived exceptions, though be
    aware that pydot may raise Python built-in exceptions or pyparsing
    exceptions as well.
    """


class Error(PydotException):
    """General error handling class."""

    value: str

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class ContractionError(PydotException):
    """Base class for errors raised by node contraction."""


class OverlappingGroupsError(ContractionError):
    """A node was assigned to more than one contraction group.

    ``overlapping`` maps each offending node name to the sorted list
    of group names that claimed it.
    """

    def __init__(self, overlapping: dict[str, list[str]]) -> None:
        self.overlapping = overlapping
        details = ", ".join(
            f"{node!r} in groups {groups}"
            for node, groups in sorted(overlapping.items())
        )
        super().__init__(f"Overlapping contraction groups: {details}")


class UnknownNodeError(ContractionError):
    """A contraction group mentions a node absent from the graph.

    ``unknown`` lists the node names that were not found.
    """

    def __init__(self, unknown: list[str]) -> None:
        self.unknown = unknown
        super().__init__(
            "Contraction groups reference unknown nodes: "
            + ", ".join(repr(name) for name in unknown)
        )


class PortMappingError(ContractionError):
    """An edge endpoint uses a port that has no contracted port mapping.

    ``node`` and ``port`` identify the endpoint that could not be
    contracted, ``edge`` the (source, destination) pair of the edge
    it appeared on.
    """

    def __init__(
        self, node: str, port: str, edge: tuple[str, str]
    ) -> None:
        self.node = node
        self.port = port
        self.edge = edge
        super().__init__(
            f"Edge {edge[0]!r} -> {edge[1]!r} uses port {port!r} on "
            f"contracted node {node!r}; provide a port_map entry for "
            f"({node!r}, {port!r}) or remove the port before "
            "contracting."
        )


class AttributeConflictError(ContractionError):
    """Conflicting attribute values and no reducer was provided.

    ``attribute`` is the conflicting attribute name, ``values`` the
    conflicting values in encounter order, and ``context`` describes
    where the conflict happened (a contracted node or merged edges).
    """

    def __init__(
        self, attribute: str, values: list[object], context: str
    ) -> None:
        self.attribute = attribute
        self.values = values
        self.context = context
        super().__init__(
            f"Conflicting values {values!r} for attribute "
            f"{attribute!r} while contracting {context}; pass an "
            "attribute_reducer to resolve the conflict explicitly."
        )


class ClusterConflictError(ContractionError):
    """Members of a contraction group disagree on cluster membership.

    ``group`` is the contracted group name. ``memberships`` maps each
    member node name to the list of cluster paths (tuples of subgraph
    names, outermost first) it is declared in; an empty tuple means
    the top-level graph. This pinpoints exactly which nodes sit in
    which clusters.
    """

    def __init__(
        self, group: str, memberships: dict[str, list[tuple[str, ...]]]
    ) -> None:
        self.group = group
        self.memberships = memberships
        lines = [
            f"  {node!r}: "
            + (
                ", ".join(
                    "/".join(path) if path else "<top-level>"
                    for path in paths
                )
            )
            for node, paths in sorted(memberships.items())
        ]
        super().__init__(
            f"Cannot contract group {group!r}: its members belong to "
            "different clusters:\n" + "\n".join(lines)
        )
