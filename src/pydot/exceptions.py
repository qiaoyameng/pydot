# SPDX-FileCopyrightText: 2024 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Exception classes for pydot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class ContractionConflict:
    """A locatable problem found while validating a node contraction.

    ``paths`` contain subgraph paths; an empty tuple identifies the root
    graph. For edge-related conflicts, ``edge_id`` is the zero-based ID
    assigned while depth-first traversing graph and subgraph edges.
    """

    kind: str
    message: str
    group: Optional[str] = None
    node: Optional[str] = None
    edge_id: Optional[int] = None
    endpoint: Optional[Any] = None
    attribute: Optional[str] = None
    values: Tuple[Any, ...] = field(default_factory=tuple)
    edge_ids: Tuple[int, ...] = field(default_factory=tuple)
    container_indices: Tuple[int, ...] = field(default_factory=tuple)
    paths: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)


class ContractionError(Error):
    """One or more node contraction conflicts were found."""

    def __init__(
        self, conflicts: Sequence[ContractionConflict], value: str = ""
    ) -> None:
        self.conflicts = tuple(conflicts)
        if not value:
            value = "; ".join(conflict.message for conflict in self.conflicts)
        super().__init__(value)
