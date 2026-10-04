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


class XDotError(PydotException):
    """Base class for errors while parsing XDOT drawing instructions.

    The byte offset where the malformed instruction starts is kept in
    ``position``.
    """

    position: int

    def __init__(self, message: str, position: int = 0) -> None:
        super().__init__(message)
        self.position = position

    def __str__(self) -> str:
        return f"{self.args[0]} (at byte {self.position})"


class XDotTruncatedError(XDotError):
    """An instruction ran past the end of the input or was malformed."""


class XDotNegativeLengthError(XDotError):
    """A length-prefixed string carried a negative byte count."""


class XDotUnknownOpError(XDotError):
    """An opcode character outside the supported XDOT subset."""


class XDotPointCountError(XDotError):
    """An instruction declared a point/stop count it could not supply."""
