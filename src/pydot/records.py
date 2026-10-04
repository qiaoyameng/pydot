# SPDX-FileCopyrightText: 2026 pydot contributors
#
# SPDX-License-Identifier: MIT

"""Parsing, building and validating Graphviz ``record``-shaped labels.

Nodes declared with shape ``record`` or ``Mrecord`` use a small label
language to lay out nested fields::

    {<port_id> text | <other_id> other text | {sub | fields}}

The pieces are:

* text -- the literal characters shown in the field;
* ``|``  -- a field separator;
* ``{`` / ``}`` -- a nested field group;
* ``<port>`` -- a port declaration.

The characters with special meaning may be written literally using a
backslash escape (``\\|``, ``\\{``, ``\\}``, ``\\<``, ``\\>``,
``\\ ``) or by quoting a run of text with double quotes.

This module turns such labels into nested trees of :class:`RecordField`,
builds labels back from trees, derives the arrangement direction of
each level from the graph ``rankdir``, and checks for duplicate port
declarations and edges that reference ports that do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Sequence, Union

if TYPE_CHECKING:
    import pydot

RECORD_SHAPES: Final = frozenset({"record", "Mrecord"})

# Compass points that may legally follow the port in an edge endpoint,
# e.g. ``node:port:n`` or ``node:nw``.
_COMPASS_POINTS: Final = frozenset(
    {
        "n", "ne", "e", "se", "s", "sw", "w", "nw",
        "c", "_", "center",
    }
)

# Characters that get a backslash in front of them when a label is
# serialized. Whitespace is included because unescaped whitespace is
# ignored by the record scanner.
_ESCAPABLE: Final = '|{}<> " \t\r\n\\'

FieldTree: Final = Union[
    str,
    tuple,
    list,
    dict,
    "RecordField",
    None,
]


class RecordError(ValueError):
    """Base class for errors involving record labels."""


class RecordSyntaxError(RecordError):
    """A record label cannot be parsed.

    The ``path`` describes the field path (relative to the node) at
    which the problem was detected, when it can be determined.
    """

    def __init__(self, message: str, path: Sequence[str] = ()) -> None:
        self.path = tuple(path)
        where = (" at '" + ".".join(self.path) + "'") if self.path else ""
        super().__init__(message + where)


class DuplicatePortError(RecordError):
    """The same port is declared more than once on one node."""

    def __init__(self, node: str, port: str, path: Sequence[str]) -> None:
        self.node = node
        self.port = port
        self.path = tuple(path)
        super().__init__(
            f"Duplicate port {port!r} on record node {node!r} "
            f"at field '{'.'.join(self.path)}'"
        )


class UnknownPortError(RecordError):
    """An edge references a port that a record node does not declare."""

    def __init__(
        self,
        node: str,
        port: str,
        path: Sequence[str] = (),
        edge: object = None,
    ) -> None:
        self.node = node
        self.port = port
        self.path = tuple(path)
        self.edge = edge
        prefix = f"Edge {edge} " if edge is not None else ""
        super().__init__(
            f"{prefix}references undefined port {port!r} "
            f"on record node {node!r}"
        )


@dataclass
class RecordField:
    """One field of a record label.

    ``text`` holds the actual text shown for the field after escapes
    have been resolved (the empty string for empty fields). ``port``
    holds the declared port identifier, if any. ``fields``
    distinguishes a grouping field from a leaf: it is ``None`` for a
    leaf and a list of child fields for a group.
    """

    text: str = ""
    port: str | None = None
    fields: list[RecordField] | None = None

    @property
    def is_group(self) -> bool:
        """Whether this field is a grouping (``{ ... }``) field."""
        return self.fields is not None

    def find(self, port: str) -> RecordField | None:
        """Return the (possibly nested) field declaring ``port``."""
        if self.port == port:
            return self
        if self.fields is None:
            return None
        for child in self.fields:
            found = child.find(port)
            if found is not None:
                return found
        return None

    def port_path(
        self, port: str, prefix: tuple[str, ...] = ()
    ) -> tuple[str, ...] | None:
        """Return the path of the field declaring ``port``."""
        if self.port == port:
            return prefix
        if self.fields is None:
            return None
        for index, child in enumerate(self.fields):
            found = child.port_path(port, prefix + (f"[{index}]",))
            if found is not None:
                return found
        return None

    def ports(self) -> set[str]:
        """Return every port declared in this field or its children."""
        result = {self.port} if self.port is not None else set()
        if self.fields is not None:
            for child in self.fields:
                result.update(child.ports())
        return result


def _escape(text: str) -> str:
    out: list[str] = []
    for char in text:
        out.append("\\" + char if char in _ESCAPABLE else char)
    return "".join(out)


def _serialize_field(field: RecordField) -> str:
    if field.is_group:
        body = "|".join(
            _serialize_field(child) for child in field.fields or []
        )
        group = "{" + body + "}"
        head = ""
        if field.port is not None:
            head += f"<{_escape(field.port)}>"
        if field.text:
            head += _escape(field.text)
        return head + group

    if field.port is not None and field.text:
        return f"<{_escape(field.port)}> {_escape(field.text)}"
    if field.port is not None:
        return f"<{_escape(field.port)}>"
    return _escape(field.text)


class _Scanner:
    """Character scanner over a DOT-level unescaped record label."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.pos = 0

    @property
    def done(self) -> bool:
        return self.pos >= len(self.raw)

    def _error(self, message: str) -> RecordSyntaxError:
        return RecordSyntaxError(message)

    def _skip_ws(self) -> None:
        while not self.done and self.raw[self.pos].isspace():
            self.pos += 1

    def _read_quoted_run(self, out: list[str]) -> None:
        # Assumes the current character is '"'.
        self.pos += 1
        while not self.done:
            char = self.raw[self.pos]
            if char == '"':
                self.pos += 1
                return
            if char == "\\":
                self.pos += 1
                if self.done:
                    raise self._error("Unterminated escape in quoted text")
            out.append(self.raw[self.pos])
            self.pos += 1
        raise self._error("Unterminated quoted text")

    def _read_port(self) -> str:
        # Assumes the current character is '<'.
        self.pos += 1
        chars: list[str] = []
        self._skip_ws()
        while not self.done:
            char = self.raw[self.pos]
            if char == ">":
                self.pos += 1
                self._skip_ws()
                if not chars:
                    raise self._error("Empty port declaration '<>'")
                return "".join(chars)
            if char == '"':
                self._read_quoted_run(chars)
                continue
            if char == "\\":
                self.pos += 1
                if self.done:
                    raise self._error("Unterminated escape in port")
                chars.append(self.raw[self.pos])
            elif char.isspace():
                pass
            elif char in "|{}<>":
                raise self._error(
                    f"Unexpected {char!r} inside port declaration"
                )
            else:
                chars.append(char)
            self.pos += 1
        raise self._error("Unterminated port declaration (missing '>')")

    def _read_group(self) -> RecordField:
        # Assumes the current character is '{'.
        self.pos += 1
        children = self._read_fields(stop_at_brace=True)
        self._skip_ws()
        if self.done or self.raw[self.pos] != "}":
            raise self._error("Unterminated field group (missing '}')")
        self.pos += 1
        return RecordField(fields=children)

    def _read_field(self) -> RecordField:
        """Read exactly one field (the cursor is not on '|' or '}')."""
        port: str | None = None
        text_chars: list[str] = []
        pending_ws: list[str] = []
        group: RecordField | None = None

        def flush_ws() -> None:
            # Keep at most one space, and only if text already exists.
            if pending_ws and text_chars:
                text_chars.append(" ")
            pending_ws.clear()

        self._skip_ws()
        if not self.done and self.raw[self.pos] == "<":
            port = self._read_port()

        while not self.done:
            char = self.raw[self.pos]
            if char in "|}":
                break
            if char == "{":
                if group is not None:
                    raise self._error(
                        "A field may contain only one nested group"
                    )
                pending_ws.clear()
                while text_chars and text_chars[-1] == " ":
                    text_chars.pop()
                group = self._read_group()
                self._skip_ws()
                continue
            if char == "<":
                raise self._error(
                    "Unexpected '<': a port may only start a field"
                )
            if char == '"':
                flush_ws()
                self._read_quoted_run(text_chars)
                continue
            if char == "\\":
                self.pos += 1
                if self.done:
                    raise self._error("Unterminated escape in field text")
                flush_ws()
                text_chars.append(self.raw[self.pos])
            elif char.isspace():
                pending_ws.append(char)
            else:
                flush_ws()
                text_chars.append(char)
            self.pos += 1

        if group is not None:
            group.port = port
            group.text = "".join(text_chars)
            return group
        return RecordField(text="".join(text_chars), port=port)

    def _read_fields(self, stop_at_brace: bool) -> list[RecordField]:
        fields: list[RecordField] = []
        at_start = True
        trailing_separator = False
        while True:
            self._skip_ws()
            if self.done:
                if stop_at_brace:
                    raise self._error(
                        "Unterminated field group (missing '}')"
                    )
                if trailing_separator:
                    fields.append(RecordField())
                return fields
            char = self.raw[self.pos]
            if char == "}":
                if not stop_at_brace:
                    raise self._error("Unexpected '}' with no matching '{'")
                if trailing_separator:
                    fields.append(RecordField())
                return fields
            if char == "|":
                if at_start or trailing_separator:
                    fields.append(RecordField())
                trailing_separator = True
                at_start = False
                self.pos += 1
                continue
            fields.append(self._read_field())
            trailing_separator = False
            at_start = False

    def parse(self) -> list[RecordField]:
        fields = self._read_fields(stop_at_brace=False)
        self._skip_ws()
        if not self.done:
            raise self._error(f"Unexpected {self.raw[self.pos]!r}")
        return fields


def _dot_unescape(label: str) -> str:
    """Remove the DOT quoting layer from a stored label value.

    pydot stores attribute values exactly as written in the DOT source,
    so a quoted label arrives with its surrounding quotes and DOT
    escapes (``\\n`` etc.). HTML-like labels are not record labels and
    are handled by the caller.
    """
    if len(label) >= 2 and label.startswith('"') and label.endswith('"'):
        body = label[1:-1]
        out: list[str] = []
        index = 0
        while index < len(body):
            char = body[index]
            if char == "\\" and index + 1 < len(body):
                nxt = body[index + 1]
                if nxt == '"':
                    out.append('"')
                elif nxt == "\\":
                    out.append("\\")
                elif nxt == "n":
                    out.append("\n")
                elif nxt == "l":
                    out.append("\n")
                elif nxt == "r":
                    out.append("\r")
                elif nxt == "t":
                    out.append("\t")
                else:
                    # A backslash that the DOT scanner keeps verbatim,
                    # e.g. the start of a record-level escape.
                    out.append("\\")
                    out.append(nxt)
                index += 2
                continue
            out.append(char)
            index += 1
        return "".join(out)
    return label


def _coerce_field(spec: FieldTree) -> RecordField:
    if isinstance(spec, RecordField):
        return spec
    if spec is None:
        return RecordField()
    if isinstance(spec, str):
        return RecordField(text=spec)
    if isinstance(spec, dict):
        children = spec.get("fields")
        return RecordField(
            text=str(spec.get("text", "")),
            port=spec.get("port"),
            fields=[_coerce_field(child) for child in children]
            if children is not None
            else None,
        )
    if isinstance(spec, list):
        # A bare list is a grouping field whose items are children.
        return RecordField(
            fields=[_coerce_field(child) for child in spec]
        )
    if isinstance(spec, tuple):
        # ('<port>', 'text', [children]) -- port and text optional,
        # a trailing list of children marks a grouping field.
        items = list(spec)
        port: str | None = None
        if (
            items
            and isinstance(items[0], str)
            and items[0].startswith("<")
            and items[0].endswith(">")
            and len(items[0]) > 1
        ):
            port = items[0][1:-1]
            items = items[1:]

        children: list | None = None
        if items and isinstance(items[-1], list):
            children = [_coerce_field(child) for child in items[-1]]
            items = items[:-1]

        if len(items) > 1:
            raise RecordError(
                f"Cannot interpret field specification {spec!r}: "
                "use ('<port>', 'text', [children])"
            )
        text = str(items[0]) if items else ""
        return RecordField(
            text=text,
            port=port,
            fields=children,
        )
    raise RecordError(f"Unsupported field specification: {spec!r}")


class RecordLabel:
    """A parsed or constructed record label.

    The root always holds the ordered, nested list of fields.
    """

    def __init__(
        self,
        fields: Sequence[RecordField | FieldTree] | None = None,
    ) -> None:
        self.fields: list[RecordField] = (
            [_coerce_field(item) for item in fields] if fields else []
        )

    # -- construction ------------------------------------------------

    @classmethod
    def build(cls, fields: Sequence[FieldTree] | None = None) -> RecordLabel:
        """Build a record label from a nested field specification.

        Accepted leaf forms are a plain string (text), ``None`` (empty
        field), or :class:`RecordField`; a list/tuple is interpreted as
        ``('<port>', text, [children])`` where the port and text are
        optional and a nested list marks a grouping field.
        """
        return cls(fields)

    def add_field(
        self,
        text: str = "",
        port: str | None = None,
        fields: Sequence[FieldTree] | None = None,
    ) -> RecordField:
        """Append a leaf or grouping field and return it."""
        field = RecordField(
            text=text,
            port=port,
            fields=[_coerce_field(child) for child in fields]
            if fields is not None
            else None,
        )
        self.fields.append(field)
        return field

    # -- parsing -----------------------------------------------------

    @classmethod
    def parse(cls, label: str) -> RecordLabel:
        """Parse a record label string into a nested field tree.

        ``label`` may be the raw attribute value as stored by pydot,
        including surrounding DOT quotes. Only labels of ``record`` and
        ``Mrecord`` nodes follow this syntax.
        """
        scanner = _Scanner(_dot_unescape(label))
        self = cls.__new__(cls)
        self.fields = scanner.parse()
        return self

    # -- serialization ----------------------------------------------

    def serialize(self) -> str:
        """Serialize back to the raw (unquoted) record label string."""
        return "|".join(_serialize_field(field) for field in self.fields)

    def to_string(self) -> str:
        """Alias of :meth:`serialize`."""
        return self.serialize()

    def __str__(self) -> str:
        return self.serialize()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordLabel):
            return NotImplemented
        return self.fields == other.fields

    # -- inspection --------------------------------------------------

    def ports(self) -> set[str]:
        """Return the set of all declared port identifiers."""
        result: set[str] = set()
        for field in self.fields:
            result.update(field.ports())
        return result

    def find(self, port: str) -> RecordField | None:
        """Return the field declaring ``port``, searching nested groups."""
        for field in self.fields:
            found = field.find(port)
            if found is not None:
                return found
        return None

    def port_path(self, port: str) -> tuple[str, ...] | None:
        """Return the field path at which ``port`` is declared."""
        for index, field in enumerate(self.fields):
            found = field.port_path(port, (f"[{index}]",))
            if found is not None:
                return found
        return None

    def check_duplicate_ports(self, node: str = "?") -> None:
        """Raise :class:`DuplicatePortError` on repeated port names."""
        seen: dict[str, tuple[str, ...]] = {}

        def visit(
            fields: Sequence[RecordField], prefix: tuple[str, ...]
        ) -> None:
            for index, field in enumerate(fields):
                path = prefix + (f"[{index}]",)
                if field.port is not None:
                    if field.port in seen:
                        raise DuplicatePortError(node, field.port, path)
                    seen[field.port] = path
                if field.is_group:
                    visit(field.fields or [], path)

        visit(self.fields, ())

    # -- layout ------------------------------------------------------

    def field_directions(self, rankdir: str = "TB") -> dict:
        """Map field paths to their arrangement direction.

        The key ``()`` is the top-level direction; each grouping field
        is keyed by its index path (e.g. ``("[2]", "[0]")``). A single
        brace-enclosed group at the root is transparent, as in
        Graphviz: ``"{a|b}"`` lays out exactly like ``"a|b"``.
        """
        result: dict[tuple[str, ...], str] = {}

        def walk(
            fields: Sequence[RecordField],
            level: int,
            prefix: tuple[str, ...],
        ) -> None:
            result[prefix] = _direction_for_level(level, rankdir)
            for index, field in enumerate(fields):
                if field.is_group:
                    walk(
                        field.fields or [],
                        level + 1,
                        prefix + (f"[{index}]",),
                    )

        if len(self.fields) == 1 and self.fields[0].is_group:
            # A lone outer group is transparent for layout.
            result[()] = _direction_for_level(0, rankdir)
            walk(self.fields[0].fields or [], 0, ("[0]",))
        else:
            walk(self.fields, 0, ())
        return result

    def level_directions(self, rankdir: str = "TB") -> list[str]:
        """Return the arrangement direction of every nesting level.

        Element 0 describes the top-level fields; each subsequent
        element describes the children of groups one level deeper.
        Directions alternate between vertical and horizontal: for
        ``rankdir=TB`` the levels are ``TB, LR, TB, ...``, for
        ``rankdir=LR`` they are ``LR, TB, LR, ...``, and ``BT`` /
        ``RL`` reverse both orders. Invalid ``rankdir`` values are
        treated as ``TB``.
        """
        depth = 0

        def deepest(fields: Sequence[RecordField], level: int) -> None:
            nonlocal depth
            depth = max(depth, level)
            for field in fields:
                if field.is_group:
                    deepest(field.fields or [], level + 1)

        if len(self.fields) == 1 and self.fields[0].is_group:
            deepest(self.fields[0].fields or [], 0)
        else:
            deepest(self.fields, 0)
        return [
            _direction_for_level(level, rankdir)
            for level in range(depth + 1)
        ]


def _direction_for_level(level: int, rankdir: str) -> str:
    if rankdir not in ("TB", "BT", "LR", "RL"):
        rankdir = "TB"
    horizontal_base = rankdir in ("LR", "RL")
    forward_h = "LR" if rankdir in ("TB", "LR") else "RL"
    forward_v = "TB" if rankdir in ("TB", "LR") else "BT"
    # For a vertical rankdir (TB/BT) levels 0, 2, ... are vertical;
    # for a horizontal rankdir (LR/RL) those levels are horizontal.
    level_is_horizontal = (level % 2 == 1) ^ horizontal_base
    return forward_h if level_is_horizontal else forward_v


# ---------------------------------------------------------------------------
# Graph-level helpers
# ---------------------------------------------------------------------------

def _strip_quotes(value: object) -> str:
    if not isinstance(value, str):
        return "" if value is None else str(value)
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _node_key(name: str) -> str:
    """Normalize a node name for matching against edge endpoints."""
    return _strip_quotes(name)


def _is_html_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= 2
        and value.lstrip().startswith("<")
        and value.rstrip().endswith(">")
        and not value.lstrip().startswith("<>")
    )


def get_node_shape(
    node: pydot.Node,
    extra_defaults: Sequence[dict] = (),
) -> str | None:
    """Return the effective shape of a node.

    A node's own ``shape`` attribute wins; otherwise the ``node``
    default attributes of its parent graph (and any additional default
    dictionaries supplied by the caller for enclosing subgraphs) are
    consulted.
    """
    shape = node.get("shape")
    if shape is not None:
        return _strip_quotes(shape)
    for defaults in reversed(extra_defaults):
        default_shape = defaults.get("shape")
        if default_shape is not None:
            return _strip_quotes(default_shape)
    parent = node.get_parent_graph()
    if parent is not None:
        for defaults in parent.get_node_defaults():
            default_shape = defaults.get("shape")
            if default_shape is not None:
                return _strip_quotes(default_shape)
    return None


def is_record_node(
    node: pydot.Node, extra_defaults: Sequence[dict] = ()
) -> bool:
    """Whether a node is declared with shape ``record``/``Mrecord``."""
    name = node.get_name()
    if name in ("node", "graph", "edge"):
        return False
    return (
        get_node_shape(node, extra_defaults) in RECORD_SHAPES
    )


def get_node_record(
    node: pydot.Node, extra_defaults: Sequence[dict] = ()
) -> RecordLabel | None:
    """Parse the record label of a record node.

    Returns ``None`` for non-record nodes, HTML-like labels and nodes
    without a label.
    """
    if not is_record_node(node, extra_defaults):
        return None
    label = node.get("label")
    if label is None or _is_html_label(label):
        return None
    return RecordLabel.parse(str(label))


def _split_endpoint(endpoint: object) -> tuple[str, str | None]:
    """Split an edge endpoint into ``(node_name, port)``.

    Compass points (``node:n``, ``node:port:ne``) are removed.
    """
    if not isinstance(endpoint, str):
        return ("", None)
    ref = endpoint
    if len(ref) >= 2 and ref.startswith('"'):
        # Quoted node id, possibly followed by ':' port.
        closing = ref.find('"', 1)
        if closing != -1:
            name = _dot_unescape(ref[: closing + 1])
            remainder = ref[closing + 1 :]
            if not remainder:
                return (name, None)
            if remainder[0] == ":":
                parts = remainder[1:].split(":")
            else:
                parts = []
        else:
            parts = [ref]
            name = ref
    else:
        parts = ref.split(":")
        name = parts[0]
        parts = parts[1:]

    if not parts:
        return (name, None)
    # Drop a trailing compass point.
    if parts and _strip_quotes(parts[-1]) in _COMPASS_POINTS:
        parts = parts[:-1]
    if not parts:
        return (name, None)
    port = _dot_unescape(parts[0]) if parts[0].startswith('"') else parts[0]
    port = _strip_quotes(port)
    return (name, port)


def _iter_real_nodes(graph: pydot.Graph):
    for node in graph.get_node_list():
        if node.get_name() in ("node", "graph", "edge"):
            continue
        yield node


def check_graph_records(
    graph: pydot.Dot,
    inherited_defaults: Sequence[dict] = (),
) -> None:
    """Validate record labels and edge port references in ``graph``.

    Subgraphs are checked recursively and inherit the ``node`` default
    attributes of their enclosing graphs.

    Checks are limited to nodes declared as ``record`` or ``Mrecord``;
    ordinary string labels of other shapes are left untouched.

    Raises :class:`RecordSyntaxError`, :class:`DuplicatePortError` or
    :class:`UnknownPortError` on the first problem found. Error
    messages name the offending node and, for label problems, the
    field path.
    """
    records: dict[str, RecordLabel] = {}

    own_defaults = list(graph.get_node_defaults())
    defaults = list(inherited_defaults) + own_defaults

    def collect_nodes(a_graph: pydot.Graph, scope_defaults: list[dict]):
        for node in _iter_real_nodes(a_graph):
            if not is_record_node(node, scope_defaults):
                continue
            raw_name = node.get_name()
            name = _node_key(raw_name)
            record = get_node_record(node, scope_defaults)
            if record is None:
                records.setdefault(
                    name, RecordLabel([RecordField(text=raw_name)])
                )
                continue
            record.check_duplicate_ports(name)
            records.setdefault(name, record)

        for subgraph in a_graph.get_subgraph_list():
            sub_defaults = list(scope_defaults) + list(
                subgraph.get_node_defaults()
            )
            collect_nodes(subgraph, sub_defaults)

    collect_nodes(graph, defaults)

    def check_edges(a_graph: pydot.Graph) -> None:
        for edge in a_graph.get_edge_list():
            for endpoint in (edge.get_source(), edge.get_destination()):
                name, port = _split_endpoint(endpoint)
                if port is None or name not in records:
                    continue
                record = records[name]
                if record.find(port) is None:
                    raise UnknownPortError(
                        node=name,
                        port=port,
                        path=record.port_path(port) or (),
                        edge=str(edge),
                    )
        for subgraph in a_graph.get_subgraph_list():
            check_edges(subgraph)

    check_edges(graph)
