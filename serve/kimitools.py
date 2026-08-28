#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Kimi's native tool-call protocol: the markers, the rendering, the reader.

`xtml.py` holds K3's format and `chatfmt.py` holds what four
prefix/suffix strings can say. This holds the third thing, and it is third
because it is neither: a tool protocol carried in a container's *tokenizer*
that the container's own `chat.json` says nothing about.

Kimi-Linear is the case that forces the split. Its release declares five
tool-call control tokens and ships **no chat template at all** — the
vocabulary without the grammar. The grammar is Kimi K2's, K2 publishes it,
and `tests/serve/test_chatfmt_upstream.py` diffs this module against that
template the way `test_xtml.TestAgainstUpstream` diffs `xtml.py` against
`encoding_k3.py`. It found the one place this was wrong.

The split is by *subject*, which is what keeps it honest:

- **Whether a container can do tools at all** is a fact about `chat.json`
  and the tokenizer, so `chatfmt.py` decides it and refuses with its own
  `ChatFormatError`. Four strings cannot express a tool declaration.
- **How a tool call is spelled** is a fact about this protocol, so it lives
  here and a malformed one raises `KimiToolError`. Same shape as
  `xtml.XTMLError`: a format module owns its errors, and `api.py` maps each
  to a 400.

That also keeps the dependency one-way. This module imports nothing from
`chatfmt`, so `chatfmt` can import it.

The protocol, as the template renders it:

    <|im_system|>tool_declare<|im_middle|>[{...}]<|im_end|>   declaration
    ...<|tool_calls_section_begin|>                           assistant turn
       <|tool_call_begin|>ID<|tool_call_argument_begin|>ARGS<|tool_call_end|>
       ...
       <|tool_calls_section_end|>
    <|im_system|>NAME<|im_middle|>## Return of ID\ncontent<|im_end|>  result

Two details are easy to get wrong and both were:

  - **The result turn is named for the tool**, `name or "tool"`, never
    `system`. The template says `message.get('name') or message['role']`.
  - **The id, alone, follows `<|tool_call_begin|>`** — not the function
    name. K2 encodes the name inside the id (`functions.NAME:INDEX`), which
    is why the reply reader below parses it back out of there.
"""

import json
from typing import Any, Optional

from .regions import Delta, ToolCall
from .xtml import Segment

# In protocol order. All five or none: a container carrying some of them
# carries a different protocol, and half of one renders as prose.
MARKERS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_argument_begin|>",
    "<|tool_call_end|>",
)

_SECTION_BEGIN, _SECTION_END, _CALL_BEGIN, _ARG_BEGIN, _CALL_END = MARKERS

_ID_PREFIX = "functions."


class KimiToolError(ValueError):
    """A tool call this protocol cannot spell.

    `param` names the request field at fault, so the 400 points at
    `messages[3].tool_calls[0].id` rather than at `messages`.
    """

    def __init__(self, message: str, *, param: Optional[str] = None):
        super().__init__(message)
        self.param = param


def detect(engine: Any) -> dict[int, str]:
    """{token id: marker} if this container carries the whole protocol.

    Defensive by construction: a marker that is not a *single* token in this
    vocabulary is not markup here, it is text, and rendering it would show
    the model its own structure as prose. One missing marker disqualifies
    the set rather than degrading it, because there is no half of this
    protocol that means anything.
    """
    found: list[tuple[int, str]] = []
    for text in MARKERS:
        got = engine.tokenize(text, markup=True)
        if len(got) != 1:
            return {}
        found.append((got[0], text))
    return dict(found)


def rename_turn(opening: str, who: str) -> str:
    """`<|im_system|>system<|im_middle|>` -> `<|im_system|>{who}<|im_middle|>`.

    A Kimi turn opener is three parts: a role token, a free-text name, and a
    separator. `chat.json` spells the whole thing out because for an
    ordinary turn the name never varies; a tool result is the one turn whose
    name comes from the message.

    An opener that is not that shape gets the name appended to it instead of
    a silent substitution, so a format this does not understand renders
    visibly wrong rather than plausibly wrong.
    """
    i = opening.rfind("<|")
    if i <= 0 or not opening.endswith("|>"):
        return opening + who
    j = opening.find("|>")
    if j < 0 or j + 2 > i:
        return opening + who
    return opening[:j + 2] + who + opening[i:]


def declaration(tools: list[dict]) -> list[Segment]:
    """The `tool_declare` system turn carrying the tools as compact JSON.

    The markers are separate segments from the payload so that a
    caller-controlled string cannot be tokenized as model markup — the same
    boundary `chatfmt` keeps for message content.
    """
    return [
        Segment("<|im_system|>tool_declare<|im_middle|>", markup=True),
        Segment(json.dumps(tools, separators=(",", ":"))),
        Segment("<|im_end|>", markup=True),
    ]


def call_section(tool_calls: list[dict], index: int) -> list[Segment]:
    """One assistant turn's calls, all inside a single section."""
    out = [Segment(_SECTION_BEGIN, markup=True)]
    for j, call in enumerate(tool_calls):
        call_id = call.get("id")
        if not call_id:
            raise KimiToolError(
                f"messages[{index}].tool_calls[{j}] has no id",
                param=f"messages[{index}].tool_calls[{j}].id")
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        out += [Segment(_CALL_BEGIN, markup=True),
                Segment(str(call_id)),
                Segment(_ARG_BEGIN, markup=True),
                Segment(arguments),
                Segment(_CALL_END, markup=True)]
    out.append(Segment(_SECTION_END, markup=True))
    return out


def result_header(message: dict, index: int, system_open: str) -> list[Segment]:
    """The opener of a tool-result turn, and the `## Return of` line.

    `system_open` is the container's own system prefix: the shape of the
    turn stays chat.json's business, and only the name inside it comes from
    the message.
    """
    tool_call_id = message.get("tool_call_id")
    if not tool_call_id:
        raise KimiToolError(
            f"messages[{index}] is a tool result without tool_call_id",
            param=f"messages[{index}].tool_call_id")
    who = message.get("name") or "tool"
    return [Segment(rename_turn(system_open, who), markup=True),
            Segment(f"## Return of {tool_call_id}\n")]


class ToolParser:
    """The reply side: which markers mean what, and where text goes.

    Owned by the reply reader rather than mixed into it — `feed_marker` and
    `feed_text` each answer "did I consume this", so the reader keeps its
    own rules for everything else, the reasoning channel included.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self._state = "content"
        self._header = ""
        self._arguments = ""
        self._current: Optional[ToolCall] = None

    @property
    def in_structure(self) -> bool:
        """Inside the tool section, where ordinary text is not the reply."""
        return self._state != "content"

    def feed_marker(self, marker: str, delta: Delta) -> bool:
        if marker == _SECTION_BEGIN:
            self._state = "section"
        elif marker == _CALL_BEGIN:
            self._header = ""
            self._arguments = ""
            self._current = None
            self._state = "header"
        elif marker == _ARG_BEGIN:
            if self._state == "header":
                self._current = self._parse_header()
                self.calls.append(self._current)
                self._state = "arguments"
        elif marker == _CALL_END:
            if self._state == "header":
                self._current = self._parse_header()
                self.calls.append(self._current)
            if self._current is not None:
                self._current.json_block = self._arguments
            self._current = None
            self._state = "section"
        elif marker == _SECTION_END:
            if self._state == "header":
                self._current = self._parse_header()
                self.calls.append(self._current)
            if self._current is not None:
                self._current.json_block = self._arguments
            self._current = None
            self._state = "content"
        else:
            return False
        return True

    def feed_text(self, piece: str, delta: Delta) -> bool:
        if self._state == "header":
            self._header += piece
        elif self._state == "arguments":
            self._arguments += piece
            if self._current is not None:
                self._current.json_block = self._arguments
                delta.tool_calls.append(self._current.index)
        elif self._state == "section":
            pass          # stray text between structures is not the reply
        else:
            return False
        return True

    def finish(self) -> None:
        """Flush a call whose arguments the stream ended in the middle of."""
        if self._state == "header":
            self._current = self._parse_header()
            self.calls.append(self._current)
        if self._current is not None:
            self._current.json_block = self._arguments

    def _parse_header(self) -> ToolCall:
        """`functions.NAME:INDEX` -> the name and the index it carries.

        K2 puts the function's name in the id rather than beside it, so this
        is where a reply's calls get their names back. A header that does
        not carry an index is numbered by position, which is what an
        OpenAI-shaped reply needs and what the id would have said anyway.
        """
        raw = self._header.strip()
        if raw.startswith(_ID_PREFIX):
            raw = raw[len(_ID_PREFIX):]
        name = raw
        index = len(self.calls)
        if ":" in raw:
            maybe_name, maybe_index = raw.rsplit(":", 1)
            if maybe_name:
                name = maybe_name
            try:
                index = int(maybe_index)
            except ValueError:
                pass
        return ToolCall(name=name, index=index)
