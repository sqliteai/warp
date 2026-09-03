#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""GLM's native tool-call protocol: the XML markers, the rendering, the reader.

`kimitools.py` holds the tool protocol a Kimi container carries in its
tokenizer. GLM-5.3-Flash is the other direction the engine ships: its
tokenizer has no XTML markers and none of Kimi's five tool tokens, but its
`specials.json` does carry an XML tool protocol of its own — `<tool_call>`,
`<arg_key>`, `<arg_value>` — that lives in `chat_template.jinja` and that the
declarative `chat.json` says nothing about. Four prefix/suffix strings cannot
express a tool declaration, an argument list, or a result turn, so those
turns come from here, exactly as the tool turns of a Kimi container come
from `kimitools`.

The split by subject is the same one `kimitools` makes:

- **Whether a container can do tools at all** is decided by `chatfmt.py`
  and this module's `detect()`, which asks whether *every* GLM marker is a
  single token in the vocabulary. GLM's are; a container that half-carries
  them carries a different protocol, and half of one renders as prose.
- **How a tool call is spelled** — the grammar — lives here, and a request
  this protocol cannot spell raises `GlmToolError`, mapped to a 400 by
  `api.py` the same way `KimiToolError` is.

The grammar is not Kimi's, and the differences are exactly what the name
exists to keep honest:

- **A call is flat XML, not a section.** Kimi nests calls inside
  `<|tool_calls_section_begin|>...<|tool_calls_section_end|>` and spells the
  function's name inside the id. GLM writes
  `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`
  with the name following the opening tag directly and no id at all. The
  arguments are key/value pairs, not one JSON block, so the reply reader
  returns a `ToolCall` whose `arguments` dict carries one entry per pair.
- **A result is an `<|observation|>` turn wrapping one or more**
  **`<tool_response>…content…</tool_response>` blocks.** Kimi names the turn
  for the tool and prefixes it `## Return of <id>`; GLM opens an
  `<|observation|>` turn and repeats the response block per result. The
  template groups consecutive tool results under a single `<|observation|>`;
  `chatfmt` knows which message starts a run and emits the opener for it,
  and this module renders the individual blocks that make it up.
- **The declaration is an instruction turn, not a token literal.** GLM's
  `<# Tools #>` turn is system text with the signatures as JSON inside
  `<tools>…</tools>`; only the `<|system|>` opener is a control token. The
  signatures are caller content, so they go out as a plain segment, in the
  same way the Kimi declaration ships its JSON payload plain.

The module imports nothing from `chatfmt`, so `chatfmt` can import it — the
dependency stays one-way, as it does for `kimitools`.
"""

import json
from typing import Any, Optional

from .regions import Delta, ToolCall
from .xtml import Segment

# In protocol order. All nine or none: a container carrying only some of
# them carries a different protocol, and half of one renders as prose. The
# first six are the reply grammar a model emits; the last three are what a
# `tool` result turn is *made of*, so they must be single tokens too or a
# rendered result would show the model its own turn structure as text.
MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<|observation|>",
)

(_TOOL_CALL, _TOOL_CALL_END,
 _ARG_KEY, _ARG_KEY_END,
 _ARG_VALUE, _ARG_VALUE_END,
 _TOOL_RESPONSE, _TOOL_RESPONSE_END,
 OBSERVATION) = MARKERS


class GlmToolError(ValueError):
    """A tool call this protocol cannot spell.

    `param` names the request field at fault, so the 400 points at
    `messages[3].tool_calls[0]` rather than at `messages`.
    """

    def __init__(self, message: str, *, param: Optional[str] = None):
        super().__init__(message)
        self.param = param


# The `<# Tools #>` system text that follows the `<|system|>` opener. Only
# the opener is markup; `<tools>` and the XML example are the template's own
# instruction prose, sent as plain text the way the template's writer wrote
# it — and those tags (`<tools>`, `{...}`) are not control tokens, so sending
# them as markup would not resolve and would show as junk.
_DECL_HEADER = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML "
    "tags:\n"
    "<tools>"
)
_DECL_FOOTER = (
    "</tools>\n\n"
    "For each function call, output the function name and arguments within "
    "the following XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
    "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
    "<arg_value>{arg-value-2}</arg_value>...</tool_call>"
)


def detect(engine: Any) -> dict[int, str]:
    """{token id: marker} if this container carries the whole protocol.

    Defensive by construction, like `kimitools.detect`: a marker that is not
    a *single* token in this vocabulary is not markup here, it is text, and
    rendering it would show the model its own structure as prose. One missing
    marker disqualifies the set rather than degrading it.
    """
    found: list[tuple[int, str]] = []
    for text in MARKERS:
        got = engine.tokenize(text, markup=True)
        if len(got) != 1:
            return {}
        found.append((got[0], text))
    return dict(found)


def declaration(tools: list[dict]) -> list[Segment]:
    """The `<|system|>` turn that declares the tools as JSON.

    The signatures are the caller's, so each goes out as a plain segment —
    the same boundary `chatfmt` keeps for message content. The instruction
    prose around them is the template's, hardcoded here the way the Kimi
    declaration hardcodes its `tool_declare` opener.
    """
    out = [Segment("<|system|>", markup=True), Segment(_DECL_HEADER)]
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else tool
        out.append(Segment("\n" + json.dumps(fn, ensure_ascii=False) + "\n"))
    out.append(Segment(_DECL_FOOTER))
    return out


def call_section(tool_calls: list[dict], index: int) -> list[Segment]:
    """The flat `<tool_call>…</tool_call>` list of one assistant turn.

    GLM nests calls in no section and attaches no id, so each call is the
    name, then one `<arg_key>`/`<arg_value>` pair per argument, directly.
    The `<tool_call>`, `<arg_key>` and `<arg_value>` markers are markup;
    the name and the argument values are the caller's and go out plain.
    `arguments` may be a JSON string — the OpenAI wire shape — or a dict.
    """
    out: list[Segment] = []
    for j, call in enumerate(tool_calls):
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            raise GlmToolError(
                f"messages[{index}].tool_calls[{j}] has no function name",
                param=f"messages[{index}].tool_calls[{j}].function.name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                raise GlmToolError(
                    f"messages[{index}].tool_calls[{j}] carries JSON "
                    f"arguments this protocol cannot split into "
                    f"<arg_key>/<arg_value> pairs",
                    param=f"messages[{index}].tool_calls[{j}]"
                          f".function.arguments") from None
        if not isinstance(arguments, dict):
            raise GlmToolError(
                f"messages[{index}].tool_calls[{j}] arguments must be an "
                "object",
                param=f"messages[{index}].tool_calls[{j}].function.arguments")
        out.append(Segment(_TOOL_CALL, markup=True))
        out.append(Segment(name))
        for key, value in arguments.items():
            out.append(Segment(_ARG_KEY, markup=True))
            out.append(Segment(str(key)))
            out.append(Segment(_ARG_KEY_END, markup=True))
            out.append(Segment(_ARG_VALUE, markup=True))
            out.append(Segment(value if isinstance(value, str)
                              else json.dumps(value, ensure_ascii=False)))
            out.append(Segment(_ARG_VALUE_END, markup=True))
        out.append(Segment(_TOOL_CALL_END, markup=True))
    return out


def tool_response(content_segments: list[Segment]) -> list[Segment]:
    """One `<tool_response>…content…</tool_response>` block.

    `content_segments` is the already-rendered result content — this file
    decides only the block around it, and `chatfmt` decides whether a run of
    results opens with `<|observation|>`.
    """
    return [Segment(_TOOL_RESPONSE, markup=True),
            *content_segments,
            Segment(_TOOL_RESPONSE_END, markup=True)]


class ToolParser:
    """The reply side: which markers mean what, and where text goes.

    Owned by the reply reader rather than mixed into it — `feed_marker` and
    `feed_text` each answer "did I consume this", so the reader keeps its own
    rules for everything else, the reasoning channel included. A reply the
    model writes for this grammar looks like:

        <tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome
        </arg_value></tool_call><tool_call>...</tool_call>

    Read back as one `ToolCall` per `<tool_call>`, its `name` from the text
    that follows the opening tag and its `arguments` dict from the pairs.
    There is no id in the grammar, so `to_openai()` names each call by
    position, exactly as it does for a Kimi call the model failed to number.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self._state = "content"
        self._key = ""
        self._value = ""
        self._current: Optional[ToolCall] = None

    @property
    def in_structure(self) -> bool:
        """Inside a call, where ordinary text is not the reply."""
        return self._state != "content"

    def feed_marker(self, marker: str, delta: Delta) -> bool:
        if marker == _TOOL_CALL:
            self._current = ToolCall(name="", index=len(self.calls))
            self.calls.append(self._current)
            self._state = "name"
        elif marker == _ARG_KEY:
            self._key = ""
            self._state = "key"
            if self._current is not None:
                # The name is complete once the first argument begins;
                # this is what announces the call to a streaming client.
                delta.tool_calls.append(self._current.index)
        elif marker == _ARG_KEY_END:
            self._state = "between"
        elif marker == _ARG_VALUE:
            self._value = ""
            self._state = "value"
        elif marker == _ARG_VALUE_END:
            self._finish_arg()
            self._state = "between"
        elif marker == _TOOL_CALL_END:
            self._finish_call()
            self._state = "content"
        else:
            return False
        return True

    def feed_text(self, piece: str, delta: Delta) -> bool:
        if self._state == "name" and self._current is not None:
            self._current.name += piece
        elif self._state == "key":
            self._key += piece
        elif self._state == "value":
            self._value += piece
            if self._current is not None:
                delta.tool_calls.append(self._current.index)
        else:
            return False
        return True

    def finish(self) -> None:
        """Flush a call the stream ended in the middle of, and its last
        argument's value."""
        if self._current is not None:
            self._finish_arg()

    def _finish_arg(self) -> None:
        if self._current is not None and self._key:
            self._current.arguments[self._key] = self._value
        self._key = ""
        self._value = ""

    def _finish_call(self) -> None:
        self._finish_arg()
        self._current = None