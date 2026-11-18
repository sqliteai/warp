# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
dsml.py — DeepSeek-V4.1's prompt format, and the reader for its replies.

A port of the release's `encoding/encoding.py`, the way `xtml.py` is a port
of K3's `encoding_k3.py`, and checked against that file the same way:
`tests/serve/test_dsml_upstream.py` renders the same conversations through
both and compares the strings, whenever `DS41_DIR` names a release.

What it emits is a list of Segment, not a string. That is the security
boundary this whole family rests on: control tokens go through the
tokenizer's markup entry point and **everything a user, a document or a
tool wrote goes through the other one**. Concatenating the two and encoding
once would let a tool result containing `｜DSML｜` open a tool-call block
with a real control token.

Three things distinguish V4.1's format from V4's, and all three are here
because getting one wrong produces a prompt the model answers plausibly and
wrongly:

  - **the DSML tag names carry a leading space.** `<｜DSML｜ calls>`, not
    `<｜DSML｜tool_calls>`. Only `｜DSML｜` is a control token; the `<`, the
    ` calls`, and the `>` are ordinary text either side of it.
  - **reasoning effort is a number, 1 to 100**, rendered in a `<｜System｜>`
    prefix and only at the start of a thinking conversation. The strings
    "low", "high" and "max" map onto 50, 75 and 100.
  - **a mid-conversation system message** is a turn of its own, and counts
    as a user turn when deciding where the assistant header goes.

There is no `tool` role: tool results are `<tool_result>` blocks inside a
user message, and `<tool_result>` is NOT a control token — it is four
characters of text the model was trained to read.
"""

import copy
import json
from typing import Any, Optional

from .regions import Delta, ToolCall
from .xtml import Segment

# ---- the control tokens, and what is merely text around them -------------
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
SYSTEM = "<｜System｜>"
REMINDER = "<｜latest_reminder｜>"
IMAGE = "<｜deepseek_image｜>"
DSML = "｜DSML｜"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The six task tokens are internal classification tasks — not an OpenAI
# concept, and `build_chat_segments` never emits one on its own. They are in
# the required set anyway: they are part of this format, they are in this
# release's tokenizer, and a container without them is not the release. A
# format that half-resolves is the failure mode this probe exists to
# prevent, so it is all of them or none.
TASKS = {"action": "<｜action｜>", "query": "<｜query｜>",
         "authority": "<｜authority｜>", "domain": "<｜domain｜>",
         "title": "<｜title｜>", "read_url": "<｜read_url｜>"}

MARKERS = (BOS, EOS, USER, ASSISTANT, SYSTEM, REMINDER, IMAGE, DSML,
           THINK_OPEN, THINK_CLOSE) + tuple(TASKS.values())

CALLS_TAG = " calls"
INVOKE_TAG = " invoke"
PARAM_TAG = " parameter"

TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"

EFFORT_TEMPLATE = ("Reasoning Effort: {budget} "
                   "(range 1-100, the higher the value, the more thorough "
                   "the reasoning)\n\n")
EFFORT_NAMES = {"low": 50, "high": 75, "max": 100}
DEFAULT_EFFORT = "high"

RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\nYou MUST strictly adhere to the following "
    "schema to reply:\n{schema}")

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{d}{calls}>" block like the following:

<{d}{calls}>
<{d}{invoke} name="$TOOL_NAME">
<{d}{param} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{d}{param}>
...
</{d}{invoke}>
<{d}{invoke} name="$TOOL_NAME2">
...
</{d}{invoke}>
</{d}{calls}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think}), you MUST output your complete reasoning inside {think}...{endthink} BEFORE any tool calls or final response.

Otherwise, output directly after {endthink} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


class DSMLError(ValueError):
    """A conversation that cannot be rendered in DeepSeek-V4.1's format."""

    def __init__(self, message: str, *, param: Optional[str] = None):
        super().__init__(message)
        self.param = param


def detect(engine: Any) -> dict[int, str]:
    """control-token id -> marker, or raise.

    Every marker must encode to exactly one token in markup mode. One that
    does not is not in the container's specials, which means the format and
    the tokenizer disagree — and a prompt built from it would be markup the
    model reads as prose. See serve/engine.py's marker_ids.
    """
    return engine.marker_ids_for(MARKERS, what="DeepSeek-V4.1's DSML")


def image_prompt(width: int, height: int) -> str:
    """The placeholder an image occupies in the prompt.

    One token, whatever the image's size: the release expands it into a run
    of positions in the model, not in the text. `width` and `height` are
    taken and ignored, because the two other formats in this server need
    them and api.py calls all three the same way.
    """
    del width, height
    return IMAGE


# ---- helpers -------------------------------------------------------------

def _content_text(content: Any) -> str:
    """A message's content as text.

    A list is OpenAI's content-block form. Images have already been replaced
    by their placeholder in build_chat_segments, so anything image-shaped
    that reaches here came without one — which api.py's flow makes
    impossible and a direct caller could still do.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind in ("image_url", "image"):
            raise DSMLError(
                "an image block reached the renderer without a placeholder; "
                "images have to be encoded before the conversation is",
                param="messages")
        else:
            parts.append(f"[Unsupported {kind}]")
    return "\n\n".join(p for p in parts if p != "")


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _text(out: list[Segment], s: str) -> None:
    """Ordinary text. Anything a user, a document or a tool wrote arrives
    here and nowhere else."""
    if s:
        out.append(Segment(s, markup=False))


def _mark(out: list[Segment], s: str) -> None:
    out.append(Segment(s, markup=True))


def _dsml_tag(out: list[Segment], close: bool, name: str,
              tail: str = ">") -> None:
    """`<｜DSML｜ calls>` and friends: one control token with text either
    side of it, never one string."""
    _text(out, "</" if close else "<")
    _mark(out, DSML)
    _text(out, name + tail)


def _split_tool_name(name: str, namespace: Optional[str] = None):
    prefix, sep, bare = name.partition("::")
    if sep:
        if namespace not in (None, prefix):
            raise DSMLError(f"conflicting tool namespaces: {namespace} != {prefix}",
                            param="tools")
        namespace, name = prefix, bare
    if "::" in name:
        raise DSMLError(f"tool name must not contain '::': {name}", param="tools")
    if namespace is not None and "::" in namespace:
        raise DSMLError(f"tool namespace must not contain '::': {namespace}",
                        param="tools")
    return namespace, name


def _tool_name_for_encoding(tool: dict) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    namespace, name = _split_tool_name(tool["name"], namespace)
    return name if namespace is None else f"{namespace}::{name}"


def tools_from_openai(tools: list) -> list[dict]:
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict) or "function" not in tool:
            raise DSMLError("each tool must be {'type':'function','function':{...}}",
                            param="tools")
        fn = dict(tool["function"])
        if tool.get("namespace") is not None:
            fn["namespace"] = tool["namespace"]
        fn["name"] = _tool_name_for_encoding(fn)
        ns = fn.pop("namespace", None)
        if isinstance(ns, dict) and ns.get("description"):
            fn["description"] = ns["description"] + "\n" + (fn.get("description") or "")
        out.append(fn)
    return out


def tool_calls_from_openai(tool_calls: list) -> list[dict]:
    calls = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        ns, name = _split_tool_name(fn.get("name", ""),
                                    tc.get("namespace") or fn.get("namespace"))
        call = {"name": name, "arguments": fn.get("arguments", "")}
        if ns is not None:
            call["namespace"] = ns
        calls.append(call)
    return calls


def render_effort(index: int, thinking: bool, effort) -> str:
    """The numeric budget prefix — thinking mode, first message only."""
    if effort is None:
        effort = DEFAULT_EFFORT
    if isinstance(effort, bool) or not (
            isinstance(effort, int) or effort in EFFORT_NAMES):
        raise DSMLError(
            f"invalid reasoning effort {effort!r}: DeepSeek-V4.1 takes an "
            f"integer in [1, 100] or one of {sorted(EFFORT_NAMES)}",
            param="reasoning_effort")
    if isinstance(effort, str):
        effort = EFFORT_NAMES[effort]
    if not 1 <= effort <= 100:
        raise DSMLError(
            f"reasoning effort {effort} is outside [1, 100]",
            param="reasoning_effort")
    if index == 0 and thinking:
        return EFFORT_TEMPLATE.format(budget=effort)
    return ""


def render_tools(out: list[Segment], tools: list[dict]) -> None:
    """The tool block, with `｜DSML｜` as a control token everywhere it
    appears — including inside the instructions, where the model is being
    shown the markup it should emit."""
    # Formatted with a sentinel where the control token goes, then split on
    # it: `.format` would produce one string and lose the distinction, and
    # the template spells `｜DSML｜` eleven times because it is SHOWING the
    # model the markup it should emit.
    body = TOOLS_TEMPLATE.format(
        d="\x00", calls=CALLS_TAG, invoke=INVOKE_TAG, param=PARAM_TAG,
        think=THINK_OPEN, endthink=THINK_CLOSE,
        tool_schemas="\n".join(_json(t) for t in tools))
    parts = body.split("\x00")
    for i, piece in enumerate(parts):
        if i:
            _mark(out, DSML)
        # <think> and </think> are control tokens too, and the template
        # spells them out four times in its instructions.
        for sub in _split_keep(piece, (THINK_OPEN, THINK_CLOSE)):
            if sub in (THINK_OPEN, THINK_CLOSE):
                _mark(out, sub)
            else:
                _text(out, sub)


def _split_keep(text: str, needles: tuple[str, ...]) -> list[str]:
    """Split on any of `needles`, keeping them as their own pieces."""
    out, i = [], 0
    while i < len(text):
        hit, at = None, len(text)
        for n in needles:
            p = text.find(n, i)
            if p != -1 and p < at:
                at, hit = p, n
        if hit is None:
            out.append(text[i:])
            break
        if at > i:
            out.append(text[i:at])
        out.append(hit)
        i = at + len(hit)
    return out


def encode_arguments(out: list[Segment], tool_call: dict) -> None:
    """Arguments as DSML `parameter` elements.

    `string="true"` means the value is the text between the tags as it
    stands; anything else is JSON. The values are a model's own output on
    the way back in, so they go through the text entry point.
    """
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        for _ in range(2):           # tolerate double-encoded JSON
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    break
            else:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call.get("arguments")}
    first = True
    for k, v in arguments.items():
        if not first:
            _text(out, "\n")
        first = False
        _dsml_tag(out, False, PARAM_TAG, "")
        _text(out, f' name="{k}" string="{"true" if isinstance(v, str) else "false"}">')
        _text(out, v if isinstance(v, str) else _json(v))
        _dsml_tag(out, True, PARAM_TAG)


# ---- message rendering ---------------------------------------------------

def _last_user_index(messages: list[dict]) -> int:
    """Where the assistant header goes. A mid-conversation system message
    counts as a user turn — V4.1's rule, and not V4's."""
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "user" or (role == "system" and idx > 0):
            return idx
    return -1


def merge_tool_messages(messages: list[dict]) -> list[dict]:
    """There is no `tool` role. A tool result becomes a `<tool_result>`
    block in a user message, merged into the preceding one when that is
    already a user turn."""
    merged: list[dict] = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")
        if role == "tool":
            block = {"type": "tool_result",
                     "tool_use_id": msg.get("tool_call_id", ""),
                     "content": msg.get("content", "")}
            if merged and merged[-1].get("role") == "user" and \
                    "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            blocks = msg.get("content_blocks")
            if blocks is None:
                blocks = [{"type": "text",
                           "text": _content_text(msg.get("content"))}]
            if merged and merged[-1].get("role") == "user" and \
                    "content_blocks" in merged[-1] and \
                    merged[-1].get("task") is None:
                merged[-1]["content_blocks"].extend(blocks)
            else:
                msg["content_blocks"] = blocks
                merged.append(msg)
        else:
            merged.append(msg)
    return merged


def sort_tool_results(messages: list[dict]) -> list[dict]:
    """Tool results in the order the assistant asked for them."""
    order: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            order = {}
            for i, tc in enumerate(msg["tool_calls"]):
                tid = tc.get("id") or (tc.get("function") or {}).get("id", "")
                if tid:
                    order[tid] = i
        elif role == "user" and msg.get("content_blocks"):
            blocks = [b for b in msg["content_blocks"]
                      if b.get("type") == "tool_result"]
            if len(blocks) > 1 and order:
                blocks = sorted(blocks,
                                key=lambda b: order.get(b.get("tool_use_id", ""), 0))
                it, new = iter(blocks), []
                for b in msg["content_blocks"]:
                    new.append(next(it) if b.get("type") == "tool_result" else b)
                msg["content_blocks"] = new
    return messages


def _drop_thinking(messages: list[dict]) -> list[dict]:
    """Reasoning before the last user turn is dropped. The model is shown
    what it said, not what it thought while saying it."""
    last = _last_user_index(messages)
    keep = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
    out = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep or idx >= last:
            out.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            out.append(msg)
    return out


def render_message(out: list[Segment], index: int, messages: list[dict],
                   thinking: bool, drop: bool, effort) -> None:
    msg = messages[index]
    last_user = _last_user_index(messages)
    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning = msg.get("reasoning_content")
    if tools:
        tools = tools_from_openai(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai(tool_calls)

    prefix = render_effort(index, thinking, effort)
    if index == 0 and (prefix or role == "system"):
        _mark(out, SYSTEM)
    _text(out, prefix)

    if role == "system":
        if index > 0:
            _mark(out, SYSTEM)          # mid-conversation system message
        _text(out, _content_text(content))
        if tools:
            _text(out, "\n\n")
            render_tools(out, tools)
        if response_format:
            _text(out, "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(
                schema=_json(response_format)))

    elif role == "user":
        _mark(out, USER)
        blocks = msg.get("content_blocks")
        if blocks:
            for i, block in enumerate(blocks):
                if i:
                    _text(out, "\n\n")
                kind = block.get("type")
                if kind == "text":
                    _text(out, block.get("text", ""))
                elif kind == "tool_result":
                    body = block.get("content", "")
                    if isinstance(body, list):
                        parts = []
                        for b in body:
                            parts.append(b.get("text", "")
                                         if b.get("type") == "text"
                                         else f"[Unsupported {b.get('type')}]")
                        body = "\n\n".join(parts)
                    _text(out, TOOL_RESULT_OPEN + str(body) + TOOL_RESULT_CLOSE)
                else:
                    _text(out, f"[Unsupported {kind}]")
        else:
            _text(out, _content_text(content))

    elif role == "latest_reminder":
        _mark(out, REMINDER)
        _text(out, _content_text(content))

    elif role == "tool":
        raise DSMLError("tool messages must be merged into the user turn "
                        "first — see merge_tool_messages", param="messages")

    elif role == "assistant":
        # A task output is one word in a named vocabulary, not a turn, so
        # the assistant that answers a task message has no thinking block.
        prev_task = index > 0 and messages[index - 1].get("task") is not None
        if thinking and not prev_task:
            if not drop or index > last_user:
                _text(out, reasoning or "")
                _mark(out, THINK_CLOSE)
        _text(out, _content_text(content))
        if tool_calls:
            _text(out, "\n\n")
            _dsml_tag(out, False, CALLS_TAG)
            _text(out, "\n")
            for i, tc in enumerate(tool_calls):
                if i:
                    _text(out, "\n")
                _dsml_tag(out, False, INVOKE_TAG, "")
                _text(out, f' name="{_tool_name_for_encoding(tc)}">\n')
                encode_arguments(out, tc)
                _text(out, "\n")
                _dsml_tag(out, True, INVOKE_TAG)
            _text(out, "\n")
            _dsml_tag(out, True, CALLS_TAG)
        _mark(out, EOS)
    else:
        raise DSMLError(f"unknown role {role!r}", param="messages")

    # The transition into the next turn.
    if index + 1 < len(messages) and \
            messages[index + 1].get("role") not in ("assistant", "latest_reminder"):
        return

    task = msg.get("task")
    if task is not None:
        # An internal classification task. The assistant is asked for one
        # word in a named vocabulary rather than for a turn, so the header
        # is the task token and — except for "action" — there is no
        # assistant marker at all.
        if task not in TASKS:
            raise DSMLError(f"unknown task {task!r}: this format has "
                            f"{sorted(TASKS)}", param="messages")
        if task != "action":
            _mark(out, TASKS[task])
        else:
            _mark(out, ASSISTANT)
            _mark(out, THINK_OPEN if thinking else THINK_CLOSE)
            _mark(out, TASKS[task])
    elif role == "user" or (role == "system" and index > 0):
        _mark(out, ASSISTANT)
        if thinking and (not drop or index >= last_user):
            _mark(out, THINK_OPEN)
        else:
            _mark(out, THINK_CLOSE)


def build_chat_segments(messages: list, tools: Optional[list] = None, *,
                        thinking: bool = False,
                        add_generation_prompt: bool = True,
                        image_prompts: Optional[list[str]] = None,
                        thinking_effort=None,
                        tool_choice: Optional[str] = None,
                        response_format: Optional[dict] = None,
                        ) -> list[Segment]:
    """The whole conversation, as segments.

    Signature matches xtml.build_chat_segments and chatfmt's, so api.py
    does not know which format it has.
    """
    del add_generation_prompt, tool_choice
    msgs = [copy.deepcopy(m) for m in messages]
    if tools:
        # Upstream hangs the tool declarations off a system message. A
        # request with tools and no system turn gets an empty one, which is
        # what the release's own examples do.
        for m in msgs:
            if m.get("role") == "system":
                m["tools"] = tools
                break
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    if response_format is not None:
        for m in msgs:
            if m.get("role") == "system":
                m["response_format"] = response_format
                break
        else:
            msgs.insert(0, {"role": "system", "content": "",
                            "response_format": response_format})
    # Always, not only when there are images: it is also what normalizes
    # list content into content_blocks, and what refuses a user string that
    # spells the image token.
    _place_images(msgs, list(image_prompts or []))

    msgs = merge_tool_messages(msgs)
    msgs = sort_tool_results(msgs)

    # Thinking is kept whole when the conversation has tools: the model's
    # own reasoning is what justified the calls it made, and dropping it
    # leaves the results unexplained.
    drop = not any(m.get("tools") for m in msgs)
    if thinking and drop:
        msgs = _drop_thinking(msgs)

    out: list[Segment] = []
    _mark(out, BOS)
    for idx in range(len(msgs)):
        render_message(out, idx, msgs, thinking, drop, thinking_effort)
    return out


def _place_images(messages: list[dict], prompts: list[str]) -> None:
    """Images out, placeholders in — upstream's process_image_messages.

    List content becomes `content_blocks`, each image block becomes a text
    block holding the placeholder, and `content` is set to the blocks' texts
    joined. Doing it in that order matters: render_message reads
    content_blocks when they exist and `content` when they do not, and a
    message that has both has to agree with itself.

    A user string that already contains the placeholder is REFUSED. It would
    encode as ordinary text — the segment it lands in is markup=False, so it
    cannot forge the control token — but it would still make the count of
    placeholders disagree with the count of queued images, and the tower's
    embeddings would land at the wrong positions.
    """
    for msg in messages:
        for field in ("content", "reasoning_content"):
            v = msg.get(field)
            if isinstance(v, str) and IMAGE in v:
                raise DSMLError(
                    f"{field} contains the image token {IMAGE}; images have "
                    f"to be sent as image content blocks", param="messages")
        if isinstance(msg.get("content"), list) and "content_blocks" not in msg:
            msg["content_blocks"] = msg.pop("content")
        blocks = msg.get("content_blocks")
        if not blocks:
            continue
        msg["content_blocks"] = _image_blocks(blocks, prompts)
        if not isinstance(msg.get("content"), str):
            msg["content"] = "\n\n".join(
                b.get("text", "") for b in msg["content_blocks"]
                if isinstance(b, dict) and b.get("type") == "text")


def _image_blocks(blocks: list, prompts: list[str]) -> list:
    out = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        kind = block.get("type")
        if kind in ("image_url", "image"):
            if not prompts:
                raise DSMLError(
                    "an image block has no placeholder: images have to be "
                    "encoded before the conversation is", param="messages")
            out.append({"type": "text", "text": prompts.pop(0)})
        elif kind == "tool_result" and isinstance(block.get("content"), list):
            block = dict(block)
            block["content"] = _image_blocks(block["content"], prompts)
            out.append(block)
        elif kind == "text":
            text = block.get("text") or ""
            if IMAGE in text:
                raise DSMLError(
                    f"a text block contains the image token {IMAGE}",
                    param="messages")
            out.append(block)
        else:
            out.append(block)
    return out


# ---- reading the reply back ---------------------------------------------

class DSMLParser:
    """Incremental reader for one assistant turn.

    Structure is decided by token id, not by what the text spells: a reply
    that writes `｜DSML｜` as prose cannot open a tool-call block, because
    the marker it wrote is not the control token.

    The generation prompt ends with `<think>` or `</think>`, so the parser
    is told which channel it starts in — the opening is in the prompt and
    never in the completion.
    """

    def __init__(self, *, thinking: bool = False,
                 markers: Optional[dict[int, str]] = None):
        self._markers = dict(markers or {})
        self.reasoning = ""
        self.content = ""
        self.tool_calls: list[ToolCall] = []
        self._in_think = thinking
        self._in_calls = False
        self._buf = ""
        self._done = False
        self._call: Optional[ToolCall] = None
        self._arg_key = ""
        self._arg_string = True
        self._arg_text = ""
        self._in_param = False
        # Content that may be the "\n\n<" in front of a tool-call block: held
        # back until the next token says whether a block opens, so a client
        # never streams text the final message drops.
        self._held = ""

    @property
    def finished(self) -> bool:
        return self._done

    def feed_token(self, token_id: int, piece: str) -> Delta:
        delta = Delta()
        if self._done:
            return delta
        marker = self._markers.get(token_id)
        if marker is not None:
            self._marker(marker, delta)
        elif piece:
            self._literal(piece, delta)
        return delta

    def feed(self, text: str) -> Delta:
        """Text-only path, for a caller with no token ids. Markers are found
        by scanning, so a reply that spells one out is read as structure."""
        delta = Delta()
        if self._done or not text:
            return delta
        for piece in _split_keep(text, MARKERS):
            if piece in self._markers.values() or piece in MARKERS:
                self._marker(piece, delta)
            else:
                self._literal(piece, delta)
        return delta

    def finish(self) -> Delta:
        """Flush. A reply cut off by the token limit ends mid-element; its
        text is still delivered, because a truncated answer beats none."""
        delta = Delta()
        if self._buf:
            self._literal("", delta)
        self._release(delta)
        self._flush_call()
        return delta

    def drop_held(self) -> None:
        """The caller cut the content (a stop string): what is held back
        comes after the cut, so it must not be released by finish()."""
        self._held = ""

    def openai_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant"}
        msg["content"] = self.content if self.content else None
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [c.to_openai() for c in self.tool_calls]
        return msg

    # ---- internals ------------------------------------------------------

    def _marker(self, marker: str, delta: Delta) -> None:
        if marker != DSML:
            self._release(delta)
        if marker == THINK_CLOSE:
            self._in_think = False
            return
        if marker == THINK_OPEN:
            self._in_think = True
            return
        if marker == EOS:
            self._done = True
            self._flush_call()
            return
        if marker == DSML:
            # A parameter's value ends at the `</` that opens its closing
            # tag, and that `</` arrives as ordinary text before this
            # marker does. Take it back off, or every string argument comes
            # out with two characters of markup glued to the end of it.
            if self._in_param and self._arg_text.endswith("</"):
                self._arg_text = self._arg_text[:-2]
            # The tag name arrives as ordinary text right after it, so the
            # decision is deferred to _literal.
            self._buf = "\x00"
            return
        # Any other control token inside a reply is not structure this
        # format has — it is the model emitting a turn marker of its own.
        # Ending the turn is the honest reading.
        self._done = True

    def _literal(self, piece: str, delta: Delta) -> None:
        if self._buf.startswith("\x00"):
            self._buf += piece
            self._tag(delta)
            return
        if self._in_calls:
            if self._in_param:
                self._arg_text += piece
            return
        if self._in_think:
            self.reasoning += piece
            delta.reasoning += piece
        else:
            text = self._held + piece
            keep = _lead_in_suffix(text)
            self._held = text[len(text) - keep:]
            out = text[:len(text) - keep]
            self.content += out
            delta.content += out

    def _release(self, delta: Delta) -> None:
        """Send held-back content: whatever followed it was not a block."""
        if self._held:
            self.content += self._held
            delta.content += self._held
            self._held = ""

    def _tag(self, delta: Delta) -> None:
        """A `｜DSML｜` was seen; decide what element this is once its name
        and the closing `>` have arrived."""
        buf = self._buf[1:]
        if ">" not in buf:
            return                                   # still arriving
        head, _, rest = buf.partition(">")
        self._buf = ""
        name = head.strip()
        if name.startswith(CALLS_TAG.strip()):
            if not self._in_calls:
                self._in_calls = True
                # The two newlines before the block are prompt formatting,
                # not content. They are still held back, so they were never
                # streamed; drop them there, and send anything else held.
                if self._held.endswith("\n\n<"):
                    self._held = self._held[:-3]
                elif self._held.endswith("<"):
                    self._held = self._held[:-1]
                self._release(delta)
                if self.content.endswith("\n\n<"):
                    self.content = self.content[:-3]
                elif self.content.endswith("<"):
                    self.content = self.content[:-1]
            else:
                self._in_calls = False
                self._flush_call()
        elif name.startswith(INVOKE_TAG.strip()):
            self._release(delta)
            if "name=" in head:
                self._flush_call()
                # `ns::name` is one namespaced tool. OpenAI has no field for
                # the namespace, and upstream returns the bare name with the
                # namespace beside it, so the name a client sees is bare.
                raw = _attr(head, "name")
                _, bare = raw.partition("::")[0], raw.partition("::")[2]
                self._call = ToolCall(name=bare or raw,
                                      index=len(self.tool_calls))
            else:
                self._flush_call()
        elif name.startswith(PARAM_TAG.strip()):
            if "name=" in head:
                self._arg_key = _attr(head, "name")
                self._arg_string = _attr(head, "string") != "false"
                self._arg_text = ""
                self._in_param = True
            else:
                self._close_param()
        if rest:
            self._literal(rest, delta)

    def _close_param(self) -> None:
        if not self._in_param or self._call is None:
            self._in_param = False
            return
        value: Any = self._arg_text
        if not self._arg_string:
            try:
                value = json.loads(self._arg_text)
            except (TypeError, ValueError):
                pass
        self._call.arguments[self._arg_key] = value
        self._in_param = False
        self._arg_key = ""
        self._arg_text = ""

    def _flush_call(self) -> None:
        self._close_param()
        if self._call is not None:
            self.tool_calls.append(self._call)
            self._call = None


def _lead_in_suffix(text: str) -> int:
    """Length of the end of `text` that could still become the "\n\n<" (or
    the bare "<") a tool-call block is written after."""
    for lead in ("\n\n<", "\n\n", "<", "\n"):
        if text.endswith(lead):
            return len(lead)
    return 0


def _attr(head: str, key: str) -> str:
    """The value of `key="..."` in a tag header. The header is the model's
    own output, so this reads it rather than trusting it: a missing or
    malformed attribute gives the empty string and the call is refused
    downstream, not here."""
    needle = key + '="'
    at = head.find(needle)
    if at < 0:
        return ""
    at += len(needle)
    end = head.find('"', at)
    return head[at:end] if end >= 0 else head[at:]


def parse_reply(text: str, *, thinking: bool = False,
                markers: Optional[dict[int, str]] = None) -> dict:
    """One-shot convenience for tests and for a non-streaming caller."""
    p = DSMLParser(thinking=thinking, markers=markers)
    p.feed(text)
    p.finish()
    return p.openai_message()
