# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
api.py — turning an OpenAI request into a prompt, and back again.

Everything between "a client sent JSON" and "the engine has token ids",
kept out of server.py so it can be tested without a socket. The HTTP layer
handles sockets, auth and streaming; this handles meaning.

The K3-specific mappings live here, and they are the interesting part: an
OpenAI request has fields K3 has no notion of (reasoning_effort) and K3 has
concepts OpenAI never named (the think channel, XTML tool results). Where
the two do not line up, this module says so in an error rather than picking
something plausible.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from . import chatfmt, glmtools, kimitools, xtml
from .engine import Engine
from .regions import RegionParser


class APIError(Exception):
    """An error in the shape OpenAI clients parse."""

    def __init__(self, message: str, *, status: int = 400,
                 type: str = "invalid_request_error",
                 param: Optional[str] = None,
                 code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.param = param
        self.code = code

    def to_json(self) -> dict:
        return {"error": {"message": self.message, "type": self.type,
                          "param": self.param, "code": self.code}}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ---- request validation --------------------------------------------------


def _require(body: dict, key: str) -> Any:
    if key not in body:
        raise APIError(f"'{key}' is a required property", param=key)
    return body[key]


def _number(body: dict, key: str, default, lo, hi):
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(f"'{key}' must be a number", param=key)
    if not (lo <= value <= hi):
        raise APIError(f"'{key}' must be between {lo} and {hi}", param=key)
    return value


def validate_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise APIError("'messages' must be a non-empty array", param="messages")
    roles = {"system", "user", "assistant", "tool", "developer"}
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise APIError(f"messages[{i}] must be an object",
                           param=f"messages[{i}]")
        role = m.get("role")
        if role not in roles:
            raise APIError(
                f"messages[{i}].role must be one of {sorted(roles)}",
                param=f"messages[{i}].role")
        content = m.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise APIError(
                f"messages[{i}].content must be a string, an array, or null",
                param=f"messages[{i}].content")
        if isinstance(content, list):
            for j, part in enumerate(content):
                param = f"messages[{i}].content[{j}]"
                if not isinstance(part, dict):
                    raise APIError(f"{param} must be an object", param=param)
                kind = part.get("type")
                if not isinstance(kind, str) or not kind:
                    raise APIError(f"{param}.type must be a non-empty string",
                                   param=f"{param}.type")
                if kind not in ("image", "image_url") and not isinstance(
                        part.get("text"), str):
                    raise APIError(f"{param}.text must be a string",
                                   param=f"{param}.text")
        calls = m.get("tool_calls")
        if calls is not None:
            param = f"messages[{i}].tool_calls"
            if role != "assistant":
                raise APIError(f"{param} is only valid for assistant messages",
                               param=param)
            if not isinstance(calls, list):
                raise APIError(f"{param} must be an array", param=param)
            for j, call in enumerate(calls):
                cp = f"{param}[{j}]"
                if not isinstance(call, dict):
                    raise APIError(f"{cp} must be an object", param=cp)
                fn = call.get("function", call)
                if not isinstance(fn, dict):
                    raise APIError(f"{cp}.function must be an object",
                                   param=f"{cp}.function")
                if not isinstance(fn.get("name"), str) or not fn["name"]:
                    raise APIError(
                        f"{cp}.function.name must be a non-empty string",
                        param=f"{cp}.function.name")
                arguments = fn.get("arguments")
                if (arguments is not None and
                        not isinstance(arguments, (dict, str))):
                    raise APIError(
                        f"{cp}.function.arguments must be an object or JSON string",
                        param=f"{cp}.function.arguments")
        m = dict(m)
        if role == "developer":
            # OpenAI's newer name for a system turn. K3 has one system role,
            # so fold it in rather than inventing an XTML attribute for it.
            m["role"] = "system"
        out.append(m)
    return out


# ---- reasoning effort ----------------------------------------------------

# K3's encoder accepts low / high / max, and nothing else — see xtml.py for
# why "medium" is documented by the model and refused by its own code. The
# OpenAI field is reasoning_effort, so both spellings are accepted.
_EFFORT_OFF = {"none", "minimal", "off"}


def resolve_thinking(body: dict, default_thinking: bool) -> tuple[bool, Optional[str]]:
    """(thinking, thinking_effort) from the request.

    Returns the think channel on or off, and how hard to push it. An
    unsupported value is an error rather than a silent substitution: a
    server that quietly turns `medium` into `high` is a server that reports
    a different amount of reasoning than it did.
    """
    effort = body.get("reasoning_effort")
    if effort is None:
        effort = body.get("thinking_effort")
    if effort is None:
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")

    thinking = body.get("thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise APIError("'thinking' must be a boolean", param="thinking")

    if effort is None:
        return (default_thinking if thinking is None else thinking), None

    if not isinstance(effort, str):
        raise APIError("'reasoning_effort' must be a string",
                       param="reasoning_effort")
    low = effort.lower()
    if low in _EFFORT_OFF:
        return False, None
    if low not in xtml._VALID_THINKING_EFFORTS:
        raise APIError(
            f"reasoning_effort={effort!r} is not supported by this model. "
            f"Kimi K3 accepts {sorted(xtml._VALID_THINKING_EFFORTS)}, or "
            f"{sorted(_EFFORT_OFF)} to answer without reasoning. "
            f"(K3's own encoder documents 'medium' and then rejects it.)",
            param="reasoning_effort")
    if thinking is False:
        raise APIError("'thinking' is false but 'reasoning_effort' asks for "
                       "reasoning — pick one", param="reasoning_effort")
    return True, low


# ---- images --------------------------------------------------------------

_DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+)?(?P<b64>;base64)?,",
                       re.IGNORECASE)


def _iter_image_refs(messages: list[dict]):
    """Every image in the conversation, in the order K3 will read them."""
    for mi, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for pi, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind not in ("image", "image_url"):
                continue
            ref = part.get(kind)
            if isinstance(ref, dict):
                ref = ref.get("url") or ref.get("path") or ref.get("data")
            if not isinstance(ref, str) or not ref:
                raise APIError(
                    f"messages[{mi}].content[{pi}] is an image with no url",
                    param=f"messages[{mi}].content[{pi}]")
            yield ref


def resolve_image(ref: str, tmpdir: str, *,
                  allow_local_files: bool) -> tuple[str, bool]:
    """(path the engine can open, whether we created it).

    Handled: `data:` URIs, `file://` URLs and plain filesystem paths.

    Not handled: http and https. Fetching those would make the server issue
    outbound requests to addresses its clients choose, which is a
    server-side request forgery in every deployment where the server can
    reach more of the network than the client can. A client that wants a
    remote image can fetch it and send the bytes.

    The second element says whether the caller owns the file and must
    delete it: a decoded data: URL is a temp file, and a long-running
    server that keeps one per image request fills its disk.
    """
    if _DATA_URL.match(ref):
        header, _, payload = ref.partition(",")
        if ";base64" not in header.lower():
            raise APIError("only base64 data: URLs are supported",
                           param="image_url")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise APIError(f"image_url is not valid base64: {e}",
                           param="image_url")
        if not raw:
            raise APIError("image_url decoded to no bytes", param="image_url")
        fd, path = tempfile.mkstemp(dir=tmpdir, suffix=".img")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path, True

    lowered = ref.lower()
    if lowered.startswith(("http://", "https://")):
        raise APIError(
            "remote image URLs are not fetched by this server: it would "
            "make the server issue requests to addresses its clients pick. "
            "Send the image as a base64 data: URL instead.",
            param="image_url")

    if lowered.startswith("file://"):
        ref = ref[7:]
    if not allow_local_files:
        raise APIError(
            "local image paths are disabled; start the server with "
            "--allow-local-images to read images off its filesystem, or "
            "send a base64 data: URL", param="image_url")
    path = Path(ref).expanduser()
    if not path.is_file():
        raise APIError(f"image not found: {ref}", param="image_url")
    return str(path), False


# ---- prompt assembly -----------------------------------------------------


class Prompt:
    """A request rendered down to what the engine takes."""

    def __init__(self, tokens: list[int], *, thinking: bool,
                 n_images: int = 0):
        self.tokens = tokens
        self.thinking = thinking
        self.n_images = n_images


def build_prompt(engine: Engine, body: dict, *, default_thinking: bool,
                 allow_local_images: bool, tmpdir: str,
                 fmt: Any = xtml) -> Prompt:
    """Messages in, token ids out.

    `fmt` is the chat format: the xtml module for a K3 container, or a
    chatfmt.ChatFormat built from the container's own chat.json. Both
    expose build_chat_segments and image_prompt with the same signatures,
    so nothing below here knows which one it has.

    The order here is not arbitrary. Images are encoded *first*, because
    each one contributes a media block to the text and the engine's queue
    has to line up with the placeholders in it; then the conversation is
    rendered; then the segments are tokenized one at a time; then the
    placeholders are expanded into the runs of positions the tower needs.
    Skipping the last step would show the model one slot per image and
    drop the rest of the embeddings on the floor.
    """
    messages = validate_messages(_require(body, "messages"))
    thinking, effort = resolve_thinking(body, default_thinking)

    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise APIError("'tools' must be an array", param="tools")

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        # {"type":"function","function":{"name":...}} — K3 has no way to
        # name one tool, so the closest honest rendering is "you must call
        # something", plus the name in a system note the model can read.
        fn = tool_choice.get("function") or {}
        named = fn.get("name")
        if named:
            messages = messages + [{
                "role": "system",
                "content": f"You must call the tool `{named}` in your next "
                           f"message, and no other."}]
        tool_choice = "required"
    elif tool_choice is not None and not isinstance(tool_choice, str):
        raise APIError("'tool_choice' must be a string or an object",
                       param="tool_choice")

    response_format = body.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise APIError("'response_format' must be an object",
                       param="response_format")

    image_prompts: list[str] = []
    engine.image_clear()
    n_images = 0
    scratch: list[str] = []
    try:
        for ref in _iter_image_refs(messages):
            if not engine.vision:
                raise APIError(
                    "this server was started without the vision tower; "
                    "restart it with --vision to accept images", status=400,
                    code="vision_disabled")
            path, owned = resolve_image(ref, tmpdir,
                                        allow_local_files=allow_local_images)
            if owned:
                scratch.append(path)
            # Dimensions before add: both read the file, and add is where a
            # corrupt image is reported, so failing there leaves nothing
            # half-queued.
            w, h = engine.image_dimensions(path)
            engine.image_add(path)
            image_prompts.append(fmt.image_prompt(w, h))
            n_images += 1
    finally:
        # waste_image_add encodes on the spot rather than holding the path
        # (see waste.h), so the bytes are in the engine and the file is
        # ours to drop — including on the error path, where otherwise a
        # client sending broken images fills the disk one request at a time.
        for path in scratch:
            try:
                os.unlink(path)
            except OSError:
                pass

    kwargs: dict[str, Any] = {}
    if effort is not None:
        kwargs["thinking_effort"] = effort
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        segments = fmt.build_chat_segments(
            messages, tools, thinking=thinking, add_generation_prompt=True,
            image_prompts=image_prompts or None, **kwargs)
    except chatfmt.ChatFormatError as e:
        raise APIError(str(e), param=e.param or "messages")
    except xtml.XTMLError as e:
        raise APIError(str(e), param="messages")
    except kimitools.KimiToolError as e:
        raise APIError(str(e), param=e.param or "messages")
    except glmtools.GlmToolError as e:
        raise APIError(str(e), param=e.param or "messages")

    tokens = engine.tokenize_segments(segments)
    if n_images:
        tokens = engine.image_expand(tokens)

    ctx_max = engine.model_info()["ctx_max"]
    if ctx_max and len(tokens) >= ctx_max:
        raise APIError(
            f"prompt is {len(tokens)} tokens, and this model's context is "
            f"{ctx_max}", status=400, code="context_length_exceeded",
            param="messages")

    return Prompt(tokens, thinking=thinking, n_images=n_images)


# ---- sampling ------------------------------------------------------------


def generation_options(body: dict, *, default_max_tokens: int,
                       ctx_max: int, prompt_len: int) -> dict:
    """OpenAI sampling fields, mapped onto waste_gen_params."""
    if body.get("n") not in (None, 1):
        raise APIError("'n' must be 1: this engine generates one completion "
                       "per request", param="n")

    temperature = _number(body, "temperature", 0.0, 0.0, 2.0)
    top_p = _number(body, "top_p", None, 0.0, 1.0)
    top_k = body.get("top_k")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise APIError("'top_k' must be a non-negative integer",
                           param="top_k")

    # max_completion_tokens is the current spelling; max_tokens the old one.
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if limit is None:
        limit = default_max_tokens
    elif isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise APIError("'max_tokens' must be a positive integer",
                       param="max_tokens")
    if ctx_max:
        room = max(1, ctx_max - prompt_len)
        limit = min(limit, room)

    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool)
                             or not isinstance(seed, int)):
        raise APIError("'seed' must be an integer", param="seed")

    return {"temperature": float(temperature),
            "top_p": None if top_p is None else float(top_p),
            "top_k": top_k,
            "seed": None if seed is None else (seed & 0xFFFFFFFFFFFFFFFF),
            "max_tokens": int(limit)}


def stop_strings(body: dict) -> list[str]:
    stop = body.get("stop")
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        return list(stop)
    raise APIError("'stop' must be a string or an array of strings",
                   param="stop")


# ---- responses -----------------------------------------------------------


def finish_reason(parser: RegionParser, *, hit_limit: bool,
                  stopped: bool) -> str:
    if parser.tool_calls:
        return "tool_calls"
    if stopped:
        return "stop"
    if hit_limit:
        return "length"
    return "stop"


def chat_completion(parser: RegionParser, *, model: str, request_id: str,
                    created: int, reason: str, usage: dict,
                    extra: Optional[dict] = None) -> dict:
    message = parser.openai_message()
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None,
                     "finish_reason": reason}],
        "usage": usage,
    }
    if extra:
        body["waste"] = extra
    return body


def chunk(request_id: str, created: int, model: str, delta: dict,
          reason: Optional[str] = None) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "logprobs": None,
                     "finish_reason": reason}],
    }


def usage_block(prompt_tokens: int, completion_tokens: int,
                reasoning_tokens: int = 0) -> dict:
    u = {"prompt_tokens": prompt_tokens,
         "completion_tokens": completion_tokens,
         "total_tokens": prompt_tokens + completion_tokens}
    if reasoning_tokens:
        u["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return u


def engine_extra(stats: dict, *, ms: float) -> dict:
    """Engine counters, alongside the OpenAI body.

    An expert-streaming engine's interesting number is not tokens per
    second, it is how often a token needed the disk. A host that wants to
    show that has nowhere in the OpenAI schema to read it from.
    """
    accesses = stats["experts_hit"] + stats["experts_missed"]
    return {
        "tokens_generated": stats["tokens_generated"],
        "experts_hit": stats["experts_hit"],
        "experts_missed": stats["experts_missed"],
        "expert_hit_rate": (stats["experts_hit"] / accesses) if accesses else None,
        "bytes_read": stats["bytes_read"],
        "direct_io": bool(stats["direct_io"]),
        "ms": round(ms, 1),
    }


def model_object(model_id: str, created: int, info: Optional[dict] = None) -> dict:
    obj = {"id": model_id, "object": "model", "created": created,
           "owned_by": "waste"}
    if info:
        obj["waste"] = info
    return obj


def sse(payload: Any) -> bytes:
    """One Server-Sent Event. `[DONE]` is a literal, not JSON."""
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def now() -> int:
    return int(time.time())
