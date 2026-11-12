# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
fake_engine.py — an engine that says what the test told it to.

The HTTP layer has to be testable without a model. Not because a real one
is slow — the synthetic container in test_engine.py is milliseconds — but
because its weights are noise, so it cannot be made to emit a tool call, a
reasoning channel, or any other reply a test wants to assert about.

So this replays a script. It implements the surface serve/server.py and
serve/api.py actually use, with the same types, and it tokenizes the way
the real one does in the way that matters: markers are their own ids and
ordinary text is not. test_engine.py is what checks the real binding;
this checks everything above it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from serve import xtml
from serve.engine import EngineError, WASTE_E_UNSUPPORTED

MARKERS = {1: xtml.OPEN_TOKEN, 2: xtml.CLOSE_TOKEN,
           3: xtml.SEP_TOKEN, 4: xtml.END_OF_MSG_TOKEN}
_BY_TEXT = {v: k for k, v in MARKERS.items()}

# What a chat.json container carries instead: Kimi-Linear's markup, which
# shares no marker with XTML. Same ids-are-structure discipline, different
# vocabulary — which is the whole point of testing against it.
LINEAR_MARKERS = {11: "<|im_system|>", 12: "<|im_user|>",
                  13: "<|im_assistant|>", 14: "<|im_middle|>",
                  15: "<|im_end|>"}

# Ordinary text starts here, so no piece of content can collide with a
# marker id — the same separation the real container gets from its
# reserved block.
_TEXT_BASE = 1000


def split_markers(
        text: str,
        by_text: Optional[dict[str, int]] = None) -> list[tuple[int, str]]:
    """Cut a reply into (token_id, piece), markers as their own tokens.

    Everything else becomes one token per character, which is not how BPE
    works but is the most demanding shape for a streaming parser: every
    boundary is a chunk boundary.
    """
    by_text = _BY_TEXT if by_text is None else by_text
    out: list[tuple[int, str]] = []
    rest = text
    while rest:
        best, marker = -1, ""
        for m in by_text:
            p = rest.find(m)
            if p >= 0 and (best < 0 or p < best):
                best, marker = p, m
        if best < 0:
            out.extend((_TEXT_BASE + ord(c), c) for c in rest)
            break
        out.extend((_TEXT_BASE + ord(c), c) for c in rest[:best])
        out.append((by_text[marker], ""))
        rest = rest[best + len(marker):]
    return out


@dataclass
class FakeEngine:
    """A scripted engine. Set `reply` to what the model should say."""

    reply: str = "Hello!"
    vision: bool = False
    ctx_max: int = 4096
    # A container whose tokenizer has no XTML markers — any non-K3 one.
    # marker_ids() is what discovers that, and it raises.
    no_markers: bool = False
    # The control tokens this container's tokenizer does have, and where
    # ChatFormat.load looks for a chat.json.
    markers: dict[int, str] = field(default_factory=lambda: dict(MARKERS))
    model_path: str = ""
    # Raised by generate(), to exercise the error paths.
    fail_with: Optional[Exception] = None
    # Sleep this long before each token, to test client disconnects.
    delay: float = 0.0
    # What a fresh container answers without being asked. Serve-level code
    # only reads this to report it (GET /v1/models) and to compare it
    # against a request's reasoning_effort, so a boolean stands in for the
    # real engine's reasoning_effort floor.
    default_thinking: bool = True

    prompts: list[list[int]] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    resets: int = 0
    closed: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ---- what api.py and server.py use ----------------------------------

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def _by_text(self) -> dict[str, int]:
        return {v: k for k, v in self.markers.items()}

    def marker_ids(self) -> dict[int, str]:
        if self.no_markers:
            raise EngineError(
                f"{xtml.OPEN_TOKEN} is not a single token in this container "
                f"(got 5): its specials.json does not carry K3's XTML "
                f"markers", WASTE_E_UNSUPPORTED)
        return dict(self.markers)

    def marker_ids_for(self, texts, *, what: str = "this chat format"
                       ) -> dict[int, str]:
        """The real engine asks the tokenizer; this one asks its own marker
        table. A container is a DSML container here only if the test set one
        up, which is what `dsml_markers` in this file does."""
        by_text = self._by_text
        out: dict[int, str] = {}
        for text in texts:
            if text not in by_text:
                raise EngineError(
                    f"{text} is not a single token in this container "
                    f"(got 5): its specials.json does not carry {what}'s "
                    f"markers", WASTE_E_UNSUPPORTED)
            out[by_text[text]] = text
        return out

    def model_info(self) -> dict:
        return {"n_layers": 4, "n_experts": 8, "top_k": 2, "hidden": 128,
                "ctx_max": self.ctx_max, "params_total": 1000,
                "params_active": 100, "arch": "kimi-k3",
                "quant_summary": "experts VQ2R, trunk Q4G"}

    def stats(self) -> dict:
        return {"tokens_generated": 0, "experts_hit": 3, "experts_missed": 1,
                "bytes_read": 4096, "sec_total": 0.1, "sec_io": 0.01,
                "direct_io": 0}

    def memory_used(self) -> dict:
        return {"trunk_bytes": 1, "state_bytes": 1, "scratch_bytes": 1,
                "min_expert_cache": 1, "floor_bytes": 4,
                "recommended_bytes": 8, "vision_bytes": 0}

    def tokenize(self, text: str, *, markup: bool = False,
                 add_bos: bool = False) -> list[int]:
        if markup:
            return [tid for tid, _ in split_markers(text, self._by_text)]
        # Text mode: markers are ordinary characters, never control ids.
        return [_TEXT_BASE + ord(c) for c in text]

    def tokenize_segments(self, segments, *, add_bos: bool = False) -> list[int]:
        out: list[int] = []
        for s in segments:
            if s.text:
                out.extend(self.tokenize(s.text, markup=s.markup))
        return out

    def detokenize(self, tokens: list[int]) -> str:
        out = []
        for t in tokens:
            if t in self.markers:
                out.append(self.markers[t])
            elif t >= _TEXT_BASE:
                out.append(chr(t - _TEXT_BASE))
        return "".join(out)

    def image_clear(self) -> None:
        self.images.clear()

    def image_add(self, path: str) -> int:
        if not self.vision:
            raise EngineError("image_add", WASTE_E_UNSUPPORTED)
        self.images.append(path)
        return 4

    @staticmethod
    def image_dimensions(path: str) -> tuple[int, int]:
        return 640, 480

    def image_expand(self, tokens: list[int]) -> list[int]:
        return list(tokens)

    def state_reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True

    # ---- generation ------------------------------------------------------

    def generate(self, prompt: list[int], on_token: Callable, *,
                 temperature: float = 0.0, top_p=None, top_k=None, seed=None,
                 max_tokens: int = 512, grammar=None,
                 stop_tokens=None) -> bool:
        import time

        with self._lock:
            if self.fail_with is not None:
                raise self.fail_with
            self.prompts.append(list(prompt))
            self.calls.append({"temperature": temperature, "top_p": top_p,
                               "top_k": top_k, "seed": seed,
                               "max_tokens": max_tokens,
                               "stop_tokens": list(stop_tokens or [])})

            tokens = split_markers(self.reply, self._by_text)[:max_tokens]
            for i, (tid, piece) in enumerate(tokens):
                if self.delay:
                    time.sleep(self.delay)
                info = _Info(i, tid)
                if not on_token(tid, piece, info):
                    return False
                if stop_tokens and tid in stop_tokens:
                    return True
            return True


@dataclass
class _Info:
    """Stands in for waste_token_info."""

    token_index: int
    token: int
    logprob: float = 0.0
    experts_hit: int = 0
    experts_missed: int = 0
    bytes_read: int = 0
    ms_total: float = 0.0
    ms_io: float = 0.0


# ---- reply scripts -------------------------------------------------------

def reply_plain(answer: str, *, thinking: bool = True,
                reasoning: str = "") -> str:
    """What the model emits after the generation prompt opened a channel."""
    out = []
    if thinking:
        out.append(reasoning)
        out.append(f"{xtml.CLOSE_TOKEN}think{xtml.SEP_TOKEN}")
        out.append(f"{xtml.OPEN_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(answer)
    out.append(f"{xtml.CLOSE_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}message{xtml.SEP_TOKEN}")
    out.append(xtml.END_OF_MSG_TOKEN)
    return "".join(out)


def reply_tool_call(name: str, arguments: dict, *, answer: str = "",
                    thinking: bool = True, reasoning: str = "") -> str:
    """A reply that calls one tool, rendered the way K3 renders one."""
    out = []
    if thinking:
        out.append(reasoning)
        out.append(f"{xtml.CLOSE_TOKEN}think{xtml.SEP_TOKEN}")
        out.append(f"{xtml.OPEN_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(answer)
    out.append(f"{xtml.CLOSE_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(f"{xtml.OPEN_TOKEN}tools{xtml.SEP_TOKEN}")
    out.append(f'{xtml.OPEN_TOKEN}call tool="{name}" index="1"{xtml.SEP_TOKEN}')
    for key, value in arguments.items():
        kind = xtml._xtml_type(value)
        out.append(f'{xtml.OPEN_TOKEN}argument key="{key}" '
                   f'type="{kind}"{xtml.SEP_TOKEN}')
        out.append(xtml._xtml_value(value))
        out.append(f"{xtml.CLOSE_TOKEN}argument{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}call{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}tools{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}message{xtml.SEP_TOKEN}")
    out.append(xtml.END_OF_MSG_TOKEN)
    return "".join(out)
