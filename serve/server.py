# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
server.py — the HTTP surface.

    python3 -m serve ~/models/k3.waste --port 8000

Endpoints:

    GET  /health                 liveness, plus what is loaded
    GET  /v1/models              the registry: the loaded model, plus any
                                 registered-but-not-loaded containers
    GET  /v1/models/{id}
    POST /v1/models/load         swap models (requires --models; unloads
                                 the previous model unless --keep-previous)
    POST /v1/chat/completions    streaming and not, tools, images
    POST /v1/completions         raw continuation, no chat template

Stdlib only — ThreadingHTTPServer and BaseHTTPRequestHandler, the same
shape colibri's server has. The threads are real but the engine is not
shared: a waste_ctx takes one caller, so generations queue on its lock.
That is the honest design for this engine rather than a shortcut. On a
model streaming experts off an SSD at a few tokens a second, the wait for
the lock is small next to the wait for the answer.

Streaming is written straight from the token callback, on the thread that
holds the engine lock. No handoff queue, and the client's socket closing
propagates back as a return value the engine understands: the callback
says stop, waste_generate unwinds, the lock is released, the next request
starts. A disconnected client stops costing tokens immediately, which on a
model this slow is the difference between a wasted minute and a wasted
hour.
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, NamedTuple, Optional

from . import api, dsml, glmtools, xtml
from .chatfmt import ChatFormat, ChatFormatError, PlainParser
from .engine import (Cancelled, Engine, EngineError, plan_memory,
                     usable_ram)
from .regions import RegionParser

SERVER_NAME = "waste"

# A body larger than this is refused before it is read into memory. A chat
# request is text; anything at this size is a base64 image, and several of
# those are a decode bomb rather than a conversation.
MAX_BODY_BYTES = 64 * 1024 * 1024


class ModelLoadError(EngineError):
    """A swap could not open the container it was asked for. Flows out of
    POST /v1/models/load as a 500 with the engine's own reason; the
    previously loaded model is still current."""

    def __init__(self, message: str):
        from .engine import WASTE_E_IO
        super().__init__("model load", WASTE_E_IO, message)


class ModelSlot(NamedTuple):
    """Everything one container makes true, published as one object.

    The per-model facts used to be attributes the server re-bound one at
    a time on every load, and handlers re-read one at a time. That let a
    request racing a swap see the new engine with the old chat format —
    or the old engine with the new one — and build a prompt for a format
    the container it then generated on does not speak. So the block is a
    value instead: `detect` builds a fully-resolved slot and the swap
    publishes it in a single assignment. A request takes the slot once,
    locks *its* engine, and re-checks the slot is still current before
    generating; every fact it uses thereafter comes from the slot, not
    from the moving server.
    """

    model_id: str
    engine: Engine
    model_info: dict
    markers: dict
    # One of: the xtml module, the dsml module, a ChatFormat, or None —
    # None with a chat_error, which refuses chat completions.
    chat_format: object
    chat_error: Optional[str]
    stop_tokens: list
    default_thinking: bool

    # ---- what the current model makes true -------------------------------
    #
    # The block that follows used to run once, in __init__, and every
    # handler read its verdicts as constants for the life of the process.
    # With a swappable registry they are per-model: a container without
    # XTML markers must not inherit the previous container's chat format,
    # stop tokens, or thinking default. So the same block is a factory,
    # run again on every load, and it keeps the old comment because every
    # word of it still holds.

    @classmethod
    def detect(cls, engine: Engine, model_id: str,
               start_thinking: bool) -> "ModelSlot":
        """Derive everything the handlers read from the container rather
        than from the request: model_info, the reply format, stop tokens,
        the thinking default.

        The engine lock is NOT taken here — the caller holds it, or (at
        construction) no request can have arrived yet.
        """
        default_thinking = start_thinking
        try:
            model_info = engine.model_info()
        except EngineError:
            model_info = {}
        # Markers by token id: the parser decides structure from ids, not
        # from what the text happens to spell. See regions.py.
        #
        # Two formats, asked for in order of what they can express. XTML is
        # the whole protocol — channels, tools, images — so it is tried
        # first. A container without it may still describe a plain
        # conversation in its own chat.json, the file `waste chat` has
        # always read; serving from the same file means one definition per
        # container rather than one per client.
        #
        # Neither resolving may raise out of here. It used to, which took
        # the whole process down — including /health, /v1/models and
        # /v1/completions, none of which need a chat format at all — and
        # told the operator only that a token had come out as five tokens.
        # Hold the reason instead and refuse the one endpoint that cannot be
        # served. What does not change is that the refusal happens: markup
        # the tokenizer does not have encodes as ordinary text, and the
        # model would read its own turn structure as prose and answer
        # anyway. See #34.
        try:
            markers = engine.marker_ids()
            chat_format, chat_error = xtml, None
            stop_tokens = [tid for tid, text in markers.items()
                           if text == "<|end_of_msg|>"]
        except EngineError as e:
            markers = {}
            chat_format = None
            chat_error = str(e)
            stop_tokens = []
            # DeepSeek-V4.1's DSML, before the declarative fallback and for
            # the same reason XTML comes before both: it is a whole protocol
            # — turns, thinking, tools, images — where chat.json is a plain
            # conversation. The probe is every marker or none, so a
            # container that is not this release falls through rather than
            # half-resolving.
            try:
                markers = dsml.detect(engine)
            except EngineError as e_ds:
                markers = {}
                e = f"{e}; and {e_ds}"
            else:
                chat_format, chat_error = dsml, None
                stop_tokens = [tid for tid, text in markers.items()
                               if text == dsml.EOS]
                # The generation prompt always opens a channel — so a DSML
                # container cannot be asked to answer without one being
                # chosen. Default it on, as the release does.
                default_thinking = True
                return cls(model_id, engine, model_info, markers,
                           chat_format, chat_error, stop_tokens,
                           default_thinking)
            try:
                fmt = ChatFormat.load(engine)
            except ChatFormatError as e2:
                # Both reasons, because either one alone misleads: "no XTML"
                # reads as "wrong model" when the chat.json is simply
                # missing, and the chat.json reason alone hides that the
                # richer formats were tried first.
                chat_error = f"{e}; and {e2}"
            else:
                markers = fmt.markers
                chat_format, chat_error = fmt, None
                stop_tokens = [fmt.stop_id]
                # On by default only when the format names a channel. A
                # container without one refuses a request that asks for it
                # rather than answering without it; a container whose
                # generation prompt always opens one — GLM's does — cannot
                # be asked to answer without it either.
                default_thinking = fmt.think is not None
        return cls(model_id, engine, model_info, markers,
                   chat_format, chat_error, stop_tokens, default_thinking)

    def new_parser(self, thinking: bool, tools=None):
        """The reply reader for whichever format this container speaks.

        `thinking` says which channel the generation prompt left open.
        XTML and DSML both have channels to leave open; a chat.json format
        has one only when it names a think marker.
        """
        if self.chat_format is xtml:
            return RegionParser(in_think=thinking, in_response=not thinking,
                                markers=self.markers)
        if self.chat_format is dsml:
            return dsml.DSMLParser(thinking=thinking, markers=self.markers)
        fmt = self.chat_format
        # Which tool protocol the reply reader should own: a GLM container
        # speaks its own `<tool_call>` grammar, anything else that reaches
        # a PlainParser speaks Kimi K2's five control tokens.
        tool_parser = None
        if getattr(fmt, "tool_protocol", "") == "glm":
            tool_parser = glmtools.ToolParser(tools=tools)
        return PlainParser(markers=self.markers,
                           think_close_id=getattr(fmt, "think_close_id", -1),
                           in_think=thinking and getattr(fmt, "think", None)
                           is not None,
                           tool_parser=tool_parser)


class ChatServer(ThreadingHTTPServer):
    """Threaded HTTP, one engine, one lock."""

    daemon_threads = True
    allow_reuse_address = True
    # socketserver's listen backlog is 5. Every request holds the engine
    # lock for a whole generation, so clients arrive in bursts that all
    # connect before the accept loop drains them, and a burst of 8 (what
    # TestConcurrency sends) overflowed it: macOS 27 answered the surplus
    # connects with a reset, and the test failed about half its runs. 128
    # is macOS's default somaxconn, so a larger value would be clipped.
    request_queue_size = 128

    def __init__(self, addr, handler, *, engine: Engine, model_id: str,
                 api_key: Optional[str] = None,
                 auto_register: bool = False,
                 default_max_tokens: int = 4096,
                 default_thinking: bool = True,
                 allow_local_images: bool = False,
                 log_requests: bool = True,
                 tmpdir: Optional[str] = None,
                 models: Optional[dict] = None,
                 keep_previous: bool = False,
                 engine_kwargs: Optional[dict] = None,
                 engine_factory: Optional[Callable] = None,
                 usable_ram: Optional[int] = None,
                 memory_plan: Optional[Callable] = None):
        super().__init__(addr, handler)
        self.engine_kwargs = dict(engine_kwargs or {})
        # The ceiling every swap-open is given, when there is one. It is
        # also the number the resident-set check counts with: an explicit
        # budget is used exactly as given (waste.h), so a resident model's
        # footprint is known without asking the container. 0 means the
        # engine sizes itself, and then the check asks instead.
        self.ram_budget_bytes = int(self.engine_kwargs.get("ram_budget_bytes")
                                    or 0)
        # What the machine may use, in bytes: passed in by `serve/__main__`
        # (which measures it once for its own startup checks), or measured
        # on first use. None = not known yet; 0 = this platform would not
        # say, and nothing is refused on a number nobody has.
        self._usable_ram = usable_ram
        # How a container's own plan is read, for the check below and for
        # a host whose containers are not files on this machine. The real
        # one reads the manifest only — no weights — so asking it before
        # an open costs nothing next to the open.
        self.memory_plan = memory_plan or plan_memory
        self.keep_previous = keep_previous
        # Whether POST /v1/models/load may register a path it has not seen
        # before, rather than refusing any id outside --models. Off by
        # default for the reason --allow-local-images is: without it, any
        # client that can reach this port could make the server open an
        # arbitrary file on its own filesystem.
        self.auto_register = auto_register
        # Whether this server was told it has a swappable set — and so
        # whether a generation naming an unknown model is a 404 rather than
        # something the loaded model answers. It is the operator's opt-in,
        # recorded once, not `len(self.registry)`: with auto_register any
        # client can change that length by registering a path, so the same
        # request would be answered by the loaded model before someone else's
        # registration and 404'd after it. `--models` naming the container the
        # server already serves collapses to one entry too (setdefault keeps
        # the loaded path), which left --models given and nothing strict.
        self.strict_models = bool(models) or auto_register
        self.engine_factory = engine_factory or self._default_engine_factory
        self._slot_lock = threading.RLock()
        # model_id -> the Engine holding it. With keep_previous there can be
        # more than one; without it, exactly one — the previous entry is
        # closed and dropped as the new one becomes current.
        self.engines = {model_id: engine}
        # model_id -> container path. The loaded model is registered from
        # its own path; --models adds the rest of the swappable set. An id
        # already taken by the loaded model keeps the loaded path: the
        # registry is a way to name what can be swapped to, not a way to
        # point the loaded model somewhere else.
        self.registry = {model_id: engine.model_path or model_id}
        for mid, path in (models or {}).items():
            self.registry.setdefault(mid, path)
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self.allow_local_images = allow_local_images
        self.log_requests = log_requests
        self.started = api.now()
        self._tmp = tmpdir or tempfile.mkdtemp(prefix="waste-serve-")
        self.tmpdir = self._tmp

        self._start_thinking = default_thinking
        self._detect(engine, model_id)

    # The per-model facts live on the current ModelSlot and move only by
    # whole-slot assignment; these accessors are how the handlers and the
    # tests read them without taking a slot themselves. A request path
    # must not use them per-fact — it takes `current_slot()` once.

    @property
    def engine(self) -> Engine:
        return self._slot.engine

    @property
    def model_id(self) -> str:
        return self._slot.model_id

    @property
    def model_info(self) -> dict:
        return self._slot.model_info

    @property
    def markers(self) -> dict:
        return self._slot.markers

    @property
    def chat_format(self):
        return self._slot.chat_format

    @property
    def chat_error(self):
        return self._slot.chat_error

    @property
    def stop_tokens(self) -> list:
        return self._slot.stop_tokens

    @property
    def default_thinking(self) -> bool:
        return self._slot.default_thinking

    def current_slot(self) -> ModelSlot:
        """The current model's facts, as one consistent value.

        Deliberately lock-free. The slot moves only by whole-object
        assignment in `_detect`, so one attribute read is already the
        consistent snapshot the lock was there to give: a reader sees the
        old slot or the new one, never a mix. Taking `_slot_lock` here
        would be worse than useless — `load_model` holds it across the
        new container's open, so a `GET /health` or `GET /v1/models`
        arriving mid-load would wait out a multi-gigabyte read before it
        could answer liveness. Writers still take the lock, which is what
        serialises two swaps against each other.
        """
        return self._slot

    def check_engine(self, slot: ModelSlot) -> ModelSlot:
        """Call with slot.engine.lock held, after acquiring it: is this
        slot still the one the server serves?

        Without keep_previous the swap target may have been *closed* by
        the time this request's lock was granted — an engine closes under
        its own lock, so a request queued on it either ran first (fine) or
        gets the lock after close. Answering on the closed engine is the
        500 the check exists to prevent; answering on the other engine is
        the torn-state bug. Either way the request is not served: 409 with
        the model that is current now, so the client re-reads
        GET /v1/models and retries explicitly rather than being silently
        migrated to a different model mid-conversation.
        """
        current = self.current_slot()
        if slot is not current:
            raise api.APIError(
                f"model switched to {current.model_id} while this request "
                f"was queued; re-read GET /v1/models and retry",
                status=409, type="model_switched", param="model")
        return current

    def _default_engine_factory(self, path: str) -> Engine:
        """How a swap opens a container. Overridable — tests hand in a
        factory that builds the scripted engine, and a host embedding the
        server may want its own construction arguments."""
        return Engine(path, **self.engine_kwargs)

    def _detect(self, engine: Engine, model_id: str) -> None:
        """Bind `engine` as the current model and publish everything the
        handlers read from the container — as one slot, in one assignment.

        The engine lock is NOT taken here — the caller holds it, or (at
        construction) no request can have arrived yet.
        """
        slot = ModelSlot.detect(engine, model_id, self._start_thinking)
        with self._slot_lock:
            self._slot = slot

    # ---- the model registry ----------------------------------------------

    def check_model_request(self, body: dict) -> None:
        """What a generation request may name as its model.

        Absent, empty, or equal to the loaded model's id: fine — that is
        what every client that does not know about this registry sends,
        and it must keep working. With a registry — `--models` was given —
        a name it does not know is a 404: before the registry existed any
        name was silently served by the loaded model, which made `model`
        a decorative string, and an operator who names a swappable set
        has said a wrong name should say so rather than answer as another
        model. A name the registry knows but that is not resident is a
        409, not a surprise multi-gigabyte swap in the middle of a
        conversation — the client asks for that explicitly with
        POST /v1/models/load.

        Without `--models` or `--auto-register` the registry holds only the
        loaded model, and nothing here rejects anything: a single-container
        server has no swappable set to defend, its id defaults to the
        container's file name, and clients — including the example in
        `serve --help` — send a fixed model name they cannot easily change.
        Validating every name would 404 deployments that answered them
        yesterday; strict validation is what those two flags opt into, and
        it follows the flags rather than how many ids the registry holds
        (see `strict_models`).
        """
        mid = body.get("model")
        if not isinstance(mid, str) or not mid or mid == self.model_id:
            return
        # No swappable set was asked for, so there is nothing to be strict
        # about: the only entry is the loaded model itself, and anything
        # named is served by it.
        if not self.strict_models:
            return
        if mid not in self.registry:
            raise api.APIError(f"no such model: {mid}", status=404,
                               type="not_found_error", param="model")
        if mid in self.engines:
            raise api.APIError(
                f"model {mid} is resident but not the model being served; "
                f"POST /v1/models/load to switch to it",
                status=409, type="model_not_loaded", param="model")
        raise api.APIError(
            f"model {mid} is registered but not loaded; POST /v1/models/load "
            f"to switch to it", status=409, type="model_not_loaded",
            param="model")

    def register_path(self, path: str, model_id: Optional[str] = None) -> str:
        """Add a container to the registry by its pathname, and return the
        id it was given. Only reachable when this server was started with
        --auto-register.

        `model_id` defaults the way --models PATH[=ID] does: the file name
        with the .waste suffix removed. Registering the same path again
        under the same id is a no-op; registering a *different* path under
        an id already taken is refused — the same duplicate-id protection
        parse_registry gives the CLI's --models, because two containers
        behind one name means a caller's "load this model" sometimes loads
        a different one than it just listed.

        This only edits the registry. Whether the load then fits next to
        what stays resident is check_room's question, asked by load_model
        exactly as it is for any other registered id.
        """
        p = Path(path).expanduser()
        if not p.exists():
            raise api.APIError(f"no such container: {p}", status=404,
                               type="not_found_error", param="path")
        mid = model_id or p.name.removesuffix(".waste")
        if not mid:
            raise api.APIError(
                "could not derive a model id from this path; give one "
                "explicitly with 'model'", param="path")
        existing = self.registry.get(mid)
        if existing is not None and existing != str(p):
            raise api.APIError(
                f"model id {mid} is already registered to {existing}; give "
                f"an explicit 'model' id to register {p} under a "
                f"different name", status=409, type="model_conflict",
                param="model")
        self.registry[mid] = str(p)
        return mid

    # ---- what the machine has room for -----------------------------------
    #
    # A swap holds two contexts at once: the new container is opened before
    # the old one is closed, deliberately, so that a failed open leaves the
    # server serving what it was serving. docs/SERVE.md used to ask the
    # operator to size --budget so that moment fits, and nothing checked;
    # with the default budget of 0 each context sizes itself to up to 3/4
    # of waste_usable_ram (waste.h), so the moment was ~1.5x what the
    # process may use. With --keep-previous the total is not two but every
    # model ever loaded, which no amount of sizing at startup can bound.
    #
    # So the resident set is counted, against the budgets it was opened
    # with, and a load that would not fit is refused *before* the open.

    def usable_ram_bytes(self) -> int:
        """What this process may use, in bytes. 0 = this platform will not
        say, in which case nothing here refuses anything: a limit nobody
        can measure is not a limit, and refusing to serve a model because
        the machine would not report its RAM is worse than the thing this
        check exists to prevent."""
        if self._usable_ram is None:
            try:
                self._usable_ram = int(usable_ram())
            except EngineError:
                self._usable_ram = 0
        return self._usable_ram

    def engine_budget(self, engine: Engine) -> int:
        """What one resident model holds.

        An explicit budget is exactly what the engine was opened with —
        waste.h calls it a hard ceiling on all engine allocations, used as
        given — and every open goes through `engine_kwargs`, whose
        contract is that it holds the same arguments the startup engine
        was opened with. When the engine chose its own, the ceiling is
        whatever its ladder resolved to, and waste_memory_used reports the
        result: the plan's floor plus the expert cache it actually
        allocated (n_slots x record bytes). Neither is an estimate; both
        come from the context that holds the memory.
        """
        if self.ram_budget_bytes:
            return self.ram_budget_bytes
        try:
            used = engine.memory_used()
        except EngineError:
            return 0
        return int(used.get("floor_bytes", 0)
                   + used.get("min_expert_cache", 0))

    def planned_budget(self, path: str) -> int:
        """What a container will hold when no --budget is configured.

        A prediction, and named one. With 0 the engine does not insist on
        its ideal cache size: waste_open sizes itself to whatever fits under
        ~3/4 of waste_usable_ram, taking a smaller cache on a tighter
        machine rather than refusing to open. So the figure this check
        needs is not "what it would ideally take" — recommended_bytes — but
        "the least it can run with": floor_bytes, the resident trunk and
        state before a single expert is cached. Pricing at the recommended
        figure would refuse a load the engine, sizing itself down, would
        have opened fine.

        A prediction, and named one: nothing here can run the ladder without
        opening the container, so this reads only the manifest through
        waste_plan_memory. The CLI never relies on this: --models requires
        an explicit --budget, so every figure on that path is exact; this is
        what a load with no budget is priced by instead, including one
        registered on the fly by --auto-register.
        """
        try:
            ctx = int(self.engine_kwargs.get("ctx_tokens") or 0)
            plan = self.memory_plan(path, ctx)
        except (EngineError, OSError):
            return 0
        return int(getattr(plan, "floor_bytes", 0))

    def check_room(self, model_id: str, path: str) -> None:
        """Refuse a load that would not fit next to what stays resident.

        Call with _slot_lock held, *before* the container is opened. Only
        the case that needs a new context is checked: a load that moves the
        slot to an engine already in `engines` — what --keep-previous makes
        free — adds no memory and is never refused.

        What stays resident is the whole point of the two modes: without
        keep_previous the previous model alone is resident alongside the
        new one until the swap completes, so the number is 2 x budget; with
        it, every model in `engines` stays and the total is their sum. That
        sum is the cap --keep-previous needs — it derives one from the
        budgets rather than from a hand-set count, so it cannot disagree
        with what the engines actually hold.

        Evicting a resident model to make room was the alternative and is
        rejected: it changes what is resident behind a client's back, which
        is the exact failure --keep-previous exists to prevent.

        507 rather than 503: the request is well-formed and the server will
        never satisfy it by being asked again — an operator has to change
        --budget, drop --keep-previous, or restart. A 503 invites a retry
        loop that cannot end.

        Nothing is opened, closed or moved when this refuses, so the model
        that was serving keeps serving and answers with the reason.
        """
        usable = self.usable_ram_bytes()
        if not usable:
            return
        if self.keep_previous:
            stays = sorted(self.engines)
        else:
            stays = [self._current()[0].model_id]
        held = sum(self.engine_budget(self.engines[m]) for m in stays)
        need = self.ram_budget_bytes or self.planned_budget(path)
        if held + need <= usable:
            return
        raise api.APIError(
            f"loading {model_id} would need "
            f"{api.human_bytes(need)} next to the "
            f"{api.human_bytes(held)} held by "
            f"{', '.join(stays)}: {api.human_bytes(held + need)} against "
            f"the {api.human_bytes(usable)} this machine may use. The "
            f"current model is still serving. Lower --budget, drop "
            f"--keep-previous, or start a second server for a resident set "
            f"this size cannot hold",
            status=507, type="insufficient_memory", param="model")

    def load_model(self, model_id: str) -> Optional[str]:
        """Make `model_id` the current model, and return the id of the
        model that was current before (None when it already was).

        Unloading: without keep_previous the previous engine is closed as
        part of the swap — one model resident at a time, and the freed
        RAM goes to the new container's expert cache. With keep_previous
        the previous engine stays open in `engines`, so switching back to
        it later is a slot move rather than a reopen of a multi-gigabyte
        container. Generation always serves the current slot; other
        resident models answer after the next load names them, not
        before.

        The new container is opened *before* the old one is unloaded,
        which is the opposite of the order a tight-RAM machine would
        prefer, and the reason is rollback: a swap whose open fails — a
        truncated container, an --exclusive-open conflict — must leave
        the server serving the model it was serving, and the only way to
        guarantee that is not to have closed it yet. A failed swap costs
        a load's worth of RAM for a moment; a swap that leaves no model
        loaded costs the whole server. When two containers genuinely will
        not fit together the load is refused before the open — see
        check_room, and 507 — rather than being attempted into paging.

        The old engine's lock is held across the whole swap, so a
        generation in flight finishes before the slot moves under it.
        A request that queued on the old engine and is granted the lock
        only after the swap finds the slot moved and answers 409 — see
        check_engine — rather than generating on a closed or wrong
        container. The per-model facts move with the slot: one assignment
        publishes engine, format, markers, stop tokens and thinking
        default together, so no reader can see a half-updated set.

        Lock order is engine first, slot second — here as everywhere else.
        A generation takes its engine's lock and only then calls
        check_engine, which takes _slot_lock. A swap that took _slot_lock
        first and then waited for the engine lock closes the cycle against
        a generation in flight: the swap holding the slot and waiting for
        the engine, the generation holding the engine and waiting for the
        slot. Since every request path takes _slot_lock, that wedge stops
        the whole server rather than the two requests in it. So the slot is
        read and _slot_lock released *before* the engine lock is taken, and
        re-taken underneath it, where the slot is re-checked: another swap
        that got there first held this same engine lock, so it moved the
        slot while neither of ours was held, and this one starts over on
        the slot as it is now instead of replacing a model already gone.
        """
        while True:
            # Read the slot and decide, then let _slot_lock go before the
            # engine lock is asked for — see the lock-order note above. The
            # registry is read here too, so an unknown id is a fast 404
            # rather than a wait behind whatever generation holds the
            # engine.
            with self._slot_lock:
                previous_slot, previous_engine = self._current()
                if model_id == previous_slot.model_id:
                    return None
                path = self.registry.get(model_id)
                if path is None:
                    raise api.APIError(f"no such model: {model_id}", status=404,
                                       type="not_found_error", param="model")
            # Only now the engine lock, and _slot_lock again *underneath*
            # it, the order the rest of this file uses. Re-check first:
            # a swap that got here first held this same engine lock, so
            # the slot moved while neither of our locks was held, and
            # moving it to a target that is already gone is the stale-slot
            # bug check_engine exists to prevent on the request side.
            with previous_engine.lock:
                with self._slot_lock:
                    if self._current()[0] is not previous_slot:
                        continue
                    # A model kept resident by keep_previous does not need
                    # an open at all — its waste_ctx still holds the state
                    # it had. Moving the slot to it costs a format
                    # re-detect, not a load.
                    resident = self.engines.get(model_id)
                    if resident is not None:
                        self._detect(resident, model_id)
                        prev = previous_slot.model_id
                    else:
                        # Room for it, before a byte of it is allocated —
                        # and before anything is closed, so a refusal here
                        # is indistinguishable from never having been
                        # asked: same current model, same resident set,
                        # same open containers.
                        self.check_room(model_id, path)
                        try:
                            engine = self.engine_factory(path)
                        except EngineError as e:
                            raise ModelLoadError(
                                f"could not load {model_id}: {e}") from e
                        self._detect(engine, model_id)
                        self.engines[model_id] = engine
                        if not self.keep_previous:
                            self.engines.pop(previous_slot.model_id)
                            previous_engine.close()
                        prev = previous_slot.model_id
                    err = self._slot.chat_error
                    line = f"swap: {model_id}"
                    if err:
                        line += f"  chat_error: {err}"
                    sys.stderr.write(f"{line}\n")
                    return prev

    def _current(self) -> tuple:
        """(the current ModelSlot, its engine). Call with _slot_lock held."""
        slot = self._slot
        return slot, slot.engine

    def close_engines(self) -> None:
        """Every engine this server still holds. The shutdown path; with
        keep_previous there may be several."""
        for engine in list(self.engines.values()):
            engine.close()
        self.engines.clear()

    def new_parser(self, thinking: bool, tools=None):
        """The reply reader for the current container. Reads the slot under
        the lock, so a request that calls it (as the chat path does, inside
        its locked section) gets the parser of the model it is serving."""
        return self.current_slot().new_parser(thinking, tools=tools)

    def handle_error(self, request, client_address):
        """A client hanging up is not an error worth a traceback.

        socketserver's default prints the whole stack for a reset socket,
        which on a streaming endpoint means every cancelled request writes
        a fake crash report to the log.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError,
                            TimeoutError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVER_NAME}/1"
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------

    def log_message(self, fmt, *args):
        if getattr(self.server, "log_requests", True):
            line = fmt % args
            model = getattr(self, "_log_model", None)
            if model:
                line += "  [model=%s]" % model
            sys.stderr.write("%s - %s\n" % (self.address_string(), line))

    def _log_model_from(self, body):
        """The model name this request's log line carries: what the request
        asked for, or the loaded model when it named none. A success line
        then shows the model that served it, and a 409 or 404 shows the
        model that was refused — `who was asked` is on the log line even
        when `who answered` is nobody."""
        mid = body.get("model")
        self._log_model = (mid if isinstance(mid, str) and mid
                           else self.server.model_id)

    def _send_json(self, status: int, payload: dict, *, headers=None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Set when we are about to hang up on an unread body; saying so
        # is what lets the client reconnect rather than see a reset.
        if self.close_connection:
            self.send_header("Connection", "close")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, err: api.APIError) -> None:
        self._send_json(err.status, err.to_json())

    def _read_body(self) -> dict:
        # Refusing a request without consuming its body desynchronises a
        # keep-alive connection: the bytes we did not read become the next
        # request line, and the client's following request is parsed out of
        # its own predecessor's payload. Every path that rejects before
        # reading closes the connection instead.
        def refuse(message: str, status: int) -> api.APIError:
            self.close_connection = True
            return api.APIError(message, status=status)

        length = self.headers.get("Content-Length")
        if length is None:
            raise refuse("Content-Length is required", 411)
        try:
            n = int(length)
        except ValueError:
            raise refuse("Content-Length is not a number", 400)
        if n < 0 or n > MAX_BODY_BYTES:
            raise refuse(f"request body may not exceed "
                         f"{MAX_BODY_BYTES} bytes", 413)
        raw = self.rfile.read(n) if n else b""
        if not raw:
            raise api.APIError("request body is empty")
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise api.APIError(f"request body is not valid JSON: {e}")
        if not isinstance(body, dict):
            raise api.APIError("request body must be a JSON object")
        return body

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            # Compare in constant time: a server that returns faster on a
            # wrong first byte leaks the key one byte at a time.
            import hmac
            return hmac.compare_digest(header[7:], key)
        return False

    # ---- routing --------------------------------------------------------

    def do_GET(self):
        self._log_model = None    # no keep-alive request inherits the last one's model
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                return self._health()
            if not self._authorized():
                raise api.APIError("invalid API key", status=401,
                                   type="authentication_error")
            if path == "/v1/models":
                return self._models()
            if path.startswith("/v1/models/"):
                return self._model(path[len("/v1/models/"):])
            raise api.APIError(f"no route for GET {path}", status=404,
                               type="not_found_error")
        except api.APIError as e:
            self._send_error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        self._log_model = None    # no keep-alive request inherits the last one's model
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if not self._authorized():
                self.close_connection = True
                raise api.APIError("invalid API key", status=401,
                                   type="authentication_error")
            if path == "/v1/chat/completions":
                return self._chat()
            if path == "/v1/completions":
                return self._completions()
            if path == "/v1/models/load":
                return self._load_model()
            self.close_connection = True
            raise api.APIError(f"no route for POST {path}", status=404,
                               type="not_found_error")
        except api.APIError as e:
            self._send_error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except EngineError as e:
            self._send_error(api.APIError(str(e), status=500,
                                          type="engine_error"))

    # ---- endpoints ------------------------------------------------------

    def _health(self):
        from . import engine as engine_mod
        self._send_json(200, {
            "status": "ok",
            "model": self.server.current_slot().model_id,
            "engine": engine_mod.build_info(),
            "uptime_s": api.now() - self.server.started,
        })

    def _models(self):
        srv = self.server
        # The whole registry, current model first, so a client scanning the
        # list sees what is resident before what is only available. `loaded`
        # is residency — the same set the load response reports: a model
        # `--keep-previous` still holds a waste_ctx for is loaded, whether
        # or not it is the one being served. Only the current entry carries
        # a `waste` shape: the per-model facts live on the current slot and
        # move with it, and a resident-but-idle engine's shape is re-derived
        # when a swap makes it current again. A registered-but-never-opened
        # container is loaded=false and carries no shape — its facts are
        # unknown until it is opened.
        # One slot for the whole listing: read per fact, a GET that overlaps
        # a swap could name one model as current and carry the other's shape.
        slot = srv.current_slot()
        data = [api.model_object(slot.model_id, srv.started,
                                 slot.model_info, loaded=True)]
        data += [api.model_object(mid, srv.started, None,
                                  loaded=mid in srv.engines)
                 for mid in sorted(srv.registry)
                 if mid != slot.model_id]
        self._send_json(200, {"object": "list", "data": data})

    def _model(self, model_id: str):
        srv = self.server
        slot = srv.current_slot()
        if model_id == slot.model_id:
            self._send_json(200, api.model_object(model_id, srv.started,
                                                  slot.model_info, loaded=True))
            return
        if model_id not in srv.registry:
            raise api.APIError(f"no such model: {model_id}", status=404,
                               type="not_found_error", param="model")
        # Residency, as in _models — with --keep-previous a model the
        # server has swapped away from is still open and answers loaded.
        self._send_json(200, api.model_object(model_id, srv.started, None,
                                              loaded=model_id in srv.engines))

    def _load_model(self):
        """POST /v1/models/load — swap the model this server serves.

        Body: {"model": "<id>"} where id is a name from GET /v1/models.
        With --auto-register, {"path": "<pathname>"} may name a container
        this server has not seen before instead — the id defaults to the
        file name without .waste, or the caller's own "model" if it gives
        one. The path is registered first, then loaded exactly like any
        other registry entry: the same check_room budget accounting and
        open-before-close rollback apply, because load_model does not know
        or care how an id got into the registry.

        The swap happens under the current engine's lock, so a generation
        in flight finishes first; the reply says which model went out and
        which came in. Errors: 404 unknown id or path, 409 a path id
        collision, 500 (ModelLoadError) the container would not open, 507
        the load would not fit next to what stays resident — the previous
        model is still served in every case.
        """
        body = self._read_body()
        srv = self.server
        mid = body.get("model")
        path = body.get("path")
        self._log_model_from(body)      # 404/500 lines name the model
        registered = None                  # set if this request adds an id
        if path is not None:
            # --auto-register: a pathname instead of (or alongside) an id.
            if not srv.auto_register:
                raise api.APIError(
                    "path registration is disabled; start the server with "
                    "--auto-register to let this endpoint register a "
                    "container by pathname", param="path")
            if not isinstance(path, str) or not path:
                raise api.APIError("'path' must be a non-empty string",
                                   param="path")
            if mid is not None and (not isinstance(mid, str) or not mid):
                raise api.APIError("'model' must be a non-empty string",
                                   param="model")
            # Only an id this request introduces may be rolled back below;
            # re-loading an operator's --models entry must never
            # deregister it when the load is refused.
            known_before = set(srv.registry)
            mid = srv.register_path(path, mid)   # 404 no such file / 409 id
            if mid not in known_before:
                registered = mid
        elif not isinstance(mid, str) or not mid:
            raise api.APIError("'model' must be a non-empty string",
                               param="model")
        try:
            previous = srv.load_model(mid)  # raises 404 / ModelLoadError / 507
        except BaseException:
            # A refused load (507 check_room, ModelLoadError, anything the
            # engine raises) must not leave the id this request just
            # registered behind: a later load-by-id would then bypass the
            # --auto-register gate that only the path-carrying request
            # passed. Roll back exactly what this request added.
            if registered is not None:
                srv.registry.pop(registered, None)
            raise
        self._send_json(200, {
            "object": "model.load",
            "loaded": mid,
            "previous": previous,
            "models": [m for m in srv.registry if m in srv.engines],
        })

    # ---- chat -----------------------------------------------------------

    def _chat(self):
        body = self._read_body()
        srv = self.server
        # One snapshot, taken before the lock and used for everything this
        # request decides: the engine it will lock, the format it renders
        # the prompt in, the ctx it reads, the tokens it stops on. Reading
        # them one attribute at a time off the server is what let a request
        # racing a swap see the new engine with the old chat format.
        slot = srv.current_slot()
        engine = slot.engine
        self._log_model_from(body)

        # Before anything else, and before the engine lock: a request that
        # names a model this process does not serve should hear it from a
        # 404, not from the model's own reply. Absent or matching the
        # loaded model's id passes untouched.
        srv.check_model_request(body)

        # Before anything else, and before the engine lock: this container
        # has no chat format we can render, and no request can change that.
        # 400 rather than 501 because for an OpenAI client the unsupported
        # thing is the model, which is a request parameter — and a 501 is
        # the one status those clients tend to retry.
        if slot.chat_error:
            raise api.APIError(
                f"this model cannot be used for chat completions: "
                f"{slot.chat_error}. serve/ renders Kimi K3's XTML prompt "
                f"format and no other. POST /v1/completions for raw "
                f"continuation, or use `waste chat`, which reads the "
                f"container's own chat.json",
                status=400, param="model", code="unsupported_chat_format")

        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise api.APIError("'stream' must be a boolean", param="stream")

        # The lock spans prompt building *and* generation, not just the
        # generation, for two reasons that both showed up as real defects:
        #
        #  - A waste_ctx carries conversation state across calls, which is
        #    what makes `waste chat` a conversation. HTTP requests are not
        #    a conversation: without the reset, request N is prefilled on
        #    top of request N-1 and answers it differently — and one
        #    client's turn conditions another's.
        #  - build_prompt fills the engine's image queue, which generate
        #    then consumes. Building one request's prompt while another is
        #    between its own build and generate would hand the second
        #    request the first one's pictures.
        with engine.lock:
            # First thing under the lock: did the slot move between the
            # snapshot and this grant? Without keep_previous the swap may
            # also have *closed* this engine — close takes the same lock,
            # so we are here either before it (our generation runs to
            # completion, the swap waits) or after it, and the check
            # refuses before state_reset touches the dead ctx.
            srv.check_engine(slot)

            engine.state_reset()

            prompt = api.build_prompt(
                engine, body,
                default_thinking=slot.default_thinking,
                allow_local_images=srv.allow_local_images,
                tmpdir=srv.tmpdir, fmt=slot.chat_format)

            opts = api.generation_options(
                body, default_max_tokens=srv.default_max_tokens,
                ctx_max=slot.model_info.get("ctx_max", 0),
                prompt_len=len(prompt.tokens))
            stops = api.stop_strings(body)

            request_id = api.new_id("chatcmpl")
            created = api.now()
            parser = slot.new_parser(prompt.thinking, tools=body.get("tools"))

            if stream:
                self._chat_stream(slot, body, prompt, opts, stops, parser,
                                  request_id, created)
            else:
                self._chat_blocking(slot, body, prompt, opts, stops, parser,
                                    request_id, created)

    def _run(self, slot, prompt, opts, stops, parser, on_delta):
        """Drive one generation. Returns (n_tokens, hit_limit, stopped).

        `on_delta(delta)` is called on the engine thread for each token, and
        may raise Cancelled to stop — which is how a disconnected streaming
        client stops the generation rather than paying for all of it.
        """
        engine = slot.engine
        stops = [s for s in stops if s]
        state = {"n": 0, "stopped": False, "content_sent": 0}

        def stop_at(text):
            hits = [p for s in stops if (p := text.find(s)) >= 0]
            return min(hits) if hits else None

        def safe_content_end(text):
            """Exclude a suffix that may become a stop on the next token."""
            end = len(text)
            for s in stops:
                limit = min(len(s) - 1, len(text))
                for n in range(limit, 0, -1):
                    if text.endswith(s[:n]):
                        end = min(end, len(text) - n)
                        break
            return end

        def deliver(delta, *, final=False):
            hit = stop_at(parser.content)
            if hit is not None:
                parser.content = parser.content[:hit]
            end = len(parser.content) if final or hit is not None else safe_content_end(
                parser.content)
            sent = state["content_sent"]
            delta.content = parser.content[sent:end] if end >= sent else ""
            state["content_sent"] = end
            if delta.reasoning or delta.content or delta.tool_calls:
                on_delta(delta)
            return hit is not None

        def on_token(token_id, piece, info):
            state["n"] += 1
            delta = parser.feed_token(token_id, piece)
            if deliver(delta, final=parser.finished):
                state["stopped"] = True
                return False
            if parser.finished:
                state["stopped"] = True
                return False
            return True

        completed = engine.generate(
            prompt.tokens, on_token,
            temperature=opts["temperature"], top_p=opts["top_p"],
            top_k=opts["top_k"], seed=opts["seed"],
            max_tokens=opts["max_tokens"],
            stop_tokens=slot.stop_tokens or None)
        tail = parser.finish()
        if not state["stopped"]:
            if deliver(tail, final=True):
                state["stopped"] = True
        hit_limit = completed and state["n"] >= opts["max_tokens"]
        return state["n"], hit_limit, state["stopped"]

    def _chat_blocking(self, slot, body, prompt, opts, stops, parser,
                       request_id, created):
        t0 = time.time()
        n, hit_limit, stopped = self._run(slot, prompt, opts, stops, parser,
                                          lambda d: None)
        reason = api.finish_reason(parser, hit_limit=hit_limit, stopped=stopped)
        usage = api.usage_block(len(prompt.tokens), n)
        payload = api.chat_completion(
            parser, model=slot.model_id, request_id=request_id,
            created=created, reason=reason, usage=usage,
            extra=api.engine_extra(slot.engine.stats(),
                                   ms=(time.time() - t0) * 1000))
        self._send_json(200, payload)

    def _chat_stream(self, slot, body, prompt, opts, stops, parser,
                     request_id, created):
        model = slot.model_id
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # Chunked, because the length is unknown until the model stops. With
        # HTTP/1.1 and no Content-Length the alternative is closing the
        # socket to signal the end, which costs the client its keep-alive.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        t0 = time.time()
        sent_role = False
        named: set[int] = set()

        def write(payload) -> None:
            data = api.sse(payload)
            # Manual chunked framing: BaseHTTPRequestHandler does not do it.
            self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def on_delta(delta):
            nonlocal sent_role
            try:
                if not sent_role:
                    write(api.chunk(request_id, created, model,
                                    {"role": "assistant", "content": ""}))
                    sent_role = True
                if delta.reasoning:
                    write(api.chunk(request_id, created, model,
                                    {"reasoning_content": delta.reasoning}))
                if delta.content:
                    write(api.chunk(request_id, created, model,
                                    {"content": delta.content}))
                for i in delta.tool_calls:
                    if i in named:
                        continue
                    call = parser.tool_calls[i]
                    if not call.name:
                        continue
                    # Announce the call as soon as its name is known; the
                    # arguments follow in one piece at the end, which
                    # concatenating clients assemble correctly either way.
                    named.add(i)
                    write(api.chunk(request_id, created, model, {
                        "tool_calls": [{
                            "index": i, "id": call.id or f"call_{i + 1}",
                            "type": "function",
                            "function": {"name": call.name, "arguments": ""},
                        }]}))
            except (BrokenPipeError, ConnectionResetError):
                raise Cancelled()

        try:
            n, hit_limit, stopped = self._run(slot, prompt, opts, stops,
                                              parser, on_delta)
        except Cancelled:
            # The client is gone. Nothing left to write to.
            return

        try:
            if not sent_role:
                write(api.chunk(request_id, created, model,
                                {"role": "assistant", "content": ""}))
            for i, call in enumerate(parser.tool_calls):
                if i not in named:
                    write(api.chunk(request_id, created, model, {
                        "tool_calls": [{
                            "index": i, "id": call.id or f"call_{i + 1}",
                            "type": "function",
                            "function": {"name": call.name, "arguments": ""},
                        }]}))
                write(api.chunk(request_id, created, model, {
                    "tool_calls": [{
                        "index": i,
                        "function": {"arguments": call.arguments_json()},
                    }]}))

            reason = api.finish_reason(parser, hit_limit=hit_limit,
                                       stopped=stopped)
            write(api.chunk(request_id, created, model, {}, reason))

            opts_in = body.get("stream_options") or {}
            if isinstance(opts_in, dict) and opts_in.get("include_usage"):
                usage = api.usage_block(len(prompt.tokens), n)
                write({"id": request_id, "object": "chat.completion.chunk",
                       "created": created, "model": model, "choices": [],
                       "usage": usage})
            write({"id": request_id, "object": "chat.completion.chunk",
                   "created": created, "model": model, "choices": [],
                   "waste": api.engine_extra(slot.engine.stats(),
                                             ms=(time.time() - t0) * 1000)})
            write("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    # ---- raw completions ------------------------------------------------

    def _completions(self):
        """Continuation, with no chat template applied.

        Useful for a base model, and for seeing the prompt the model
        actually gets. `prompt` is encoded as ordinary text: markup in it
        stays markup-shaped text and cannot become control tokens.
        """
        body = self._read_body()
        srv = self.server
        # Same discipline as _chat: one snapshot before the lock, and a
        # re-check under it, so a request queued behind a swap answers 409
        # instead of generating on a closed (or wrong) engine.
        slot = srv.current_slot()
        engine = slot.engine
        self._log_model_from(body)      # before check_model_request: a 404 or
        # 409 line should still name the model that was refused
        srv.check_model_request(body)
        prompt_text = body.get("prompt")
        if isinstance(prompt_text, list):
            if len(prompt_text) != 1 or not isinstance(prompt_text[0], str):
                raise api.APIError("'prompt' must be a string", param="prompt")
            prompt_text = prompt_text[0]
        if not isinstance(prompt_text, str) or not prompt_text:
            raise api.APIError("'prompt' must be a non-empty string",
                               param="prompt")

        with engine.lock:
            srv.check_engine(slot)
            engine.state_reset()      # each request stands alone
            tokens = engine.tokenize(prompt_text)
            if not tokens:
                raise api.APIError("'prompt' encoded to no tokens",
                                   param="prompt")
            opts = api.generation_options(
                body, default_max_tokens=srv.default_max_tokens,
                ctx_max=slot.model_info.get("ctx_max", 0),
                prompt_len=len(tokens))
            stops = api.stop_strings(body)

            pieces: list[str] = []
            state = {"stopped": False}

            def on_token(token_id, piece, info):
                pieces.append(piece)
                for s in stops:
                    if s and s in "".join(pieces):
                        state["stopped"] = True
                        return False
                return True

            completed = engine.generate(
                tokens, on_token, temperature=opts["temperature"],
                top_p=opts["top_p"], top_k=opts["top_k"], seed=opts["seed"],
                max_tokens=opts["max_tokens"],
                stop_tokens=slot.stop_tokens or None)

        text = "".join(pieces)
        for s in stops:
            if s and s in text:
                text = text.split(s, 1)[0]
        hit_limit = completed and len(pieces) >= opts["max_tokens"]
        self._send_json(200, {
            "id": api.new_id("cmpl"),
            "object": "text_completion",
            "created": api.now(),
            "model": slot.model_id,
            "choices": [{"index": 0, "text": text, "logprobs": None,
                         "finish_reason": "length" if hit_limit else "stop"}],
            "usage": api.usage_block(len(tokens), len(pieces)),
        })


def serve(engine: Engine, *, host: str = "127.0.0.1", port: int = 8000,
          model_id: str = "waste", api_key: Optional[str] = None,
          default_max_tokens: int = 4096, default_thinking: bool = True,
          allow_local_images: bool = False, log_requests: bool = True,
          ready: Optional[threading.Event] = None,
          models: Optional[dict] = None, keep_previous: bool = False,
          auto_register: bool = False,
          engine_kwargs: Optional[dict] = None,
          engine_factory: Optional[Callable] = None,
          usable_ram: Optional[int] = None,
          memory_plan: Optional[Callable] = None) -> ChatServer:
    """Build the server. The caller decides whether to serve_forever.

    models names the rest of the swappable registry (id -> container
    path); keep_previous decides whether a swap unloads the model it
    replaces; auto_register lets POST /v1/models/load register a path it
    has not seen before instead of refusing any id outside models;
    engine_kwargs is how the default factory opens a container on a swap —
    the same arguments the startup engine was opened with — and
    engine_factory replaces that factory wholesale, for a host that builds
    engines its own way (tests do exactly that).

    usable_ram is what the resident-set check counts against — pass the
    figure already measured so a load never probes for it — and
    memory_plan is how a container with no explicit budget is priced.
    Both default to the real thing, and neither is needed by a caller
    that never swaps models.
    """
    # IPv6-capable when the host asks for it, without forcing it: binding
    # :: on a host with IPv6 disabled fails outright.
    if ":" in host:
        ChatServer.address_family = socket.AF_INET6
    srv = ChatServer((host, port), Handler, engine=engine, model_id=model_id,
                     api_key=api_key, default_max_tokens=default_max_tokens,
                     default_thinking=default_thinking,
                     allow_local_images=allow_local_images,
                     log_requests=log_requests, models=models,
                     keep_previous=keep_previous,
                     auto_register=auto_register,
                     engine_kwargs=engine_kwargs,
                     engine_factory=engine_factory,
                     usable_ram=usable_ram, memory_plan=memory_plan)
    if ready is not None:
        ready.set()
    return srv
