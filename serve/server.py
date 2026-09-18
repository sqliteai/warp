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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from . import api, dsml, glmtools, xtml
from .chatfmt import ChatFormat, ChatFormatError, PlainParser
from .engine import Cancelled, Engine, EngineError
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
                 default_max_tokens: int = 4096,
                 default_thinking: bool = True,
                 allow_local_images: bool = False,
                 log_requests: bool = True,
                 tmpdir: Optional[str] = None,
                 models: Optional[dict] = None,
                 keep_previous: bool = False,
                 engine_kwargs: Optional[dict] = None,
                 engine_factory: Optional[Callable] = None):
        super().__init__(addr, handler)
        self.engine_kwargs = dict(engine_kwargs or {})
        self.keep_previous = keep_previous
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

    def _default_engine_factory(self, path: str) -> Engine:
        """How a swap opens a container. Overridable — tests hand in a
        factory that builds the scripted engine, and a host embedding the
        server may want its own construction arguments."""
        return Engine(path, **self.engine_kwargs)

    # ---- what the current model makes true -------------------------------
    #
    # The block that follows used to run once, in __init__, and every
    # handler read its verdicts as constants for the life of the process.
    # With a swappable registry they are per-model: a container without
    # XTML markers must not inherit the previous container's chat format,
    # stop tokens, or thinking default. So the same block is a method, run
    # again on every load, and it keeps the old comment because every word
    # of it still holds.

    def _detect(self, engine: Engine, model_id: str) -> None:
        """Bind `engine` as the current model and re-derive everything the
        handlers read from the container rather than from the request:
        model_info, the reply format, stop tokens, the thinking default.

        The engine lock is NOT taken here — the caller holds it, or (at
        construction) no request can have arrived yet.
        """
        self.engine = engine
        self.model_id = model_id
        self.default_thinking = self._start_thinking
        try:
            self.model_info = engine.model_info()
        except EngineError:
            self.model_info = {}
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
            self.markers = engine.marker_ids()
            self.chat_format = xtml
            self.chat_error = None
            self.stop_tokens = [tid for tid, text in self.markers.items()
                                if text == "<|end_of_msg|>"]
        except EngineError as e:
            self.markers = {}
            self.chat_format = None
            self.chat_error = str(e)
            self.stop_tokens = []
            # DeepSeek-V4.1's DSML, before the declarative fallback and for
            # the same reason XTML comes before both: it is a whole protocol
            # — turns, thinking, tools, images — where chat.json is a plain
            # conversation. The probe is every marker or none, so a
            # container that is not this release falls through rather than
            # half-resolving.
            try:
                self.markers = dsml.detect(engine)
            except EngineError as e_ds:
                self.markers = {}
                e = f"{e}; and {e_ds}"
            else:
                self.chat_format = dsml
                self.chat_error = None
                self.stop_tokens = [tid for tid, text in self.markers.items()
                                    if text == dsml.EOS]
                # The generation prompt always opens a channel — <think> or
                # </think> — so a DSML container cannot be asked to answer
                # without one being chosen. Default it on, as the release
                # does.
                self.default_thinking = True
                return
            try:
                fmt = ChatFormat.load(engine)
            except ChatFormatError as e2:
                # Both reasons, because either one alone misleads: "no XTML"
                # reads as "wrong model" when the chat.json is simply
                # missing, and the chat.json reason alone hides that the
                # richer formats were tried first.
                self.chat_error = f"{e}; and {e2}"
            else:
                self.markers = fmt.markers
                self.chat_format = fmt
                self.chat_error = None
                self.stop_tokens = [fmt.stop_id]
                # On by default only when the format names a channel. A
                # container without one refuses a request that asks for it
                # rather than answering without it; a container whose
                # generation prompt always opens one — GLM's does — cannot
                # be asked to answer without it either.
                self.default_thinking = fmt.think is not None

    # ---- the model registry ----------------------------------------------

    def check_model_request(self, body: dict) -> None:
        """What a generation request may name as its model.

        Absent, empty, or equal to the loaded model's id: fine — that is
        what every client that does not know about this registry sends,
        and it must keep working. A name the registry does not know is a
        404: before the registry existed any name was silently served by
        the loaded model, which made `model` a decorative string. A name
        the registry knows but that is not resident is a 409, not a
        surprise multi-gigabyte swap in the middle of a conversation —
        the client asks for that explicitly with POST /v1/models/load.
        """
        mid = body.get("model")
        if not isinstance(mid, str) or not mid or mid == self.model_id:
            return
        if mid not in self.registry:
            raise api.APIError(f"no such model: {mid}", status=404,
                               type="not_found_error", param="model")
        raise api.APIError(
            f"model {mid} is registered but not loaded; POST /v1/models/load "
            f"to switch to it", status=409, type="model_not_loaded",
            param="model")

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
        not fit together, size --budget so each open plans against what
        the engine can actually get.

        The old engine's lock is held across the whole swap, so a
        generation in flight finishes before the slot moves under it, and
        no request that took the old lock can find its engine closed.
        """
        with self._slot_lock:
            current = self._current()
            if model_id == current[0]:
                return None
            path = self.registry.get(model_id)
            if path is None:
                raise api.APIError(f"no such model: {model_id}", status=404,
                                   type="not_found_error", param="model")
            previous_id, previous_engine = current
            # A model kept resident by keep_previous does not need an
            # open at all — its waste_ctx still holds the state it had.
            # Moving the slot to it costs a format re-detect, not a load.
            resident = self.engines.get(model_id)
            with previous_engine.lock:
                if resident is not None:
                    self._detect(resident, model_id)
                    return previous_id
                try:
                    engine = self.engine_factory(path)
                except EngineError as e:
                    raise ModelLoadError(
                        f"could not load {model_id}: {e}") from e
                self._detect(engine, model_id)
                self.engines[model_id] = engine
                if self.keep_previous:
                    return previous_id
                self.engines.pop(previous_id)
                previous_engine.close()
                return previous_id

    def _current(self) -> tuple:
        with self._slot_lock:
            model_id = self.model_id
            return model_id, self.engines.get(model_id, self.engine)

    def close_engines(self) -> None:
        """Every engine this server still holds. The shutdown path; with
        keep_previous there may be several."""
        for engine in list(self.engines.values()):
            engine.close()
        self.engines.clear()

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
            "model": self.server.model_id,
            "engine": engine_mod.build_info(),
            "uptime_s": api.now() - self.server.started,
        })

    def _models(self):
        srv = self.server
        # The whole registry, current model first, so a client scanning the
        # list sees what is resident before what is only available. A
        # registered-but-not-loaded entry carries no `waste` shape: its
        # per-container facts are unknown until it is opened.
        data = [api.model_object(srv.model_id, srv.started,
                                 srv.model_info, loaded=True)]
        data += [api.model_object(mid, srv.started, None, loaded=False)
                 for mid in sorted(srv.registry)
                 if mid != srv.model_id]
        self._send_json(200, {"object": "list", "data": data})

    def _model(self, model_id: str):
        srv = self.server
        if model_id == srv.model_id:
            self._send_json(200, api.model_object(model_id, srv.started,
                                                  srv.model_info, loaded=True))
            return
        if model_id not in srv.registry:
            raise api.APIError(f"no such model: {model_id}", status=404,
                               type="not_found_error", param="model")
        self._send_json(200, api.model_object(model_id, srv.started,
                                              None, loaded=False))

    def _load_model(self):
        """POST /v1/models/load — swap the model this server serves.

        Body: {"model": "<id>"} where id is a name from GET /v1/models.
        The swap happens under the current engine's lock, so a generation
        in flight finishes first; the reply says which model went out and
        which came in. Errors: 404 unknown id, 500 (ModelLoadError) the
        container would not open — the previous model is still served.
        """
        body = self._read_body()
        srv = self.server
        mid = body.get("model")
        self._log_model_from(body)      # 404/500 lines name the model
        if not isinstance(mid, str) or not mid:
            raise api.APIError("'model' must be a non-empty string",
                               param="model")
        previous = srv.load_model(mid)      # raises 404 / ModelLoadError
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
        engine = srv.engine
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
        if srv.chat_error:
            raise api.APIError(
                f"this model cannot be used for chat completions: "
                f"{srv.chat_error}. serve/ renders Kimi K3's XTML prompt "
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
            engine.state_reset()

            prompt = api.build_prompt(
                engine, body,
                default_thinking=srv.default_thinking,
                allow_local_images=srv.allow_local_images,
                tmpdir=srv.tmpdir, fmt=srv.chat_format)

            opts = api.generation_options(
                body, default_max_tokens=srv.default_max_tokens,
                ctx_max=srv.model_info.get("ctx_max", 0),
                prompt_len=len(prompt.tokens))
            stops = api.stop_strings(body)

            request_id = api.new_id("chatcmpl")
            created = api.now()
            parser = srv.new_parser(prompt.thinking, tools=body.get("tools"))

            if stream:
                self._chat_stream(body, prompt, opts, stops, parser,
                                  request_id, created)
            else:
                self._chat_blocking(body, prompt, opts, stops, parser,
                                    request_id, created)

    def _run(self, prompt, opts, stops, parser, on_delta):
        """Drive one generation. Returns (n_tokens, hit_limit, stopped).

        `on_delta(delta)` is called on the engine thread for each token, and
        may raise Cancelled to stop — which is how a disconnected streaming
        client stops the generation rather than paying for all of it.
        """
        engine = self.server.engine
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
            stop_tokens=self.server.stop_tokens or None)
        tail = parser.finish()
        if not state["stopped"]:
            if deliver(tail, final=True):
                state["stopped"] = True
        hit_limit = completed and state["n"] >= opts["max_tokens"]
        return state["n"], hit_limit, state["stopped"]

    def _chat_blocking(self, body, prompt, opts, stops, parser,
                       request_id, created):
        t0 = time.time()
        n, hit_limit, stopped = self._run(prompt, opts, stops, parser,
                                          lambda d: None)
        reason = api.finish_reason(parser, hit_limit=hit_limit, stopped=stopped)
        usage = api.usage_block(len(prompt.tokens), n)
        payload = api.chat_completion(
            parser, model=self.server.model_id, request_id=request_id,
            created=created, reason=reason, usage=usage,
            extra=api.engine_extra(self.server.engine.stats(),
                                   ms=(time.time() - t0) * 1000))
        self._send_json(200, payload)

    def _chat_stream(self, body, prompt, opts, stops, parser,
                     request_id, created):
        model = self.server.model_id
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
            n, hit_limit, stopped = self._run(prompt, opts, stops, parser,
                                              on_delta)
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
                   "waste": api.engine_extra(self.server.engine.stats(),
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

        with srv.engine.lock:
            srv.engine.state_reset()      # each request stands alone
            tokens = srv.engine.tokenize(prompt_text)
            if not tokens:
                raise api.APIError("'prompt' encoded to no tokens",
                                   param="prompt")
            opts = api.generation_options(
                body, default_max_tokens=srv.default_max_tokens,
                ctx_max=srv.model_info.get("ctx_max", 0),
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

            completed = srv.engine.generate(
                tokens, on_token, temperature=opts["temperature"],
                top_p=opts["top_p"], top_k=opts["top_k"], seed=opts["seed"],
                max_tokens=opts["max_tokens"],
                stop_tokens=srv.stop_tokens or None)

        text = "".join(pieces)
        for s in stops:
            if s and s in text:
                text = text.split(s, 1)[0]
        hit_limit = completed and len(pieces) >= opts["max_tokens"]
        self._send_json(200, {
            "id": api.new_id("cmpl"),
            "object": "text_completion",
            "created": api.now(),
            "model": srv.model_id,
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
          engine_kwargs: Optional[dict] = None,
          engine_factory: Optional[Callable] = None) -> ChatServer:
    """Build the server. The caller decides whether to serve_forever.

    models names the rest of the swappable registry (id -> container
    path); keep_previous decides whether a swap unloads the model it
    replaces; engine_kwargs is how the default factory opens a container
    on a swap — the same arguments the startup engine was opened with —
    and engine_factory replaces that factory wholesale, for a host that
    builds engines its own way (tests do exactly that).
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
                     engine_kwargs=engine_kwargs,
                     engine_factory=engine_factory)
    if ready is not None:
        ready.set()
    return srv
