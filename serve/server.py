# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
server.py — the HTTP surface.

    python3 -m serve ~/models/k3.waste --port 8000

Endpoints:

    GET  /health                 liveness, plus what is loaded
    GET  /v1/models              the one model this process holds
    GET  /v1/models/{id}
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
from typing import Optional

from . import api, glmtools, xtml
from .chatfmt import ChatFormat, ChatFormatError, PlainParser
from .engine import Cancelled, Engine, EngineError
from .regions import RegionParser

SERVER_NAME = "waste"

# A body larger than this is refused before it is read into memory. A chat
# request is text; anything at this size is a base64 image, and several of
# those are a decode bomb rather than a conversation.
MAX_BODY_BYTES = 64 * 1024 * 1024


class ChatServer(ThreadingHTTPServer):
    """Threaded HTTP, one engine, one lock."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, *, engine: Engine, model_id: str,
                 api_key: Optional[str] = None,
                 default_max_tokens: int = 4096,
                 default_thinking: bool = True,
                 allow_local_images: bool = False,
                 log_requests: bool = True,
                 tmpdir: Optional[str] = None):
        super().__init__(addr, handler)
        self.engine = engine
        self.model_id = model_id
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self.default_thinking = default_thinking
        self.allow_local_images = allow_local_images
        self.log_requests = log_requests
        self.started = api.now()
        self._tmp = tmpdir or tempfile.mkdtemp(prefix="waste-serve-")
        self.tmpdir = self._tmp
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
            try:
                fmt = ChatFormat.load(engine)
            except ChatFormatError as e2:
                # Both reasons, because either one alone misleads: "no XTML"
                # reads as "wrong model" when the chat.json is simply
                # missing, and the chat.json reason alone hides that the
                # richer format was tried first.
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

    def new_parser(self, thinking: bool):
        """The reply reader for whichever format this container speaks.

        `thinking` says which channel the generation prompt left open, and
        only XTML has channels to leave open.
        """
        if self.chat_format is xtml:
            return RegionParser(in_think=thinking, in_response=not thinking,
                                markers=self.markers)
        fmt = self.chat_format
        # Which tool protocol the reply reader should own: a GLM container
        # speaks its own `<tool_call>` grammar, anything else that reaches
        # a PlainParser speaks Kimi K2's five control tokens.
        tool_parser = None
        if getattr(fmt, "tool_protocol", "") == "glm":
            tool_parser = glmtools.ToolParser()
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
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

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
        self._send_json(200, {
            "object": "list",
            "data": [api.model_object(self.server.model_id,
                                      self.server.started,
                                      self.server.model_info)],
        })

    def _model(self, model_id: str):
        if model_id != self.server.model_id:
            raise api.APIError(f"no such model: {model_id}", status=404,
                               type="not_found_error", param="model")
        self._send_json(200, api.model_object(self.server.model_id,
                                              self.server.started,
                                              self.server.model_info))

    # ---- chat -----------------------------------------------------------

    def _chat(self):
        body = self._read_body()
        srv = self.server
        engine = srv.engine

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
            parser = srv.new_parser(prompt.thinking)

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
          ready: Optional[threading.Event] = None) -> ChatServer:
    """Build the server. The caller decides whether to serve_forever."""
    # IPv6-capable when the host asks for it, without forcing it: binding
    # :: on a host with IPv6 disabled fails outright.
    if ":" in host:
        ChatServer.address_family = socket.AF_INET6
    srv = ChatServer((host, port), Handler, engine=engine, model_id=model_id,
                     api_key=api_key, default_max_tokens=default_max_tokens,
                     default_thinking=default_thinking,
                     allow_local_images=allow_local_images,
                     log_requests=log_requests)
    if ready is not None:
        ready.set()
    return srv
