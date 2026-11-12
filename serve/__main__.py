# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
__main__.py — `python3 -m serve MODEL`.

Flags mirror the CLI's where they mean the same thing (--budget, --ctx,
--threads, --cpus, --vision), because a person who has run `waste run` should not
have to learn a second vocabulary to serve the same container.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, DecimalException
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):                    # python3 serve/__main__.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "serve"

from . import api, dsml, xtml                                      # noqa: E402
from .engine import (CACHE_LFRU, CACHE_LRU,                  # noqa: E402
                     WASTE_E_ARG, WASTE_E_BUSY, WASTE_E_UNSUPPORTED,
                     Engine, EngineError, build_info, physical_ram,
                     plan_memory, usable_ram)
from .server import ModelLoadError, serve                       # noqa: E402

POLICIES = {"lfru": CACHE_LFRU, "lru": CACHE_LRU}


def parse_registry(specs: list[str]) -> dict[str, str]:
    """--models entries as {id: path}.

    `PATH[=ID]`, ID defaulting to the file name without .waste — the
    same default --model-id uses. Duplicate ids are an error rather than
    a quiet overwrite: two containers behind one name means a client's
    "load this model" sometimes loads a different one than it just
    listed, and that failure announces itself only under load.
    """
    registry: dict[str, str] = {}
    for spec in specs:
        path, _, custom = spec.partition("=")
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"--models: no such container: {p}")
        mid = custom or p.name.removesuffix(".waste")
        if mid in registry:
            raise SystemExit(
                f"--models: duplicate model id: {mid} (both "
                f"{registry[mid]} and {p})")
        registry[mid] = str(p)
    return registry


class RegistryBudgetError(Exception):
    """--models and --budget are not a pair that fits in this machine."""


def check_registry_budget(models: dict[str, str], *, budget: int,
                          usable: int) -> None:
    """Refuse a registry whose swap window does not fit.

    A swap holds two contexts at once: the new container is opened before
    the old one is closed, deliberately, so that a failed open leaves the
    server serving what it was serving. docs/SERVE.md asked the operator to
    size --budget so that moment fits and nothing checked it — and with the
    default budget of 0 the moment is far worse than "the sum of the two":
    0 means the engine sizes each context itself, up to 3/4 of
    waste_usable_ram (waste.h, waste_cfg.ram_budget_bytes), so two of them
    is ~1.5x what the process may use. That is a paging run, not a slow one.

    So, two failures, and they are different things to say:

    - No budget at all: the pair cannot be computed, and the number the
      engine would pick is not a number this process should be allowed to
      pick twice. Refused rather than guessed at.
    - An explicit budget with no room for its pair: 2 x budget over usable.
      The largest budget that fits is usable // 2, and naming that figure
      is the difference between an error and a puzzle.

    `usable` is passed in rather than measured here: the caller prints it,
    and a test needs no machine of its own. 0 means the platform would not
    say (see main) and is neither a pass nor a failure — refusing to start
    because a machine will not report its RAM is worse than the thing this
    exists to prevent. The runtime check in serve/server.py (check_room)
    still refuses a load that would exceed what is left.
    """
    if not models or not usable:
        return
    machine = api.human_bytes(usable)
    half = api.human_bytes(usable // 2)
    if not budget:
        per_ctx = api.human_bytes(usable - usable // 4)
        raise RegistryBudgetError(
            f"--models needs an explicit --budget: with 0 the engine sizes "
            f"each context itself, up to {per_ctx} of the {machine} this "
            f"process may use, and a swap holds two of them at once — the "
            f"new container is opened before the old one is closed, so that "
            f"a failed open leaves the server serving what it was serving. "
            f"Give --budget {half} or less, or drop --models and serve one "
            f"container.")
    if 2 * budget > usable:
        raise RegistryBudgetError(
            f"--budget {api.human_bytes(budget)} does not fit twice: a swap "
            f"holds the container being loaded and the one it replaces at "
            f"the same time, which is {api.human_bytes(2 * budget)} against "
            f"the {machine} this process may use. Use {half} or less, or "
            f"drop --models.")


def describe_registry(models: dict[str, str], *, budget: int, usable: int,
                      ctx: int = 0, plan=plan_memory) -> list[str]:
    """What --models will cost, as lines: one per container, then the
    arithmetic a swap performs.

    Returned rather than printed so the same lines can stand under the
    startup banner, under --plan, and under a refusal — the last being
    where they are worth most, because that is when the operator is
    choosing a --budget.
    """
    lines = []
    widest = max((len(mid) for mid in models), default=0)
    for mid, path in models.items():
        try:
            p = plan(path, ctx)
        except EngineError as e:
            lines.append(f"{mid:<{widest}}  unreadable: {e}")
            continue
        lines.append(f"{mid:<{widest}}  floor {api.human_bytes(p.floor_bytes)}"
                     f", recommended {api.human_bytes(p.recommended_bytes)}")
    if not usable:
        lines.append("this platform reports no usable-RAM figure; the pair "
                     "is not checked")
    elif not budget:
        lines.append(f"no --budget: each context sizes itself to up to "
                     f"{api.human_bytes(usable - usable // 4)} of the "
                     f"{api.human_bytes(usable)} this process may use")
    else:
        pair = 2 * budget
        if pair <= usable:
            lines.append(f"two at once: {api.human_bytes(pair)} against "
                         f"{api.human_bytes(usable)} usable — fits")
        else:
            lines.append(f"two at once: {api.human_bytes(pair)} against "
                         f"{api.human_bytes(usable)} usable — does not fit; "
                         f"the largest --budget is "
                         f"{api.human_bytes(usable // 2)}")
    return lines


# api.human_bytes, under the name the banner lines below were written
# with. One formatter for the plans, the registry lines and the 507
# refusal a swap can answer with: an operator comparing them should not be
# doing two conversions.
human = api.human_bytes


def parse_size(text: str) -> int:
    """`8G`, `512M`, `1024` (bytes). The CLI's --budget spelling."""
    t = text.strip().upper()
    mult = 1
    if t.endswith(("K", "M", "G", "T")):
        mult = {"K": 1 << 10, "M": 1 << 20,
                "G": 1 << 30, "T": 1 << 40}[t[-1]]
        t = t[:-1]
    try:
        value = Decimal(t) * mult
    except DecimalException:
        raise argparse.ArgumentTypeError(f"not a size: {text}") from None
    if not value.is_finite() or value < 0 or value > (1 << 64) - 1:
        raise argparse.ArgumentTypeError(f"size is out of range: {text}")
    return int(value)


def bounded_int(lo: int, hi: int):
    def parse(text: str) -> int:
        try:
            value = int(text, 10)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {text}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(
                f"integer must be between {lo} and {hi}: {text}")
        return value
    return parse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m serve",
        description="OpenAI-compatible server for a WASTE container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 -m serve ~/models/k3.waste
  python3 -m serve ~/models/k3.waste --port 8080 --budget 48G --vision
  python3 -m serve ~/models/k3.waste --api-key "$WASTE_KEY" --host 0.0.0.0

  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"model":"waste","messages":[{"role":"user","content":"hi"}]}'

  python3 -m serve ~/models/k3.waste --models ~/models/glm53.waste \\
        --models ~/models/deepseek41.waste=ds41 --budget 24G
        # --budget is required with --models: a swap opens the new
        # container before closing the old one, so 2 x budget has to fit
        # in RAM or the server refuses to start;
        # POST /v1/models/load {"model":"glm53"} swaps to it, unloading k3;
        # add --keep-previous to hold both resident instead — every model
        # switched to stays resident, and a load that would put the set
        # over this machine's RAM answers 507
""")
    ap.add_argument("model", help="path to the .waste container")
    ap.add_argument("--host", default="127.0.0.1",
                    help="default 127.0.0.1 — loopback only. Use 0.0.0.0 to "
                         "accept from the network, and set --api-key if you do")
    ap.add_argument("--port", type=bounded_int(0, 65535), default=8000)
    ap.add_argument("--model-id", default=None,
                    help="name reported by /v1/models (default: directory name)")
    ap.add_argument("--api-key", default=os.environ.get("WASTE_API_KEY"),
                    help="require this bearer token (default $WASTE_API_KEY)")

    g = ap.add_argument_group("engine")
    g.add_argument("--budget", type=parse_size, default=0, metavar="SIZE",
                   help="hard RAM ceiling, e.g. 48G. 0 lets the engine choose "
                        "— up to 3/4 of the RAM this process may use, so a "
                        "swap (below) needs it set explicitly")
    g.add_argument("--ctx", type=bounded_int(0, (1 << 32) - 1),
                   default=0, metavar="N",
                   help="context tokens (0 = container default)")
    g.add_argument("--threads", type=bounded_int(0, (1 << 31) - 1),
                   default=0, metavar="N",
                   help="compute threads (0 = one per core)")
    g.add_argument("--cpus", default=None, metavar="LIST",
                   help="restrict the compute pool to a cpu list, e.g. 0-5 "
                        "or 0-2,6-8; --threads 0 then means one per CPU "
                        "listed. Linux and Windows. Worth it where cores "
                        "differ: on a two-die Ryzen, six threads on one die "
                        "measured 16-25%% faster than six split across both")
    g.add_argument("--cache", choices=sorted(POLICIES), default="lfru")
    g.add_argument("--no-direct-io", action="store_true",
                   help="keep the page cache in the way. The bypass is on "
                        "by default and is what makes the reported hit "
                        "rates the engine's rather than the kernel's")
    g.add_argument("--vision", action="store_true",
                   help="load the vision tower, so requests may carry images")
    g.add_argument("--verify", action="store_true",
                   help="check every expert record's crc32 as it is read; "
                        "for a container copied or downloaded and not read "
                        "since. Costs ~5%% on Kimi-Linear, ~1%% on K3")
    g.add_argument("--usage", default=None, metavar="PATH",
                   help="learned hotlist (default <model>/usage.waste)")
    g.add_argument("--exclusive-open", action="store_true",
                   help="ask for single-process ownership of this container")

    s = ap.add_argument_group("serving")
    s.add_argument("--max-tokens", type=bounded_int(1, (1 << 32) - 1),
                   default=4096,
                   help="default cap when a request does not set one "
                        "(default 4096). Most clients never set one, and a "
                        "reply that stops at the cap is indistinguishable "
                        "from a model that stopped on its own")
    s.add_argument("--no-thinking", action="store_true",
                   help="answer without the think channel unless a request "
                        "asks for it. K3's reasoning can be most of a reply, "
                        "and at streaming speeds that is a long wait before "
                        "the first word of the answer")
    s.add_argument("--allow-local-images", action="store_true",
                   help="let requests name images by filesystem path. Off by "
                        "default: it lets any client read files the server "
                        "can reach")
    s.add_argument("--models", action="append", default=[], metavar="PATH[=ID]",
                   help="an additional container a client may switch to "
                        "with POST /v1/models/load (repeatable; id defaults "
                        "to the file name without .waste). Switching "
                        "unloads the model it replaces unless "
                        "--keep-previous. Requires --budget: a swap holds "
                        "the new container and the old one at once, so "
                        "2 x budget must fit in RAM, and refusing to start "
                        "otherwise is the point — see docs/SERVE.md")
    s.add_argument("--keep-previous", action="store_true",
                   help="keep a model resident when another is loaded. "
                        "Off by default, and deliberately: the RAM two "
                        "contexts need together is the sum of their "
                        "budgets, and on the machines this engine targets "
                        "that is the difference between working and "
                        "paging. Generation still serves only the current "
                        "model — each waste_ctx takes one caller — but "
                        "switching back to a resident one is a slot move "
                        "instead of a reopen of a multi-gigabyte "
                        "container. Every model switched to stays "
                        "resident, so the load that would put the set "
                        "over the machine's RAM is refused with 507")
    s.add_argument("--auto-register", action="store_true",
                   help="let POST /v1/models/load register a container "
                        "by pathname it has not seen before, instead of "
                        "refusing any id outside --models. Off by default: "
                        "any client that can reach this port could otherwise "
                        "make it open an arbitrary file on this machine — "
                        "the same reason --allow-local-images is off by "
                        "default. The id defaults to the file name without "
                        ".waste, or the caller's own 'model' if it gives "
                        "one; the load is still priced against --budget (or, "
                        "with none, the container's plan) and refused with "
                        "507 if it would not fit. With the flag on, a load "
                        "naming an id the server has not seen is refused, and "
                        "a load carrying a path it has not seen is the way in; "
                        "the two are distinguishable, so the endpoint can say "
                        "whether a path exists on this host. Being an opt-in to "
                        "switching, it is also an opt-in to the strict "
                        "validation --models turns on: a generation naming a "
                        "model this server does not know answers 404 rather "
                        "than being served by whatever is loaded. --usage "
                        "cannot be combined with it: a learned hotlist is "
                        "specific to one container")
    s.add_argument("--plan", action="store_true",
                   help="print the memory plan and exit without loading")
    s.add_argument("--no-log-requests", action="store_true",
                   help="silence the per-request log lines. On by default: "
                        "each line names the model that served or was "
                        "refused — on a server that swaps models, the log "
                        "is how you find out which one answered")

    args = ap.parse_args(argv)

    if args.usage and (args.models or args.auto_register):
        print("--usage cannot be used with --models or --auto-register: a "
              "learned hotlist is specific to one container", file=sys.stderr)
        return 2

    model = Path(args.model).expanduser()
    if not model.exists():
        print(f"no such container: {model}", file=sys.stderr)
        return 2
    model_id = args.model_id or model.name.removesuffix(".waste")

    registry = parse_registry(args.models)
    # What this process may use, measured once: the startup check below,
    # the banner, and every swap this server will perform all count
    # against one number rather than three readings that can disagree.
    # 0 means the platform would not say — see check_registry_budget.
    try:
        usable = usable_ram()
    except EngineError:
        usable = 0

    # Priced before anything is opened, and skipped under --plan, which is
    # the command that exists to tell an operator the numbers *before*
    # they pick a --budget.
    registry_lines = describe_registry(registry, budget=args.budget,
                                       usable=usable, ctx=args.ctx)
    if registry and not args.plan:
        try:
            check_registry_budget(registry, budget=args.budget, usable=usable)
        except RegistryBudgetError as e:
            print(f"{e}\n", file=sys.stderr)
            for line in registry_lines:
                print(f"  {line}", file=sys.stderr)
            return 2

    try:
        if args.plan:
            plan = plan_memory(str(model), args.ctx)
            ram = physical_ram()
            print(f"{build_info()}\n")
            print(f"  trunk        {human(plan.trunk_bytes)}")
            print(f"  state        {human(plan.state_bytes)}")
            print(f"  scratch      {human(plan.scratch_bytes)}")
            print(f"  min cache    {human(plan.min_expert_cache)}")
            print(f"  floor        {human(plan.floor_bytes)}")
            print(f"  recommended  {human(plan.recommended_bytes)}")
            if plan.vision_bytes:
                print(f"  vision       {human(plan.vision_bytes)} "
                      f"(only with --vision)")
            if ram:
                print(f"\n  this machine has {human(ram)}")
            if registry:
                print("\n  registry")
                for line in registry_lines:
                    print(f"    {line}")
            return 0

        engine = Engine(
            str(model),
            ram_budget_bytes=args.budget,
            ctx_tokens=args.ctx,
            n_threads=args.threads,
            cpu_list=args.cpus,
            cache_policy=POLICIES[args.cache],
            direct_io=not args.no_direct_io,
            vision=args.vision,
            verify_records=args.verify,
            usage_path=args.usage,
            exclusive_open=args.exclusive_open)
    except EngineError as e:
        print(f"{e}", file=sys.stderr)
        # Two statuses that say nothing useful on their own when --cpus is
        # what produced them, and here it usually is.
        if args.cpus and e.status == WASTE_E_ARG:
            print(f"--cpus: not a cpu list: {args.cpus}", file=sys.stderr)
        elif args.cpus and e.status == WASTE_E_UNSUPPORTED:
            print("--cpus: this platform does not bind threads to CPUs "
                  "(Linux and Windows only)", file=sys.stderr)
        elif args.exclusive_open and e.status == WASTE_E_BUSY:
            print("--exclusive-open: another process owns this container; "
                  "stop it or retry without --exclusive-open", file=sys.stderr)
        return 1

    try:
        info = engine.model_info()
        used = engine.memory_used()
        print(build_info())
        print(f"model    {model_id} — {info['arch']}, {info['n_layers']} layers, "
              f"{info['n_experts']} experts, ctx {info['ctx_max']}")
        if info.get("quant_summary"):
            print(f"quant    {info['quant_summary']}")
        print(f"memory   {human(used['floor_bytes'])} resident, "
              f"expert cache {human(used['min_expert_cache'])}")
        if args.vision:
            print("vision   on — requests may carry base64 images")
        if not args.api_key and args.host not in ("127.0.0.1", "localhost",
                                                  "::1"):
            print(f"\nWARNING: listening on {args.host} with no --api-key: "
                  f"anyone who can reach this port can use the model.",
                  file=sys.stderr)

        srv = serve(engine, host=args.host, port=args.port, model_id=model_id,
                    api_key=args.api_key,
                    default_max_tokens=args.max_tokens,
                    default_thinking=not args.no_thinking,
                    allow_local_images=args.allow_local_images,
                    log_requests=not args.no_log_requests,
                    models=registry,
                    keep_previous=args.keep_previous,
                    auto_register=args.auto_register,
                    usable_ram=usable,
                    engine_kwargs={
                        "ram_budget_bytes": args.budget,
                        "ctx_tokens": args.ctx,
                        "n_threads": args.threads,
                        "cpu_list": args.cpus,
                        "cache_policy": POLICIES[args.cache],
                        "direct_io": not args.no_direct_io,
                        "vision": args.vision,
                        "verify_records": args.verify,
                        "usage_path": args.usage,
                        "exclusive_open": args.exclusive_open,
                    })
    except (EngineError, OSError) as e:
        engine.close()
        print(f"{e}", file=sys.stderr)
        return 1

    # Said at startup, not on the first 400: an operator who learns this
    # from a client's error message has already written the client.
    #
    # The flush is not decoration. Redirected to a file, stdout is
    # block-buffered and stderr is not, so without it this warning lands
    # above the banner it is qualifying — and reads as if the container
    # failed to load at all.
    if srv.chat_error:
        sys.stdout.flush()
        print(f"\nWARNING: /v1/chat/completions is unavailable for this "
              f"container.\n  {srv.chat_error}.\n  /v1/completions is the "
              f"one generating endpoint here. `waste chat` reads the\n"
              f"  container's chat.json and is unaffected.", file=sys.stderr)
    elif srv.chat_format is xtml:
        print(f"thinking {'off by default' if args.no_thinking else 'on'}"
              f" — reasoning_effort per request")
    elif srv.chat_format is dsml:
        print(f"DSML — thinking {'off by default' if args.no_thinking else 'on'}"
              f", reasoning_effort 1-100 or low/high/max, tools, images")
    else:
        # Serving from chat.json rather than XTML. Say what this container
        # can and cannot do, in the same breath as saying it works — a
        # client that sends `tools` and gets a 400 should not be the first
        # time this is mentioned. All three are read from the container: the
        # channel and the images from chat.json, the tools from whether the
        # tokenizer carries the whole native protocol.
        think = ("a reasoning channel" if srv.chat_format.think
                 else "no reasoning channel")
        images = "images" if srv.chat_format.image else "no images"
        protocol = srv.chat_format.tool_protocol
        tools = f"{protocol} tools" if protocol else "no tools"
        print(f"chat     from {model}/chat.json — plain conversation, "
              f"{think},\n         {images}, {tools}")

    # What a client may switch to, and what that costs — on the same
    # lines as the banner rather than in a manual, because the swap
    # window is the one number here an operator can still get wrong.
    if registry:
        print(f"{'registry':<9} {', '.join(sorted(registry))}")
        for line in registry_lines:
            print(f"{'':<9} {line}")
    if registry and args.keep_previous:
        over = (f"a load that would put them over {human(usable)} answers 507"
                if usable else "the resident set is not checked on this "
                               "platform")
        print(f"{'':<9} keep-previous: every model switched to stays "
              f"resident;\n{'':<9} {over}")

    shown = args.host if ":" not in args.host else f"[{args.host}]"
    print(f"\nlistening on http://{shown}:{args.port}  "
          f"(POST {'/v1/completions' if srv.chat_error else '/v1/chat/completions'})")
    # Flush before blocking forever. Redirected to a file, stdout is
    # block-buffered, so without this `python3 -m serve … > log &` shows an
    # empty log for as long as the server runs — which reads exactly like a
    # server that failed to start.
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        srv.shutdown()
        srv.server_close()
        srv.close_engines()
        shutil.rmtree(srv.tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
