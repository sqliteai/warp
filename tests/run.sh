#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# tests/run.sh — every check we have, in one place, exiting non-zero on the
# first real failure.
#
# Written after losing time twice to checks that silently did not run: once
# to objects compiled against a stale header, once to a stale test binary.
# So this rebuilds first, states what it is about to do, and never treats a
# missing prerequisite as a pass — it says SKIP, loudly.
#
#   tests/run.sh [model.waste]
#
# Env: WASTE_REF_MODEL  container to use for the end-to-end checks
#      WASTE_REF_SRC    source weights, for the container round-trip
#      WASTE_ORACLE     logits dumped by tools/kimi_ref.py for THE SAME
#                       token ids this script uses (see IDS below) — a dump
#                       from a different prompt will look like an engine bug

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${WASTE_REF_MODEL:-$HOME/models/kimi-linear.waste}}"
SRC="${WASTE_REF_SRC:-/Volumes/WasteDisk/kimi-linear}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# python3, probed by RUNNING it rather than by looking it up. Same door as
# the missing cmp guarded below, wearing the opposite disguise: present,
# not absent.
#
# On Windows the name is usually the Microsoft Store's App Execution Alias:
# a zero-byte reparse point on PATH that `command -v` finds happily, that
# exits 49, and that prints an advert for the Store instead of running
# anything. Every `command -v python3` guard in this file was written for an
# *absent* interpreter, so it saw the tool present and let all 39 call sites
# run against something that is not one. Measured on a stock Windows 10 box,
# MinGW-w64 GCC 15.2, Python 3.13 installed and working under its own name:
# 7 passed, 7 failed, 16 skipped. The failures read "range server did not
# start", "convert.py resume", "serve suite", i.e. the engine called broken
# when what is wrong is a shim on PATH, which is the one thing line 10 says
# this suite must never do. The 16 skips were the same cause one step
# earlier: make_test_container.py is python3 too, so the synthetic container
# never built and every check needing one skipped for want of a container.
#
# Real Python installs the versionless `python` on Windows, and `py` besides,
# so a working interpreter is normally sitting right there under another
# name. Prefer a python3 that answers; otherwise shim the first name that
# does into PATH, which reaches the call sites here and the subprocesses
# (check_budget.sh, the serve suite) without editing either. With the shim
# the same box runs 44 passed, 0 failed, 13 skipped.
#
# If nothing answers, PY_MISS carries the reason and the checks SKIP loudly,
# which is what the `command -v` guard always meant to do, now for the right
# reason.
PY_MISS=
if ! python3 -c '' >/dev/null 2>&1; then
    PY_REAL=
    for _cand in python py; do
        command -v "$_cand" >/dev/null 2>&1 || continue
        "$_cand" -c '' >/dev/null 2>&1 && { PY_REAL="$_cand"; break; }
    done
    if [ -n "$PY_REAL" ]; then
        mkdir -p "$TMP/bin"
        printf '#!/usr/bin/env bash\nexec %s "$@"\n' "$PY_REAL" > "$TMP/bin/python3"
        chmod +x "$TMP/bin/python3"
        PATH="$TMP/bin:$PATH"
    else
        PY_MISS="no working python3 (the name on PATH is not an interpreter)"
    fi
fi

# Without a reference container, build a synthetic one: a few megabytes of
# deterministic noise in the real format. It cannot check the engine against
# the oracle — those logits belong to actual Kimi-Linear weights — but every
# check that compares the engine against itself works on it, which is what
# lets CI and a fresh clone run the engine at all instead of skipping it.
SYNTHETIC=0
if [ ! -d "$MODEL" ]; then
    if python3 tools/make_test_container.py "$TMP/tiny.waste" >/dev/null 2>&1; then
        MODEL="$TMP/tiny.waste"
        SYNTHETIC=1
    fi
fi

pass=0; fail=0; skip=0
ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; pass=$((pass+1)); }
no()   { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=$((fail+1)); }
sk()   { printf "  \033[33mSKIP\033[0m  %s — %s\n" "$1" "$2"; skip=$((skip+1)); }
head_() { printf "\n\033[1m%s\033[0m\n" "$1"; }

# Every bit-identity check in this file rests on `cmp`, which is diffutils —
# and a bare MSYS2/MinGW install ships neither it nor `diff`. Unguarded, an
# absent cmp made `cmp -s` false and the suite reported "expert cache changes
# results": the engine called broken when what is missing is a tool, the one
# thing line 10 says this suite must never do. Same reason curl is guarded
# below. Measured on a fresh MSYS2 UCRT64 install, where it turned eight
# passes into eight failures.
#
# 0 = identical, 1 = differ, 2 = no tool to tell with — callers must treat
# 2 as SKIP, never as either verdict.
HAVE_CMP=1
command -v cmp >/dev/null 2>&1 || HAVE_CMP=0
# cmp's own exit 2 means "could not read a file", which the callers below
# already treated as a difference — so it is folded into 1 rather than left
# to collide with the sentinel.
same() {
    [ "$HAVE_CMP" = 1 ] || return 2
    cmp -s "$1" "$2" && return 0
    return 1
}
NO_CMP="cmp not installed (diffutils)"

# Compare two logit dumps from paths that should compute the same thing.
#
#   0  within the threshold, and the argmax agrees
#   1  the argmax moved, which no amount of float noise excuses
#   2  the argmax holds but the threshold does not — which on a MoE is not
#      yet a verdict, so callers hand it to route_verdict below
LOGIT_EPS="${LOGIT_EPS:-1e-3}"
logitcmp() {
    python3 - "$1" "$2" "$LOGIT_EPS" <<'LOGITPY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
if len(a) != len(b) or not a:
    sys.exit(1)
if a.index(max(a)) != b.index(max(b)):
    sys.exit(1)
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < float(sys.argv[3]) else 2)
LOGITPY
}

# The second half of that question, for a routed model: the two paths picked
# different experts somewhere, so ask whether the first place they did was a
# decision the router itself could not make.
#
# This is not a way to excuse a difference. A top-K router turns an
# arbitrarily small arithmetic difference into a discrete one, so past the
# first flipped expert the two paths are running different weights and the
# distance between their logits stops measuring arithmetic — on K3 one flip
# at token 12 is worth 0.286 of max-abs against a 1e-3 threshold, with every
# logit before it agreeing to 1e-6. Judging that by distance means either
# failing on a tie forever or raising the threshold past the point where it
# could catch anything. tests/route_diff.py judges the flip instead, and a
# flip on a margin the reference could resolve is still a failure — a louder
# one, naming the layer.
# What a failed logit comparison actually looked like. A bare FAIL says two
# dumps differ and nothing else, and "differs" covers both a kernel that is
# slightly off and a path that computed something else entirely — the two
# need different investigations and the verdict cannot tell them apart. Half
# a session went into rediscovering that for the two checks above; the cost
# of not repeating it is four numbers.
logit_report() {
    python3 - "$1" "$2" <<'REPORTPY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
try:
    a, b = L(sys.argv[1]), L(sys.argv[2])
except OSError as e:
    print(f"        cannot read a dump: {e}")
    raise SystemExit
if len(a) != len(b):
    print(f"        different lengths: {len(a)} vs {len(b)}")
    raise SystemExit
if not a:
    print("        both dumps are empty")
    raise SystemExit
d = [abs(x - y) for x, y in zip(a, b)]
i = d.index(max(d))
n2 = sum(x * x for x in d) ** 0.5
r2 = sum(x * x for x in b) ** 0.5
print(f"        max abs {max(d):.6g} at logit {i} ({a[i]:.6g} vs {b[i]:.6g}), "
      f"rel L2 {(n2 / r2 if r2 else float('inf')):.6g}")
print(f"        argmax {a.index(max(a))} vs {b.index(max(b))}, "
      f"{sum(1 for x in d if x >= 1e-3)} of {len(d)} logits at or past 1e-3")
REPORTPY
}

route_verdict() {
    # $1 reference route trace, $2 its score trace, $3 the other path's
    # route trace, $4 the check's name
    if [ ! -s "$1" ] || [ ! -s "$2" ] || [ ! -s "$3" ]; then
        no "$4 diverges"
        printf "        beyond %s max-abs, and no route trace to say why\n" "$LOGIT_EPS"
        return
    fi
    why=$(python3 tests/route_diff.py --ref "$1" --other "$3" \
                  --scores "$2" 2>&1)
    case $? in
        2) ok "$4 (differs only past a routing tie)" ;;
        0) no "$4 diverges with the routing unchanged"
           why="same experts throughout, so the arithmetic itself moved" ;;
        *) no "$4 diverges" ;;
    esac
    printf "        %s\n" "$why"
}

# uv, with a deadline. A uv that is absent skips cleanly; a uv that is
# present and cannot work does not fail, it *hangs*, and takes the suite
# with it — MSYS2's mingw-w64-ucrt-x86_64-uv 0.12.1 opened no TCP sockets
# at all, five processes and zero connections (#36). The packaging is not
# this project's problem; a suite with no deadline on a subprocess is.
#
# Done by hand rather than with `timeout`, which is GNU coreutils and is not
# on a stock macOS. Exit 124 matches what timeout would have returned, so a
# caller that only checks for success sees a failure either way.
UV_TIMEOUT="${UV_TIMEOUT:-600}"
run_uv() {
    uv "$@" &
    uvpid=$!
    uvwaited=0
    while kill -0 "$uvpid" 2>/dev/null; do
        if [ "$uvwaited" -ge "$UV_TIMEOUT" ]; then
            kill -9 "$uvpid" 2>/dev/null
            wait "$uvpid" 2>/dev/null
            return 124
        fi
        sleep 1
        uvwaited=$((uvwaited + 1))
    done
    wait "$uvpid"
}

head_ "build"
if make -s test >/dev/null 2>&1 && make -s >/dev/null 2>&1; then
    ok "make && make test"
else
    no "build failed"
    make test 2>&1 | grep -E "error" | head -5
    exit 1
fi

head_ "model-container ownership"
# This intentionally opens two contexts at once. Always give it a tiny
# dedicated container rather than duplicating a caller-supplied K3 load.
LOCK_MODEL="$TMP/lock-test.waste"
if ! python3 tools/make_test_container.py "$LOCK_MODEL" \
        >"$TMP/lock-container.log" 2>&1; then
    sk "model-container ownership lock" \
       "could not build its synthetic container"
elif ./test_lock "$LOCK_MODEL" "$TMP" >"$TMP/lock.log" 2>&1; then
    ok "opt-in process exclusion, references, fail-open and cleanup"
else
    no "model-container ownership lock"
    head -20 "$TMP/lock.log"
fi

# ---------------------------------------------------------------- unit ----
head_ "kernels vs the reference implementations"

# Not a kernel, but it decides the default budget on every containerized
# host and runs in milliseconds against synthetic files, so it runs early
# and unconditionally. All of it runs everywhere: the reader takes its
# paths as parameters, so the cases that only fire on Linux are still
# compiled and checked on the platforms we actually develop on.
if ./test_memory "$TMP" 2>/dev/null | grep -q "^PASS"; then
    ok "cgroup-v2 limits vs the automatic budget's ceiling"
else
    no "cgroup-v2 limits"
fi

# The cpu list, in two halves with different reaches. Parsing runs
# everywhere and is the half that matters most: a typo in a cpu list is
# indistinguishable from the option not helping. Binding needs a platform
# that has the call, so it is a SKIP on macOS rather than a pass — exit 77,
# the one code this suite reads as "did not run".
if ./test_cpus parse >/dev/null 2>&1; then
    ok "cpu list parsing, including the typos that must be refused"
else
    no "cpu list parsing"
    ./test_cpus parse 2>&1 | head -3
fi

./test_cpus bind >"$TMP/cpus.log" 2>&1
case $? in
    0)  ok "compute pool binds to a cpu list ($(cat "$TMP/cpus.log"))" ;;
    77) sk "compute pool binds to a cpu list" \
           "$(sed 's/^SKIP: //' "$TMP/cpus.log" | head -1)" ;;
    *)  no "compute pool binds to a cpu list"; head -3 "$TMP/cpus.log" ;;
esac

if command -v uv >/dev/null 2>&1; then
    ./test_k3parts "$TMP/k3parts.bin" >/dev/null 2>&1
    if run_uv run --quiet --with torch --no-project python tools/k3parts_ref.py \
           "$TMP/k3parts.bin" 2>/dev/null | grep -q "^PASS"; then
        ok "K3 components (SiTU, both decay gates, AttnRes)"
    else
        no "K3 components"
    fi

    if KDA_T=24 KDA_H=4 KDA_K=32 KDA_V=32 run_uv run --quiet --with torch \
           --with fla-core --with einops --no-project python tools/kda_ref.py \
           2>/dev/null | grep -q "^PASS"; then
        ok "KDA kernel vs fla's naive_recurrent_kda"
    else
        no "KDA kernel"
    fi
else
    sk "kernel checks" "uv not installed"
fi

# -------------------------------------------------------------- download ----
head_ "download script"

# The shard downloader is the one tool whose failure modes only appear
# hours into a 1.4 TB pull, so its worker is exercised here against a local
# server instead: resume from a truncated file, skip an already-recorded
# shard without a request, and fall back to a clean restart when the server
# has no Range support.
fetch_worker() {                       # $1 = generated worker path
    { echo "DEST=$FT/dst; RAW=http://127.0.0.1:$1; STATE=\$DEST/.st; LOG=\$DEST/log"
      echo 'MAX_RETRY=2; MIN_FREE_GB=0; STAT_MODE='"$2"
      sed -n '/^cat > "\$DEST\/\.worker\.sh" <<WORKER$/,/^WORKER$/p' tools/fetch_weights.sh
    } > "$FT/gen.sh"
    bash "$FT/gen.sh"
}

# The port comes from the kernel, not from this file. Two fixed ones (8731,
# 8732) cost an afternoon on a machine already running something on the
# first: range_server.py could not bind, the worker talked to whatever else
# was listening, and the failure surfaced as "resume" and "state-file skip"
# — two checks blaming the code under test for the environment. Nothing
# here needs a *particular* port, only a free one, so ask for one and read
# back what was given. The `sleep 1` this replaces was the other half of
# the same bug: a fixed wait is either too long or, on a loaded machine,
# not long enough.
start_server() {                       # $@ = extra server args; sets PORT, RSRV
    local pf="$FT/port"
    rm -f "$pf" "$pf.tmp"
    python3 tests/range_server.py "$FT/srv" 0 --port-file "$pf" "$@" \
        >/dev/null 2>&1 &
    RSRV=$!
    PORT=
    for _ in $(seq 100); do            # 10 s, in 100 ms steps
        [ -s "$pf" ] && { PORT=$(cat "$pf"); break; }
        kill -0 "$RSRV" 2>/dev/null || break   # it died; stop waiting
        sleep 0.1
    done
    # A server that never reported is still a process, and the caller below
    # only kills the ones it was told about — CI reported exactly one
    # orphaned Python the first time this path fired.
    [ -n "$PORT" ] || { kill "$RSRV" 2>/dev/null; wait "$RSRV" 2>/dev/null; }
    [ -n "$PORT" ]
}

FT="$TMP/fetch"
mkdir -p "$FT/srv" "$FT/dst"
if stat --version >/dev/null 2>&1; then SM=gnu; else SM=bsd; fi

# The worker the downloader generates is curl, so without curl these five
# report the downloader broken when what is missing is a tool — which is the
# one thing this suite says it must never do. Debian slim and a bare MinGW
# both lack it.
#
# range_server.py and the fixture just below are python3, and an absent
# interpreter falls into the very same trap by a different door: it does not
# report a missing tool, it reports "range server did not start", which reads
# as the download script being broken. Measured on a fresh MSYS2 UCRT64
# install, where it turned all five of these into failures. So the guard
# carries the reason rather than a flag, and one variable covers both tools.
DL_MISS="$PY_MISS"
[ -n "$DL_MISS" ] || command -v curl >/dev/null 2>&1 || DL_MISS="curl not installed"

# The path goes in argv, not into the source: MSYS2 rewrites POSIX paths in
# a native program's arguments and cannot rewrite one quoted inside -c, so
# Windows Python was handed /tmp/... verbatim and could not find it.
[ -n "$DL_MISS" ] ||
python3 -c "import sys; open(sys.argv[1],'wb').write(bytes(range(256))*4096)" "$FT/srv/s.bin"

if [ -n "$DL_MISS" ]; then
    sk "a truncated shard resumes and verifies against Content-Length" "$DL_MISS"
    sk "a completed shard is skipped without a request" "$DL_MISS"
elif ! start_server; then
    no "range server did not start (resume, state-file skip not run)"
else
    head -c 400000 "$FT/srv/s.bin" > "$FT/dst/s.bin"
    fetch_worker $PORT $SM
    : > "$FT/dst/.st"
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    same "$FT/srv/s.bin" "$FT/dst/s.bin"; c=$?
    if [ $c = 2 ]; then
        sk "a truncated shard resumes and verifies against Content-Length" "$NO_CMP"
    elif [ $c = 0 ] && grep -q "^s.bin$" "$FT/dst/.st"; then
        ok "a truncated shard resumes and verifies against Content-Length"
    else
        no "resume"
    fi
    # second pass: recorded in the state file, so not even a HEAD goes out
    before=$(wc -l < "$FT/dst/log")
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    if [ "$(wc -l < "$FT/dst/log")" = "$before" ]; then
        ok "a completed shard is skipped without a request"
    else
        no "state-file skip"
    fi
    kill $RSRV 2>/dev/null; wait $RSRV 2>/dev/null
fi

# get_small, the other half of the script, and the half that had no checks.
# Its contract is that a file the repo's own listing names and the server
# does not deliver is a failure and not a 404 (#35): one attempt with the
# error discarded lost ten of eleven small files on a real download, and the
# run still printed ALL SHARDS COMPLETE and rc=0.
fetch_small() {                        # $1 = port, then MODE and file names
    local port="$1"; shift
    rm -rf "$FT/sm"; mkdir -p "$FT/sm"
    { echo "DEST=$FT/sm; RAW=http://127.0.0.1:$port; LOG=\$DEST/log"
      echo 'SMALL_RETRY=3; SMALL_BACKOFF=0'   # do not sleep through the suite
      echo 'log() { printf "%s\n" "$*" >> "$LOG"; }'
      echo 'hcurl() { curl "$@"; }'
      sed -n '/^SMALL_MISSING=""$/,/^}$/p' tools/fetch_weights.sh
      echo 'get_small "$@"'
      echo 'printf "MISSING:[%s]\n" "$SMALL_MISSING"'
    } > "$FT/gensm.sh"
    bash "$FT/gensm.sh" "$@" 2>/dev/null
}

echo hello > "$FT/srv/small.txt"

if [ -n "$DL_MISS" ]; then
    sk "a listed small file survives a transient failure" "$DL_MISS"
elif ! start_server --flaky 2; then
    no "range server did not start (small-file retry not run)"
else
    # Two 503s and then the file. The single attempt this replaces lost it
    # here, silently, and the run went on to report success.
    out=$(fetch_small $PORT req small.txt)
    if [ "$(cat "$FT/sm/small.txt" 2>/dev/null)" = hello ] &&
       echo "$out" | grep -q 'MISSING:\[\]' &&
       grep -q "retry in" "$FT/sm/log"; then
        ok "a listed small file survives a transient failure"
    else
        no "small-file retry ($out)"
    fi
    kill $RSRV 2>/dev/null; wait $RSRV 2>/dev/null
fi

# On a clean server, not the flaky one: --flaky counts per path, so a 404
# there arrives behind two 503s and the check would be measuring both at
# once.
if [ -n "$DL_MISS" ]; then
    sk "a file the repo listed and did not serve is reported, not passed over" "$DL_MISS"
elif ! start_server; then
    no "range server did not start (small-file accounting not run)"
else
    # `req` records it; `opt` — the guessed fallback list — is the case the
    # old "a 404 here is normal" comment was actually right about, and has
    # to stay silent.
    req_out=$(fetch_small $PORT req absent.txt); req_log=$(cat "$FT/sm/log" 2>/dev/null)
    opt_out=$(fetch_small $PORT opt absent.txt); opt_log=$(cat "$FT/sm/log" 2>/dev/null)
    if echo "$req_out" | grep -q 'MISSING:\[absent.txt\]' &&
       echo "$req_log" | grep -q "MISSING absent.txt" &&
       echo "$opt_out" | grep -q 'MISSING:\[\]' &&
       ! echo "$opt_log" | grep -q "MISSING"; then
        ok "a file the repo listed and did not serve is reported, not passed over"
    else
        no "small-file accounting (req=$req_out opt=$opt_out)"
    fi
    kill $RSRV 2>/dev/null; wait $RSRV 2>/dev/null
fi

if [ -n "$DL_MISS" ]; then
    sk "a server without Range support restarts the shard instead of giving up" "$DL_MISS"
elif ! start_server --no-range; then
    no "range server did not start (no-range fallback not run)"
else
    rm -f "$FT/dst/s.bin" "$FT/dst/.st" "$FT/dst/log"
    head -c 400000 "$FT/srv/s.bin" > "$FT/dst/s.bin"
    fetch_worker $PORT $SM
    : > "$FT/dst/.st"
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    same "$FT/srv/s.bin" "$FT/dst/s.bin"
    case $? in
        0) ok "a server without Range support restarts the shard instead of giving up" ;;
        1) no "no-range fallback" ;;
        *) sk "a server without Range support restarts the shard instead of giving up" "$NO_CMP" ;;
    esac
    kill $RSRV 2>/dev/null; wait $RSRV 2>/dev/null
fi

# ---------------------------------------------------------------- image ----
head_ "image"

# No model needed: the loader is checked against closed-form arithmetic on
# a solid colour it writes itself.
if ./test_image "$TMP" 2>/dev/null | grep -q "^IMAGE OK"; then
    ok "decode, patch grid, normalization and rejection of non-images"
else
    no "image loader"
fi

# ------------------------------------------------------------ container ----
head_ "container"

if [ -d "$MODEL" ]; then
    # One bank per call, and the count of records read is checked as well
    # as the count of problems. The glob used to be passed whole, so
    # `experts-L2.bin` landed where the record count goes, atoi made it 0,
    # and the check reported "0 records read, 0 problems" — a pass, from a
    # run that opened one file and read nothing out of it.
    banks=0; recs=0; bad=0
    for bank in "$MODEL"/experts-L*.bin; do
        [ -f "$bank" ] || continue
        banks=$((banks + 1))
        out=$(./test_container "$bank" 2 2>/dev/null) || { bad=1; continue; }
        n=$(printf '%s' "$out" | sed -n 's/^\([0-9]*\) records read, \([0-9]*\) problems$/\1 \2/p')
        set -- $n
        [ "${1:-0}" -gt 0 ] || bad=1
        [ "${2:-1}" -eq 0 ] || bad=1
        recs=$((recs + ${1:-0}))
    done
    if [ "$banks" -gt 0 ] && [ "$bad" = 0 ]; then
        ok "expert records read through the C structs ($recs records over $banks banks)"
    elif [ "$banks" = 0 ]; then
        no "expert record layout — no expert bank in $MODEL"
    else
        no "expert record layout"
    fi

    # $SRC defaults to Kimi-Linear's weights and $MODEL does not: point
    # WASTE_REF_MODEL at K3 without pointing WASTE_REF_SRC anywhere and this
    # compared K3's container against Kimi-Linear's safetensors and called
    # the result a converter bug. A mismatched pair is a missing
    # prerequisite, and this suite says SKIP to those.
    src_pair=$(python3 - "$MODEL" "$SRC" <<'PYSRC'
import json, os, sys
try:
    m = json.load(open(os.path.join(sys.argv[1], "manifest.json")))["config"]
    s = json.load(open(os.path.join(sys.argv[2], "config.json")))
except Exception:
    sys.exit(0)                       # let the check speak for itself
s = s.get("text_config", s)
for k in ("num_hidden_layers", "hidden_size"):
    a, b = m.get(k), s.get(k)
    if a is not None and b is not None and a != b:
        print(f"{os.path.basename(sys.argv[2])} is a different model from "
              f"this container ({k} {b} vs {a}) - set WASTE_REF_SRC")
        break
PYSRC
)
    if [ "$SYNTHETIC" = 1 ]; then
        sk "container round-trip" "synthetic container has no source weights"
    elif [ -n "$src_pair" ]; then
        sk "container round-trip" "$src_pair"
    elif [ -d "$SRC" ] && command -v uv >/dev/null 2>&1; then
        if run_uv run --quiet --with torch --no-project python tools/verify_container.py \
               --container "$MODEL" --src "$SRC" --experts 1 2>/dev/null \
               | grep -q "^PASS"; then
            ok "dequantized weights match the source"
        else
            no "container round-trip"
        fi
    elif [ ! -d "$SRC" ]; then
        sk "container round-trip" "source weights not at $SRC"
    else
        # The weights are here and uv is not, which the single message this
        # replaces reported as absent weights (#36). Someone then goes
        # looking for a download that is already on the disk.
        sk "container round-trip" \
           "uv not installed (the weights at $SRC are here; verify_container.py runs under a plain torch interpreter too)"
    fi
else
    sk "container checks" "no container at $MODEL"
fi

# The checksum is only worth writing if something reads it. Every expert
# record carries a crc32, and until it was checked on the read path a
# container that rotted after conversion answered with whatever the
# damaged bytes decoded to — which on a single flipped bit is the same
# argmax and slightly different numbers, i.e. invisible.
#
# Its own container, built here and thrown away: the check has to damage
# one, and neither the reference container nor CI's should be the victim.
CRC="$TMP/crc.waste"
if python3 tools/make_test_container.py "$CRC" >/dev/null 2>&1; then
    IDS_CRC=3,7,11,5,9,13,2,17
    damage() {                        # $1 = bank file, $2 = byte to flip
        python3 - "$1" "$2" <<'PY'
import struct, sys
p = sys.argv[1]
d = bytearray(open(p, "rb").read())
rec = struct.unpack_from("<I", d, 16)[0] * 4096     # rec_4k_blocks
# Every record, because a prompt only routes to some of the experts and a
# check that depends on which ones is a check that passes by luck.
for e in range(len(d) // rec):
    d[e * rec + int(sys.argv[2])] ^= 0x01
open(p, "wb").write(d)
PY
    }

    if ./test_forward "$CRC" "$IDS_CRC" /dev/null 0 >/dev/null 2>&1; then
        damage "$CRC"/experts-L1.bin 148          # inside the gate payload
        out=$(WASTE_VERIFY=1 ./test_forward "$CRC" "$IDS_CRC" /dev/null 0 2>&1); rc=$?
        if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q "checksum mismatch" &&
           printf '%s' "$out" | grep -qE "expert [0-9]+ of layer 1"; then
            ok "with verification on, a flipped bit is an error that names the record"
        else
            no "corrupted expert record not caught with verification on (rc=$rc)"
        fi
        # The default is off, and asserting it is the point: the same
        # container runs to completion and answers, because the checksum
        # costs ~5% and is a choice the caller makes. If this ever starts
        # failing, the default flipped without anyone deciding to.
        if ./test_forward "$CRC" "$IDS_CRC" /dev/null 0 >/dev/null 2>&1; then
            ok "by default the checksum is not read, and a damaged record answers"
        else
            no "verification is on by default — it is meant to be opt-in"
        fi
    else
        no "the undamaged synthetic container does not run"
    fi

    # A bank cut short of a whole record: the pread succeeds against the
    # record before it, so only the header identity catches this one — and
    # no WASTE_VERIFY here, deliberately. The header checks are O(1) and
    # always on; only the checksum is opt-in.
    if python3 tools/make_test_container.py "$TMP/trunc.waste" >/dev/null 2>&1; then
        python3 - "$TMP/trunc.waste/experts-L1.bin" <<'PY'
import os, sys
os.truncate(sys.argv[1], os.path.getsize(sys.argv[1]) - 4096)
PY
        out=$(./test_forward "$TMP/trunc.waste" "$IDS_CRC" /dev/null 0 2>&1); rc=$?
        if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qE "short read|record header"; then
            ok "a bank truncated mid-record is refused"
        else
            no "truncated bank not caught (rc=$rc)"
        fi
    fi
else
    sk "expert record verification" "cannot build a synthetic container"
fi

# A container with a tensor_prefix carries tensors the loader declines to
# load, and that skip used to leave `group` at zero for them while the
# row-scratch sizing divided by it. The architecture decided what that
# meant: arm64's sdiv answers 0, x86's idiv raises #DE — so `waste info` on
# K3 was an instant SIGFPE on every x86 build, Linux included, while this
# suite stayed green on a container that has no prefix and therefore no
# tensor that takes the skip (issue #10). This check is load-bearing on x86
# in any build, since the process dies; on arm64 it needs `make asan`,
# whose UBSan reports the division whatever the hardware does with it.
if python3 tools/make_test_container.py "$TMP/pfx.waste" \
        --prefix language_model. >/dev/null 2>&1; then
    if ./waste info "$TMP/pfx.waste" >/dev/null 2>&1; then
        ok "a container whose tensors are not all under its prefix loads"
    else
        no "a prefixed container does not load"
    fi
else
    sk "prefixed container" "cannot build a synthetic container"
fi

# ----------------------------------------------------------------- e2e ----
head_ "engine"

if [ -d "$MODEL" ]; then
    # The oracle fixture pins these to Kimi-Linear's vocabulary; a synthetic
    # container has 256 entries, and an id past the table is an out-of-range
    # read rather than a different answer.
    if [ "$SYNTHETIC" = 1 ]; then
        IDS=3,7,11,5,9,13,2,17,4,8,19,23,6,29,12,31
    else
        IDS=1008,10484,318,15383,387,11,316,276,10484,318,19509,387,31082,13,646,10484
    fi

    # The route traces cost one line per (token, layer) and are what turns
    # "these two disagree" into "these two disagree *here*, by *this much*".
    # Scores come from the reference run only: the chunked path does not
    # emit them, and the margin that matters is the one the reference saw.
    WASTE_DUMP_ROUTE="$TMP/seq.route" WASTE_DUMP_SCORES="$TMP/seq.scores" \
        ./test_forward "$MODEL" "$IDS" "$TMP/seq.bin" 0 >/dev/null 2>&1
    WASTE_CHUNK=1 WASTE_DUMP_ROUTE="$TMP/chunk.route" \
        ./test_forward "$MODEL" "$IDS" "$TMP/chunk.bin" 0 >/dev/null 2>&1
    logitcmp "$TMP/seq.bin" "$TMP/chunk.bin"
    case $? in
        0) ok "chunked prefill == token-at-a-time" ;;
        2) route_verdict "$TMP/seq.route" "$TMP/seq.scores" \
               "$TMP/chunk.route" "chunked prefill" ;;
        *) no "chunked prefill diverges" ;;
    esac

    # WASTE_Q8=0 makes the entire trunk resident as f32 — 8x a 4-bit one, so
    # K3's 26 GiB trunk asks for ~210 GB and no host runs this. Say so
    # instead of reporting the refusal as a divergence.
    q8_why=$(python3 - "$MODEL" <<'PY'
import json, os, subprocess, sys
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))
try:
    m = json.load(open(os.path.join(sys.argv[1], "manifest.json")))
except Exception:
    sys.exit(0)
need = 0
for t in m.get("trunk", []):
    # what src/model.c leaves out of the resident set at load, in both
    # modes, is not part of what this would allocate: the tower, which the
    # text path never touches, and embed_tokens, whose rows are pread one
    # per token — 1.41 GiB of a 7.50 GiB f32 Kimi-Linear trunk
    name = t.get("name", "")
    if name.startswith(("vision_tower.", "mm_projector.")) or \
       name.endswith("embed_tokens.weight"):
        continue
    n = 1
    for d in t.get("shape", []):
        n *= d
    need += n * 4
r = subprocess.run([WASTE, "plan", sys.argv[1], "--json"],
                   capture_output=True, text=True)
try:
    # usable, not physical: this decides whether the host can hold an f32
    # trunk, and in a container the host's RAM is not what it can hold
    phys = json.loads(r.stdout)["usable_ram_bytes"]
except Exception:
    phys = 0
if phys and need > phys // 2:
    print(f"an f32 trunk is {need / 2**30:.0f} GB of {phys / 2**30:.0f} GB of RAM")
PY
)
    if [ -n "$q8_why" ]; then
        sk "quantized trunk storage == f32 weights" "$q8_why"
    else
        WASTE_Q8=0 ./test_forward "$MODEL" "$IDS" "$TMP/f32.bin" 0 >/dev/null 2>&1
        if python3 - "$TMP/seq.bin" "$TMP/f32.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "quantized trunk storage == f32 weights"
        else no "quantized storage changes results"
        fi
    fi

    WASTE_BACKEND=cpu WASTE_DUMP_ROUTE="$TMP/cpu.route" \
        ./test_forward "$MODEL" "$IDS" "$TMP/cpu.bin" 0 >/dev/null 2>&1
    # No cmp is not a divergence: it only costs the bit-identity claim, and
    # the fp-noise bound below is still a real verdict on its own.
    if same "$TMP/seq.bin" "$TMP/cpu.bin"; then
        ok "SIMD backend bit-identical to the CPU baseline"
    else
        # a difference here is allowed to be tiny, but must be tiny — or,
        # where it is not, it must be downstream of a tie
        logitcmp "$TMP/seq.bin" "$TMP/cpu.bin"
        case $? in
            0) ok "SIMD backend matches the CPU baseline (within fp noise)" ;;
            2) route_verdict "$TMP/seq.route" "$TMP/seq.scores" \
                   "$TMP/cpu.route" "SIMD backend" ;;
            *) no "SIMD backend diverges from the CPU baseline" ;;
        esac
    fi

    WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" "$TMP/cache.bin" 0 >/dev/null 2>&1
    same "$TMP/seq.bin" "$TMP/cache.bin"
    case $? in
        0) ok "expert cache is bit-identical to no cache" ;;
        1) no "expert cache changes results" ;;
        *) sk "expert cache is bit-identical to no cache" "$NO_CMP" ;;
    esac

    # Read-ahead is on by default, so the synchronous path — the fallback,
    # and the thing every earlier measurement was made on — is the one no
    # check would otherwise run. It shipped broken for exactly one build:
    # a synchronous claim took a pin that never expired, the victim sampler
    # ran out of slots, and the forward pass answered with the experts it
    # had instead of failing.
    WASTE_IO_THREADS=0 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/sync.bin" 0 >/dev/null 2>&1
    same "$TMP/cache.bin" "$TMP/sync.bin"
    case $? in
        0) ok "read-ahead is bit-identical to synchronous reads" ;;
        1) no "read-ahead changes results" ;;
        *) sk "read-ahead is bit-identical to synchronous reads" "$NO_CMP" ;;
    esac

    # A trace-driven simulator is only worth having if it models *this*
    # cache. Before this check it did not: it kept a frequency count across
    # evictions that ec_claim resets and sampled 32 victims where EC_SAMPLE
    # is 16, and read 36.6% where the engine measured 30.4%. A simulator
    # that disagrees quietly is how a policy question gets the wrong answer
    # for a week, so the agreement is asserted rather than remembered.
    if [ "$SYNTHETIC" != 1 ] && [ -z "$PY_MISS" ]; then
        TR="$TMP/sim.trace"
        rm -f "$TR"
        eng=$(WASTE_LOOKAHEAD=0 WASTE_DUMP_ROUTE="$TR" WASTE_CACHE_MB=1024 \
              ./test_forward "$MODEL" "$(echo "$IDS" | tr ' ' ',')" /dev/null 8 2>&1 \
              | sed -n 's/.*= \([0-9.]*\)% hit.*/\1/p')
        sim=$(python3 tools/routing_stats.py simulate "$TR" --data "$MODEL" \
              --cache-gb 1.0 2>/dev/null |
              awk '/%/ && NF == 5 { v = $4 } END { gsub("%", "", v); print v }')
        if [ -n "$eng" ] && [ -n "$sim" ] && python3 -c "
import sys; sys.exit(0 if abs($eng - $sim) <= 5 else 1)" 2>/dev/null; then
            ok "trace simulator agrees with the engine's cache (${eng}% vs ${sim}%)"
        else
            no "trace simulator disagrees with the engine (${eng:-?}% vs ${sim:-?}%)"
        fi
    else
        sk "trace simulator vs the engine" "needs a real container and python3"
    fi

    # The router lookahead starts reads on a guess. The guess must never
    # reach the arithmetic: the real router stays authoritative and the
    # prefetch only decides when bytes move, so the logits cannot shift.
    WASTE_LOOKAHEAD=6 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/look.bin" 0 >/dev/null 2>&1
    same "$TMP/cache.bin" "$TMP/look.bin"
    case $? in
        0) ok "router lookahead is bit-identical to no lookahead" ;;
        1) no "router lookahead changes results" ;;
        *) sk "router lookahead is bit-identical to no lookahead" "$NO_CMP" ;;
    esac

    # A purged slot reads back as zeros, so the whole prototype rests on the
    # engine noticing before it multiplies one. This does not create memory
    # pressure — it checks that the volatile/nonvolatile traffic itself does
    # not disturb a record. Vacuously true off macOS, where the flag is a
    # no-op and says so.
    WASTE_PURGEABLE=1 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/purge.bin" 0 >/dev/null 2>&1
    same "$TMP/cache.bin" "$TMP/purge.bin"
    case $? in
        0) ok "purgeable slots are bit-identical to ordinary ones" ;;
        1) no "purgeable slots change results" ;;
        *) sk "purgeable slots are bit-identical to ordinary ones" "$NO_CMP" ;;
    esac

    # The scheduling decision moe_layer now makes from cache state, and the
    # background fill that changes that state, must both be invisible in the
    # output. They are not a numerical choice — the expert-parallel path
    # sums its per-expert results in j order exactly as the row split does —
    # but "not a numerical choice" is a property of the kernels, and the
    # moment it stops being true the engine starts giving different answers
    # depending on how warm the cache happened to be. That failure would be
    # unreproducible by construction, so it is pinned here.
    WASTE_XPAR=0 ./test_forward "$MODEL" "$IDS" "$TMP/xpar0.bin" 0 >/dev/null 2>&1
    WASTE_XPAR=1 ./test_forward "$MODEL" "$IDS" "$TMP/xpar1.bin" 0 >/dev/null 2>&1
    same "$TMP/xpar0.bin" "$TMP/xpar1.bin"
    case $? in
        0) ok "the expert-parallel path is bit-identical to the row split" ;;
        1) no "WASTE_XPAR changes the logits — the automatic choice cannot stand" ;;
        *) sk "expert-parallel bit-identity" "$NO_CMP" ;;
    esac
    same "$TMP/seq.bin" "$TMP/xpar0.bin"
    case $? in
        0) ok "and so is whichever of the two the cache state selects" ;;
        1) no "the default path matches neither WASTE_XPAR setting" ;;
        *) sk "automatic expert-parallel choice" "$NO_CMP" ;;
    esac

    WASTE_PRELOAD=0 ./test_forward "$MODEL" "$IDS" "$TMP/nofill.bin" 0 >/dev/null 2>&1
    same "$TMP/seq.bin" "$TMP/nofill.bin"
    case $? in
        0) ok "the background fill changes what is resident, not what is computed" ;;
        1) no "WASTE_PRELOAD changes the logits" ;;
        *) sk "background fill bit-identity" "$NO_CMP" ;;
    esac

    # The pool decides how many threads to wake from how big the job is, and
    # spins before parking. Neither may touch the arithmetic: the split is
    # by row. Pinned across the extremes of both knobs, because a kernel
    # that ever became order-dependent would make the answer depend on how
    # busy the machine was when it ran.
    pool_same=1
    for pcfg in "WASTE_WIDE_MIN=0" "WASTE_WIDE_MIN=1099511627776" \
                "WASTE_SPIN=0" "WASTE_SPIN_SLOW=1" "WASTE_THREADS=1" \
                "WASTE_THREADS=3"; do
        env $pcfg ./test_forward "$MODEL" "$IDS" "$TMP/pool.bin" 0 \
            >/dev/null 2>&1
        same "$TMP/seq.bin" "$TMP/pool.bin"
        case $? in
            0) ;;
            1) pool_same=0; no "$pcfg changes the logits" ;;
            *) pool_same=2 ;;
        esac
    done
    case $pool_same in
        1) ok "the thread pool's dispatch choices are bit-identical (6 combinations)" ;;
        2) sk "thread pool bit-identity" "$NO_CMP" ;;
    esac

    # kimi_ref.py computes its logits *from* a WASTE container, so an oracle
    # is only comparable against the container that produced it — and not
    # merely against its trunk width. The expert codebooks are k-means, and
    # the same seed on a different --device trains different books: splicing
    # one layer of cpu-trained books into an mps container moved the logits
    # by 1.24 max against this 1e-3 threshold. No shipped fixture can be
    # portable across conversions, so generate one from the container under
    # test — 16.9 s on Kimi-Linear, and it needs the container, not the
    # 92 GB of source shards. The fixture stays for hosts without uv, where
    # its provenance has to be checked instead (see the sidecar): a
    # cross-conversion diff reads as an engine bug and is not one.
    ORACLE="${WASTE_ORACLE:-tests/fixtures/oracle_kimilinear_16tok.bin}"
    oracle_arch=$(./waste info "$MODEL" --json 2>/dev/null | python3 -c \
        "import json,sys; print(json.load(sys.stdin).get('arch',''))" 2>/dev/null || true)
    if [ "$SYNTHETIC" = 1 ]; then
        sk "engine vs the PyTorch oracle" "synthetic container has no reference logits"
    elif [ -n "$oracle_arch" ] && [ "$oracle_arch" != "kimi-linear" ]; then
        # the ids above and the fixture's vocabulary are Kimi-Linear's
        sk "engine vs the PyTorch oracle" \
           "the oracle prompt is Kimi-Linear's, this container is $oracle_arch"
    else
        GEN=""
        if [ -z "${WASTE_ORACLE:-}" ] && command -v uv >/dev/null 2>&1; then
            run_uv run --no-project --with torch --with fla-core --with einops \
                python tools/kimi_ref.py --container "$MODEL" --prompt-ids "$IDS" \
                --tokens 0 --dump "$TMP/oracle.bin" >/dev/null 2>&1 || true
            [ -s "$TMP/oracle.bin" ] && GEN="$TMP/oracle.bin"
        fi
        oracle_why=""
        if [ -z "$GEN" ] && [ -z "${WASTE_ORACLE:-}" ] && [ -f "${ORACLE%.bin}.json" ]; then
            oracle_why=$(python3 - "$MODEL" "${ORACLE%.bin}.json" <<'PY'
import hashlib, json, os, subprocess, sys
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))
meta = json.load(open(sys.argv[2]))
name = os.path.basename(sys.argv[2])

# The codebooks, not just the trunk width. `trunk` alone let a container
# through whose books were trained on a different --device, and the sidecar
# records that axis ("converted_on") while the check ignored it: a default
# conversion on a CPU-only machine also reports Q4G/Q8G/F32, so the gate
# passed and the diff then failed by 2.77 (#36, gap 3). The comment above
# already says why that is not an engine error — the same seed on a
# different device trains different k-means books — so the gate has to see
# the books themselves. Hashing codebooks.bin covers --device and every
# other conversion knob at once, and it is under a megabyte to read.
#
# This is #7 recurring on a second axis, so the fix is the one the rotary
# fixture already uses: pin the artefact, not a property of it.
want_cb = meta.get("codebooks_sha256")
if want_cb:
    cb = os.path.join(sys.argv[1], "codebooks.bin")
    try:
        with open(cb, "rb") as f:
            got_cb = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        got_cb = None
    if got_cb and got_cb != want_cb:
        print(f"no uv to generate one, and the fixture's codebooks are not "
              f"this container's ({want_cb[:12]} vs {got_cb[:12]}) — a "
              f"cross-conversion diff is not an engine error, see {name}")
        sys.exit(0)

want = meta.get("trunk")
r = subprocess.run([WASTE, "info", sys.argv[1], "--json"],
                   capture_output=True, text=True)
try:
    got = json.loads(r.stdout)["quantization"].split("trunk ", 1)[1]
except Exception:
    sys.exit(0)                       # let the diff speak if info cannot
if want and got != want:
    print(f"no uv to generate one, and the fixture is from a {want} trunk "
          f"against this container's {got} — see {name}")
PY
)
        fi
        if [ -n "$oracle_why" ]; then
            sk "engine vs the PyTorch oracle" "$oracle_why"
        elif [ -n "$GEN" ] || [ -f "$ORACLE" ]; then
            if python3 - "$TMP/seq.bin" "${GEN:-$ORACLE}" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
            then
                if [ -n "$GEN" ]
                then ok "engine matches a PyTorch oracle built from this container"
                else ok "engine matches the shipped PyTorch oracle"
                fi
            else no "engine diverges from the oracle"
            fi
        else
            sk "oracle diff" "no fixture; regenerate with tools/kimi_ref.py --dump"
        fi
    fi

    if ./test_state "$MODEL" 2>/dev/null | grep -q "^STATE OK"; then
        ok "saved session resumes identically"
    else
        no "session state does not round-trip"
    fi

    # learned hotlist: a second run should start warmer than the first
    # 5G is Kimi-Linear's size, and on a container whose floor is above it
    # the engine refuses to open - correctly - so the run printed no cache
    # line and this read that as "the hotlist did nothing". K3's floor is
    # 29.19 GB. Raising the budget instead would make the check read a
    # couple of hundred GB twice, which is not what it is for; the hotlist
    # is exercised where it is cheap.
    hot_why=""
    if [ "$SYNTHETIC" != 1 ]; then
        hot_why=$(python3 - "$MODEL" <<'PYHOT'
import json, os, subprocess, sys
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))
r = subprocess.run([WASTE, "plan", sys.argv[1], "--json"],
                   capture_output=True, text=True)
try:
    floor = json.loads(r.stdout)["floor_bytes"]
except Exception:
    sys.exit(0)
if floor > 5 * (1 << 30):
    print(f"this container's floor is {floor / (1 << 30):.2f} GB and the "
          f"check opens at 5G, which the engine refuses")
PYHOT
)
    fi
    if [ "$SYNTHETIC" = 1 ]; then
        sk "learned hotlist" "synthetic container carries no tokenizer"
    elif [ -n "$hot_why" ]; then
        sk "learned hotlist" "$hot_why"
    else
    rm -f "$MODEL/usage.waste"
    # Read hits and misses together. Counting misses alone cannot tell a
    # perfect warm run from a run that never happened, and both come out 0:
    # a 4-bit trunk leaves enough of the 5G budget for the whole working
    # set, the hotlist lands 100%, and demanding "misses > 0" failed the
    # best result the check can produce. The empty line is the real
    # "nothing ran" signal.
    cold=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G --learn 2>&1 \
           | grep -oE "[0-9]+ hit / [0-9]+ miss" || true)
    warm=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G 2>&1 \
           | grep -oE "[0-9]+ hit / [0-9]+ miss" || true)
    rm -f "$MODEL/usage.waste"
    cold_m=${cold##*/ }; cold_m=${cold_m% miss}
    warm_m=${warm##*/ }; warm_m=${warm_m% miss}
    if [ -z "$cold" ] || [ -z "$warm" ]; then
        no "hotlist check read no cache line from the run"
    elif [ "$cold_m" -eq 0 ] && [ "$warm_m" -eq 0 ]; then
        # The cold run had nothing to teach, so neither outcome is
        # demonstrated. PASS here would be passing by luck, and this suite
        # does not treat an absent prerequisite as one.
        sk "learned hotlist" "the cold run already missed nothing"
    elif [ "$warm_m" -lt "$cold_m" ]; then
        ok "learned hotlist warms the cache ($cold_m -> $warm_m misses)"
    else
        no "hotlist did not reduce misses ($cold_m -> $warm_m)"
    fi
    fi
else
    sk "engine checks" "no container at $MODEL"
fi

# --------------------------------------------------------------- VQ4P ----
head_ "VQ4P engine (index_bits 6)"

# vq_rows_p6 — the packed-index apply PR #41's AVX-512 kernel joins — had
# no container `make check` could reach: the synthetic one was always
# index_bits 8, and a real --index-bits 6 conversion costs hours plus the
# source weights. So every green run until this arm covered VQ3R and
# nothing covered VQ4P, on any platform. Same discipline as the engine
# block above: no oracle, but the engine is compared against itself. The
# p6 accumulate itself is integer until the per-block fold, but what
# reaches the logits also went through the other dispatched kernels, so
# agreement between backends lands in the fp-noise branch — that is a real
# verdict, not a regression, and the checks below report it as such.
# Scope: this container is 4 layers / 3 MoE, so the arm bounds gross
# kernel errors; it does not and cannot bound the depth-amplified
# discontinuity mode (a real index_bits 6 container shows max diff
# ~0.58 at 27 layers with the kernel working correctly). Do not read a
# green run here as more than that. The container is a few MB and
# builds in milliseconds, so the arm runs on every host and in CI.
VQ4P="$TMP/tiny6.waste"
P6_IDS=3,7,11,5,9,13,2,17,4,8,19,23,6,29,12,31
if python3 tools/make_test_container.py --index-bits 6 "$VQ4P" \
        >/dev/null 2>&1; then
    # Same struct check as above, on records whose payload is three packed
    # bytes per row instead of four whole ones — the layout has to be the
    # one test_container reads regardless of the packing.
    banks=0; recs=0; bad=0
    for bank in "$VQ4P"/experts-L*.bin; do
        [ -f "$bank" ] || continue
        banks=$((banks + 1))
        out=$(./test_container "$bank" 2 2>/dev/null) || { bad=1; continue; }
        n=$(printf '%s' "$out" | sed -n 's/^\([0-9]*\) records read, \([0-9]*\) problems$/\1 \2/p')
        set -- $n
        [ "${1:-0}" -gt 0 ] || bad=1
        [ "${2:-1}" -eq 0 ] || bad=1
        recs=$((recs + ${1:-0}))
    done
    if [ "$banks" -gt 0 ] && [ "$bad" = 0 ]; then
        ok "VQ4P records read through the C structs ($recs records over $banks banks)"
    else
        no "VQ4P record layout"
    fi

    WASTE_DUMP_ROUTE="$TMP/p6_seq.route" WASTE_DUMP_SCORES="$TMP/p6_seq.scores" \
        ./test_forward "$VQ4P" "$P6_IDS" "$TMP/p6_seq.bin" 0 >/dev/null 2>&1
    WASTE_CHUNK=1 WASTE_DUMP_ROUTE="$TMP/p6_chunk.route" \
        ./test_forward "$VQ4P" "$P6_IDS" "$TMP/p6_chunk.bin" 0 >/dev/null 2>&1
    logitcmp "$TMP/p6_seq.bin" "$TMP/p6_chunk.bin"
    case $? in
        0) ok "VQ4P chunked prefill == token-at-a-time" ;;
        2) route_verdict "$TMP/p6_seq.route" "$TMP/p6_seq.scores" \
                         "$TMP/p6_chunk.route" "VQ4P chunked prefill" ;;
        *) no "VQ4P chunked prefill diverges" ;;
    esac

    WASTE_BACKEND=cpu WASTE_DUMP_ROUTE="$TMP/p6_cpu.route" \
        ./test_forward "$VQ4P" "$P6_IDS" "$TMP/p6_cpu.bin" 0 >/dev/null 2>&1
    if same "$TMP/p6_seq.bin" "$TMP/p6_cpu.bin"; then
        ok "VQ4P SIMD backend bit-identical to the CPU baseline"
    else
        LOGIT_EPS=1e-5 logitcmp "$TMP/p6_seq.bin" "$TMP/p6_cpu.bin"
        case $? in
            0) ok "VQ4P SIMD backend matches the CPU baseline (within fp noise)" ;;
            2) LOGIT_EPS=1e-5 route_verdict "$TMP/p6_seq.route" \
                   "$TMP/p6_seq.scores" "$TMP/p6_cpu.route" "VQ4P SIMD backend" ;;
            *) no "VQ4P SIMD backend diverges from the CPU baseline" ;;
        esac
    fi

    WASTE_CACHE_MB=512 ./test_forward "$VQ4P" "$P6_IDS" "$TMP/p6_cache.bin" 0 \
        >/dev/null 2>&1
    # same() + the 0/1/* case, not `cmp -s`: this is #42's own rule, and the
    # same call site by name as the VQ3R cache check above it. An unguarded
    # `cmp -s` on a PATH without diffutils (fresh MSYS2 UCRT64) is the one
    # false FAIL on an otherwise clean board.
    same "$TMP/p6_seq.bin" "$TMP/p6_cache.bin"
    case $? in
        0) ok "VQ4P expert cache is bit-identical to no cache" ;;
        1) no "VQ4P expert cache changes results" ;;
        *) sk "VQ4P expert cache is bit-identical to no cache" "$NO_CMP" ;;
    esac
elif [ -n "$PY_MISS" ]; then
    sk "VQ4P engine" "$PY_MISS"
else
    sk "VQ4P engine" "cannot build a synthetic index_bits 6 container"
fi

# make_test_container.py packs VQ4P indices a second time, and the arm above
# cannot tell whether it packs them the way convert.py does: it compares the
# engine against itself, so a generator that disagreed with the converter
# would produce a container the engine decodes into some other set of
# indices, both backends would decode it the same wrong way, and every check
# would pass on a layout no conversion writes. This runs the two packings
# against each other instead. torch belongs to convert.py and never to the
# engine, so it goes through uv like the other oracles.
if [ -n "$PY_MISS" ]; then
    sk "the two VQ4P packings agree" "$PY_MISS"
elif ! command -v uv >/dev/null 2>&1; then
    sk "the two VQ4P packings agree" "uv not installed (convert.py needs torch)"
elif run_uv run --no-project --with torch \
         python -m unittest tests.test_vq4p_packing 2>&1 | tail -3 | grep -q "^OK"; then
    ok "make_test_container.py packs VQ4P exactly as convert.py does"
else
    no "the two VQ4P packings disagree"
fi

# --------------------------------------------------------------- rotary ----
head_ "rotary (MLA on a model that is not NoPE)"

# Everything above this point runs on a Kimi, and every Kimi sets
# mla_use_nope — so none of it reaches rope_init or rope_apply in
# src/model.c. The rotation was absent from the engine for that reason and
# the suite stayed green throughout, which is the failure this section
# exists to stop repeating.
#
# It builds its own DeepSeek-V3-shaped container rather than using $MODEL,
# so it runs on every host and does not depend on which weights happen to be
# on disk. Nobody ships a V3 container yet — that needs the fp8 reader —
# but the shape is what the engine branches on, and the shape is free.
ROPE="$TMP/rope.waste"
RIDS=3,7,11,5,9,13,2,17,4,8,19,23,6,29,12,31
if ! python3 tools/make_test_container.py --rope --seed 0 "$ROPE" >/dev/null 2>&1; then
    sk "rotary checks" "make_test_container.py --rope did not build a container"
else
    WASTE_DUMP_ROUTE="$TMP/rope_seq.route" WASTE_DUMP_SCORES="$TMP/rope_seq.scores" \
        ./test_forward "$ROPE" "$RIDS" "$TMP/rope_seq.bin" 0 >/dev/null 2>&1
    if [ ! -s "$TMP/rope_seq.bin" ]; then
        no "the engine did not run a container without mla_use_nope"
    else
        # Same two-source shape as the Kimi oracle above: generate from the
        # reference where torch is available, fall back to the fixture where
        # it is not. This container is *generated* rather than converted, so
        # unlike that one it is byte-reproducible at seed 0 and the fixture
        # is portable — the digest below is what says so.
        RGEN=""
        if command -v uv >/dev/null 2>&1; then
            run_uv run --no-project --with torch \
                python tools/deepseek_ref.py --container "$ROPE" --ids "$RIDS" \
                --dump "$TMP/rope_ref.bin" >/dev/null 2>&1 || true
            [ -s "$TMP/rope_ref.bin" ] && RGEN="$TMP/rope_ref.bin"
        fi
        RFIX=tests/fixtures/oracle_ropesynth_16tok.bin
        rope_why=""
        if [ -z "$RGEN" ] && [ -f "${RFIX%.bin}.json" ]; then
            rope_why=$(python3 - "$ROPE" "${RFIX%.bin}.json" <<'PY'
import hashlib, json, os, sys
h = hashlib.sha256()
for n in sorted(os.listdir(sys.argv[1])):
    h.update(n.encode())
    h.update(open(os.path.join(sys.argv[1], n), "rb").read())
want = json.load(open(sys.argv[2])).get("container_sha256")
if want and h.hexdigest() != want:
    print("no uv to generate one, and make_test_container.py --rope no "
          "longer builds the container this fixture was made from — "
          "regenerate it, see " + os.path.basename(sys.argv[2]))
PY
)
        fi
        if [ -n "$rope_why" ]; then
            sk "engine vs the rotary oracle" "$rope_why"
        elif [ -n "$RGEN" ] || [ -f "$RFIX" ]; then
            if python3 - "$TMP/rope_seq.bin" "${RGEN:-$RFIX}" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
            then
                if [ -n "$RGEN" ]
                then ok "rotated MLA matches a PyTorch oracle built from this container"
                else ok "rotated MLA matches the shipped rotary fixture"
                fi
            # An engine that skips the rotation still produces finite,
            # weight-shaped logits — that is why this went unnoticed — so the
            # diff is the only thing that separates the two.
            else no "rotated MLA diverges from the oracle"
            fi
        else
            sk "engine vs the rotary oracle" \
               "no fixture; regenerate with tools/deepseek_ref.py --dump"
        fi

        # The chunked check above runs on $MODEL, which is NoPE. mla_layer is
        # per-token on both paths, so this should hold by construction — and
        # it is exactly the kind of "by construction" that a later batched
        # MLA would break silently.
        WASTE_CHUNK=1 WASTE_DUMP_ROUTE="$TMP/rope_chunk.route" \
            ./test_forward "$ROPE" "$RIDS" "$TMP/rope_chunk.bin" 0 >/dev/null 2>&1
        logitcmp "$TMP/rope_seq.bin" "$TMP/rope_chunk.bin"
        case $? in
            0) ok "chunked prefill == token-at-a-time with rotation" ;;
            2) route_verdict "$TMP/rope_seq.route" "$TMP/rope_seq.scores" \
                             "$TMP/rope_chunk.route" \
                             "chunked prefill with rotation" ;;
            *) no "chunked prefill diverges on a rotated model" ;;
        esac

        # Same model, same seed, one line of config: mla_use_nope written out
        # as false instead of omitted. A loader that tests the key for
        # presence reads that as NoPE and skips the rotation, which is the
        # pre-fix engine — so these logits have to match the ones above.
        FALSE="$TMP/rope_nopefalse.waste"
        if ! python3 tools/make_test_container.py --rope --nope false --seed 0 \
             "$FALSE" >/dev/null 2>&1; then
            sk "mla_use_nope: false rotates" "container not built"
        else
            ./test_forward "$FALSE" "$RIDS" "$TMP/rope_false.bin" 0 >/dev/null 2>&1
            same "$TMP/rope_seq.bin" "$TMP/rope_false.bin"; c=$?
            if [ $c = 2 ]
            then sk "mla_use_nope: false rotates" "$NO_CMP"
            elif [ -s "$TMP/rope_false.bin" ] && [ $c = 0 ]
            then ok "mla_use_nope: false rotates, like the same model without the key"
            else no "mla_use_nope: false was read as NoPE and skipped the rotation"
            fi
        fi

        # null is how an HF config says "no scaling" and convert.py copies it
        # verbatim, so it is the shape most containers on disk carry. It has
        # to load as plain RoPE — the same as {} and the same as no key at
        # all — rather than being read as a scaling with an unknown type.
        none_ok=1
        for shape in drop null empty; do
            dir="$TMP/rope_$shape.waste"
            rm -rf "$dir"
            python3 tools/make_test_container.py --rope --rope-scaling "$shape" \
                --seed 0 "$dir" >/dev/null 2>&1 || { none_ok=0; break; }
            ./test_forward "$dir" "$RIDS" "$TMP/rs_$shape.bin" 0 >/dev/null 2>&1
            [ -s "$TMP/rs_$shape.bin" ] || { none_ok=0; break; }
            same "$TMP/rs_drop.bin" "$TMP/rs_$shape.bin"
            case $? in
                0) ;;
                2) none_ok=2; break ;;
                *) none_ok=0; break ;;
            esac
        done
        if [ "$none_ok" = 1 ]
        then ok "rope_scaling null and {} load as plain RoPE, like no key at all"
        elif [ "$none_ok" = 2 ]
        then sk "rope_scaling null and {} load as plain RoPE" "$NO_CMP"
        else no "rope_scaling null or {} did not load as plain RoPE"
        fi
    fi

    # Shapes rope_init does not implement. Each has to be refused at load:
    # running one would apply no rotation or the wrong one, and that is not a
    # degraded answer but an unordered one.
    rope_refused() {              # <what> <expected message> <container args...>
        local what=$1 want=$2; shift 2
        local dir="$TMP/rope_bad.waste"
        rm -rf "$dir"
        if ! python3 tools/make_test_container.py --rope "$@" "$dir" >/dev/null 2>&1; then
            sk "$what is refused at load" "container not built"
        # Read into a variable rather than piping: a refused load is a
        # non-zero exit, which is the point, and under `set -o pipefail` that
        # would sink the pipeline no matter what grep found.
        else
            got=$(./test_forward "$dir" 3,7,11 "$TMP/bad.bin" 0 2>&1 || true)
            if printf '%s' "$got" | grep -q "$want"; then
                ok "$what is refused at load"
            elif [ -z "$got" ]; then
                no "$what: test_forward produced no output at all"
                printf "        expected a refusal naming %s; the process \
said nothing, which is not the same as loading it\n" "\"$want\""
            else
                no "$what loaded instead of being refused"
                printf "        expected a refusal naming %s, got: %s\n" \
                       "\"$want\"" "$(printf '%s' "$got" | head -1)"
            fi
        fi
    }

    # The rope table is a fixed WASTE_MAX_ROPE_HALF pairs.
    rope_refused "a rope slice wider than the build holds" \
                 "needs rotation" --qk-rope 132
    # Anything but yarn — linear, dynamic — reaches none of the ramp below it.
    rope_refused "an unimplemented rope_scaling type" \
                 "not implemented, only yarn" --rope-type linear
    # Unequal mscales put a ratio on cos/sin that rope_tables does not apply.
    rope_refused "rope_scaling with mscale != mscale_all_dim" \
                 "not implemented" --mscale 0.707
    # A scaling object that carries no type is not the same as no scaling.
    rope_refused "rope_scaling that carries no type" \
                 "carries no type" --rope-scaling notype
    # Present but not a boolean names no sequence order, so neither does a
    # default picked for it.
    rope_refused "mla_use_nope that is not true or false" \
                 "not true or false" --nope 1
fi

# --------------------------------------------------------------- budget ----
head_ "GLM-5.3-Flash (mHC, clamped SwiGLU, DSA indexer)"

# Same reasoning as the rotary section above: none of the checks that run on
# a Kimi reach any of these three, and all three fail *quietly*. mHC's
# parallel residual streams and its Sinkhorn projection produce
# weight-shaped logits from any mixing matrix; a clamp at 10 that never
# fires looks like a clamp that works; and an indexer that selects every
# pool is indistinguishable from one that selects the right ones until the
# context outgrows index_topk. So this builds its own GLM-shaped container
# — a few megabytes, seed 0, byte-reproducible — and diffs it against the
# PyTorch reference.
#
# The prompt length is load-bearing. index_topk 8 over pools of 4 means the
# indexer may keep two pools; sixteen tokens make four, so the selection is
# real. At twelve tokens or fewer every pool is kept, the branch degenerates
# to dense attention, and this check would pass with the selection deleted.
GLMC="$TMP/glm.waste"
GIDS=3,7,11,5,9,13,2,17,4,8,19,23,6,29,12,31
if ! python3 tools/make_test_container.py --glm --seed 0 "$GLMC" >/dev/null 2>&1; then
    sk "GLM checks" "make_test_container.py --glm did not build a container"
else
    WASTE_DUMP_DSA="$TMP/glm_dsa_eng.txt" \
        ./test_forward "$GLMC" "$GIDS" "$TMP/glm_seq.bin" 0 >/dev/null 2>&1
    if [ ! -s "$TMP/glm_seq.bin" ]; then
        no "the engine did not run a GLM container"
    else
        GGEN=""
        if command -v uv >/dev/null 2>&1; then
            WASTE_DUMP_DSA="$TMP/glm_dsa_ref.txt" \
            run_uv run --no-project --with torch --with fla-core --with einops \
                python tools/kimi_ref.py --container "$GLMC" \
                --prompt-ids "$GIDS" --tokens 0 \
                --dump "$TMP/glm_ref.bin" >/dev/null 2>&1 || true
            [ -s "$TMP/glm_ref.bin" ] && GGEN="$TMP/glm_ref.bin"
        fi
        GFIX=tests/fixtures/oracle_glmsynth_16tok.bin
        glm_why=""
        if [ -z "$GGEN" ] && [ -f "${GFIX%.bin}.json" ]; then
            glm_why=$(python3 - "$GLMC" "${GFIX%.bin}.json" <<'PYA'
import hashlib, json, os, sys
h = hashlib.sha256()
for n in sorted(os.listdir(sys.argv[1])):
    h.update(n.encode())
    h.update(open(os.path.join(sys.argv[1], n), "rb").read())
want = json.load(open(sys.argv[2])).get("container_sha256")
if want and h.hexdigest() != want:
    print("no uv to generate one, and make_test_container.py --glm no "
          "longer builds the container this fixture was made from — "
          "regenerate it, see " + os.path.basename(sys.argv[2]))
PYA
)
        fi
        if [ -n "$glm_why" ]; then
            sk "engine vs the GLM oracle" "$glm_why"
        elif [ -n "$GGEN" ] || [ -f "$GFIX" ]; then
            if [ -n "$GGEN" ]
            then gwhat="a PyTorch oracle built from this container"
            else gwhat="the shipped GLM fixture"
            fi
            logitcmp "$TMP/glm_seq.bin" "${GGEN:-$GFIX}"
            gc=$?
            if [ $gc = 0 ]; then
                ok "mHC, the clamped SwiGLU and DSA match $gwhat"
            elif [ $gc = 2 ] && [ -s "$TMP/glm_dsa_eng.txt" ] \
                             && [ -s "$TMP/glm_dsa_ref.txt" ]; then
                gwhy=$(python3 tests/dsa_diff.py \
                           --a "$TMP/glm_dsa_eng.txt" \
                           --b "$TMP/glm_dsa_ref.txt" 2>&1)
                case $? in
                    2) ok "mHC, the clamped SwiGLU and DSA vs $gwhat (differs only past a tie in the pool ranking)" ;;
                    0) no "the GLM path diverges from $gwhat with the pools unchanged"
                       logit_report "$TMP/glm_seq.bin" "${GGEN:-$GFIX}" ;;
                    *) no "the GLM path diverges from $gwhat"
                       logit_report "$TMP/glm_seq.bin" "${GGEN:-$GFIX}" ;;
                esac
                printf "        %s\n" "$gwhy"
            else
                no "the GLM path diverges from $gwhat"
                logit_report "$TMP/glm_seq.bin" "${GGEN:-$GFIX}"
                [ $gc = 1 ] && printf "        the argmax moved, which no tie excuses\n"
                [ -s "$TMP/glm_dsa_eng.txt" ] || printf "        no DSA trace, so which half moved is unanswered\n"
            fi
        else
            sk "engine vs the GLM oracle" \
               "no fixture; regenerate with tools/kimi_ref.py --dump"
        fi

        # The oracle above proves the two implementations agree. It does not
        # prove either of them *selected* anything: an indexer that keeps
        # every pool and a reference that keeps every pool agree perfectly.
        # So raise index_topk past the prompt on the same weights — the only
        # difference is the config number — and require the answer to move.
        GDENSE="$TMP/glm-dense.waste"
        if python3 tools/make_test_container.py --glm --seed 0 --index-topk 64 \
                "$GDENSE" >/dev/null 2>&1 &&
           same "$GLMC/trunk.bin" "$GDENSE/trunk.bin"; then
            ./test_forward "$GDENSE" "$GIDS" "$TMP/glm_dense.bin" 0 >/dev/null 2>&1
            if [ ! -s "$TMP/glm_dense.bin" ]; then
                no "the engine did not run the dense-attention GLM container"
            elif python3 - "$TMP/glm_seq.bin" "$TMP/glm_dense.bin" <<'PYC'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
# Anything this side of the oracle's own 1e-3 would mean the sparse path
# attended over the same positions the dense one did.
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) > 1e-2 else 1)
PYC
            then ok "the DSA indexer really narrows attention (same weights, index_topk 8 vs 64)"
            else no "index_topk 8 and index_topk 64 give the same logits — nothing was selected"
            fi
        else
            sk "DSA indexer narrows attention" "could not build the dense twin"
        fi

        # mHC keeps hc_mult residual streams and the DSA indexer keeps a
        # pool history, and both are session state. Neither is covered by
        # the round-trip above, which runs on a Kimi.
        if ./test_state "$GLMC" 2>/dev/null | grep -q "^STATE OK"; then
            ok "a GLM session round-trips its streams and its indexer pools"
        else
            no "GLM session state does not round-trip"
        fi

        # waste_model_prefill routes a GLM container through the per-token
        # path because the chunked one implements neither mHC nor the
        # indexer. This is what says the fallback is really taken.
        WASTE_CHUNK=1 ./test_forward "$GLMC" "$GIDS" "$TMP/glm_chunk.bin" 0 >/dev/null 2>&1
        same "$TMP/glm_seq.bin" "$TMP/glm_chunk.bin"
        case $? in
            0) ok "chunked prefill falls back to the decode path, bit for bit" ;;
            1) no "chunked prefill on a GLM container differs from decoding it" ;;
            *) sk "chunked prefill on a GLM container" "$NO_CMP" ;;
        esac
    fi
fi

head_ "GLM's vision tower"

# Its own container, for the same reason the GLM and rotary sections have
# one: nothing that runs on a Kimi reaches this tower, and the failure it
# would hide is the quiet kind. A patch order that disagrees with the
# rotary indices rotates every patch by someone else's position, and the
# embeddings stay finite and plausible; a bilinear resize where the release
# uses bicubic changes every pixel by a little.
GLMV="$TMP/glmvis.waste"
if ! python3 tools/make_test_container.py --glm --vision --seed 0 "$GLMV" \
        >/dev/null 2>&1; then
    sk "GLM vision checks" "make_test_container.py --glm --vision did not build one"
else
    # The patch tensor both sides see, generated once so the comparison is
    # of the tower and not of two random number generators.
    python3 - "$TMP/vpix.bin" <<'PYA'
import struct, sys
s, vals = 1, []
for _ in range(4 * 6 * 1176):
    s = (s * 1103515245 + 12345) % (1 << 32)
    vals.append(((s >> 16) & 0x7fff) / 32768.0 - 0.5)
open(sys.argv[1], "wb").write(struct.pack(f"<{len(vals)}f", *vals))
PYA
    if ! ./test_vision_glm tower "$GLMV" 4 6 "$TMP/vpix.bin" \
            "$TMP/vis_eng.bin" >/dev/null 2>&1; then
        no "the engine did not run GLM's tower"
    elif ! command -v uv >/dev/null 2>&1; then
        sk "GLM tower vs the PyTorch oracle" "uv not installed"
    else
        run_uv run --no-project --with torch python tools/glm_vision_ref.py \
            --container "$GLMV" --grid 4 6 --pixels "$TMP/vpix.bin" \
            --dump "$TMP/vis_ref.bin" >/dev/null 2>&1 || true
        if [ ! -s "$TMP/vis_ref.bin" ]; then
            sk "GLM tower vs the PyTorch oracle" "the reference did not run"
        elif python3 - "$TMP/vis_eng.bin" "$TMP/vis_ref.bin" <<'PYB'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if len(a) == len(b) and
         max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PYB
        then ok "24 blocks, 2D rope, per-head q/k norms and the gated merger match the oracle"
        else no "GLM's tower diverges from the oracle"
        fi
    fi

    # The preprocessing, separately, because it fails differently. An image
    # whose dimensions are already a multiple of patch*merge makes the
    # resize the identity, so what is left to compare is the patch order and
    # the normalization — and those are exact, not approximate.
    python3 - "$TMP/aligned.png" <<'PYC'
import struct, sys, zlib
W, H = 224, 140                      # both multiples of 28
raw = bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        raw += bytes(((x * 7 + y * 3) % 256, (x * x + y) % 256,
                      (x + y * 11) % 256))
def chunk(t, d):
    c = t + d
    return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
       + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
PYC
    if ! ./test_vision_glm pixels "$GLMV" "$TMP/aligned.png" \
            "$TMP/vis_px.bin" >"$TMP/vis_px.log" 2>&1; then
        no "GLM's image preprocessing did not run"
    elif python3 - "$TMP/vis_px.bin" "$GLMV" <<'PYD'
import json, os, struct, sys
# The reference's own reshape/permute/expand, spelled out: reshape to
# (C, gh/m, m, patch, gw/m, m, patch), permute to block-major, then repeat
# the patch across the temporal axis. If the engine's order disagrees the
# difference is not small, it is a different image.
W, H, patch, merge, T = 224, 140, 14, 2, 2
gh, gw = H // patch, W // patch
cfg = json.load(open(os.path.join(sys.argv[2], "vision.json")))
mean, std = cfg["image_mean"], cfg["image_std"]
px = [[[0.0] * W for _ in range(H)] for _ in range(3)]
for y in range(H):
    for x in range(W):
        v = ((x * 7 + y * 3) % 256, (x * x + y) % 256, (x + y * 11) % 256)
        for c in range(3):
            px[c][y][x] = (v[c] / 255.0 - mean[c]) / std[c]
want = []
for by in range(gh // merge):
    for bx in range(gw // merge):
        for dy in range(merge):
            for dx in range(merge):
                py, pxi = by * merge + dy, bx * merge + dx
                row = []
                for c in range(3):
                    one = [px[c][py * patch + j][pxi * patch + i]
                           for j in range(patch) for i in range(patch)]
                    for _ in range(T):
                        row += one
                want.append(row)
b = open(sys.argv[1], "rb").read()
got = struct.unpack(f"<{len(b)//4}f", b)
flat = [v for r in want for v in r]
sys.exit(0 if len(got) == len(flat) and
         max(abs(a - c) for a, c in zip(got, flat)) < 2e-4 else 1)
PYD
    then ok "patches are block-major over merge blocks, with the temporal slot copied"
    else no "GLM's patch order or normalization is not the release's"
    fi
fi

head_ "Qwen3.8-Flash-Next (GDN, QSA, HyperConnection, PLE)"

# A Qwen container is not a Kimi container with different numbers: the
# recurrence is Gated DeltaNet rather than KDA, attention is sparse over
# the original K/V rather than over a latent, the residual is four streams,
# and one layer reads an n-gram embedding a row at a time off the trunk.
# None of that is reachable from any other fixture, so this builds its own
# — a few hundred kilobytes, seed-deterministic, format v0 with unchanged
# WEXP records.
QWENC="$TMP/qwen.waste"
if ! python3 tools/make_test_container.py --qwen "$QWENC" >/dev/null 2>&1; then
    sk "Qwen checks" "make_test_container.py --qwen did not build a container"
else
    # Format first: a fixture that quietly stopped being a real container
    # would make every check below vacuous.
    qman=$(python3 -c "import json;m=json.load(open('$QWENC/manifest.json'));print(m['format_version'], m['arch'])" 2>/dev/null)
    qmagic=$(python3 -c "import struct;print(struct.unpack('<I', open('$QWENC/experts-L0.bin','rb').read(4))[0] == 0x50584557)" 2>/dev/null)
    qinfo=$(./waste info "$QWENC" 2>&1)
    ./test_forward "$QWENC" 3,7,11 "$TMP/qwen_seq.bin" 0 >"$TMP/qwen_fwd.log" 2>&1
    if [ "$qman" = "0 qwen4_exp_text" ] && [ "$qmagic" = "True" ] &&
       printf '%s' "$qinfo" | grep -q "qwen4_exp_text" &&
       [ -s "$TMP/qwen_seq.bin" ]; then
        ok "the Qwen fixture is format v0 with WEXP records and loads as qwen4_exp_text"
    else
        no "the Qwen fixture did not load (manifest='$qman' wexp=$qmagic)"
        printf '%s\n' "$qinfo" | tail -5
    fi

    # QSA pools four keys into a block. Three tokens leave the block open
    # and only the tail; the fourth closes it. Reported by test_forward so
    # the boundary is observable rather than inferred from the logits.
    q3=$(./test_forward "$QWENC" 3,7,11 /dev/null 0 2>&1 | grep '^qsa_layer')
    q4=$(./test_forward "$QWENC" 3,7,11,5 /dev/null 0 2>&1 | grep '^qsa_layer')
    if printf '%s' "$q3" | grep -q 'blk 0 tail 3' &&
       printf '%s' "$q4" | grep -q 'blk 1 tail 0'; then
        ok "QSA closes a 4-token block on the fourth token"
    else
        no "QSA block pooling is wrong (3 tok: '$q3'; 4 tok: '$q4')"
    fi

    # Chunked prefill against sequential decode, the check that has caught
    # every state bug in this engine: the two share no code above the layer
    # loop and must agree bit for bit.
    WASTE_CHUNK=1 ./test_forward "$QWENC" 3,7,11 "$TMP/qwen_chunk.bin" 0 \
        >/dev/null 2>&1
    if [ ! -s "$TMP/qwen_chunk.bin" ]; then
        no "chunked prefill did not run on a Qwen container"
    elif cmp -s "$TMP/qwen_seq.bin" "$TMP/qwen_chunk.bin"; then
        ok "chunked prefill is bit-identical to sequential decode"
    else
        no "Qwen chunked prefill disagrees with sequential decode"
    fi

    # The hyper-state dump is what the container-native oracle diffs
    # against, so its shape is checked on its own: a dump of the wrong
    # length would make that comparison read the wrong layer.
    if [ -n "$PY_MISS" ]; then
        sk "Qwen hyper-state dump" "$PY_MISS"
    elif python3 tests/test_qwen_dump.py >/dev/null 2>&1; then
        ok "WASTE_DUMP_HIDDEN writes all four residual streams after every layer"
    else
        no "WASTE_DUMP_HIDDEN is missing or the wrong size on a Qwen container"
    fi

    # A budget under the floor is refused rather than swapped into, and the
    # floor is computed from Qwen's own state keys — GDN's S, the conv
    # rings, the BF16 K/V and the raw index keys.
    qsmall=$(./waste run "$QWENC" x --budget 1 2>&1 || true)
    qhuge=$(./waste run "$QWENC" x --ctx 8000000 --budget 8M 2>&1 || true)
    if printf '%s' "$qsmall" | grep -qi "budget" &&
       printf '%s' "$qhuge" | grep -qi "budget"; then
        ok "a RAM budget under the Qwen floor is refused, not swapped into"
    else
        no "an under-floor Qwen budget was accepted"
    fi

    # top_k comes from the normalised key, and from HF's spelling when a
    # container was written without it: planning for 0 experts would
    # under-size the scratch that many pointers are cut from.
    if [ -n "$PY_MISS" ]; then
        sk "Qwen top_k alias" "$PY_MISS"
    else
        cp -R "$QWENC" "$TMP/qwen-alias.waste"
        python3 - "$TMP/qwen-alias.waste" <<'PYQ'
import json, sys
p = sys.argv[1] + "/manifest.json"
m = json.load(open(p))
m["config"].pop("num_experts_per_token", None)   # leave only HF's spelling
json.dump(m, open(p, "w"), indent=1)
PYQ
        if [ "$(./waste plan "$QWENC" --json)" = \
             "$(./waste plan "$TMP/qwen-alias.waste" --json)" ]; then
            ok "plan reads top_k from num_experts_per_tok when the canonical key is absent"
        else
            no "Qwen plan disagrees with itself over the top_k alias"
        fi
    fi

    # Refusals. Each of these is a container the engine could open and read
    # wrongly rather than fail on, which is the whole reason cfg_sane
    # bounds them: an out-of-range n-gram overruns a fixed context array, a
    # second indexer KV head is a shape nothing here implements, and a
    # missing layer_types reads as "every layer is GDN" — plausible,
    # answer-changing, and invisible.
    qwen_refused() {                  # <what> <python-edit>
        local what="$1" edit="$2" dir="$TMP/qwen-bad.waste"
        rm -rf "$dir"; cp -R "$QWENC" "$dir"
        python3 -c "$edit" "$dir" || { no "$what (fixture edit failed)"; return; }
        if ./waste info "$dir" >/dev/null 2>&1; then
            no "$what was accepted"
        else
            ok "$what is refused"
        fi
    }
    if [ -n "$PY_MISS" ]; then
        sk "Qwen container refusals" "$PY_MISS"
    else
        qwen_refused "an n-gram order past the engine's fixed context" \
            'import json,sys;p=sys.argv[1]+"/manifest.json";m=json.load(open(p));m["config"]["ngram_size"]=99;json.dump(m,open(p,"w"))'
        qwen_refused "an indexer with more than one KV head" \
            'import json,sys;p=sys.argv[1]+"/manifest.json";m=json.load(open(p));m["config"]["indexer_kv_heads"]=2;json.dump(m,open(p,"w"))'
        qwen_refused "a container that does not say which layers are attention" \
            'import json,sys;p=sys.argv[1]+"/manifest.json";m=json.load(open(p));m["config"].pop("layer_types");json.dump(m,open(p,"w"))'
        qwen_refused "a layer_types shorter than num_hidden_layers" \
            'import json,sys;p=sys.argv[1]+"/manifest.json";m=json.load(open(p));m["config"]["layer_types"]=m["config"]["layer_types"][:1];json.dump(m,open(p,"w"))'
    fi

    # The isolated ops against an independent PyTorch reference written
    # from the published equations, at the official geometry as well as at
    # toy sizes.
    if ! command -v uv >/dev/null 2>&1; then
        sk "Qwen components" "uv not installed"
    elif ./test_qwenparts "$TMP/qwenparts.bin" >/dev/null 2>&1 &&
         run_uv run --quiet --with torch --no-project python \
             tools/qwenparts_ref.py "$TMP/qwenparts.bin" 2>/dev/null |
             grep -q "^PASS"; then
        ok "PLE hashing, HyperConnection, GDN, QSA and the top-k router match the reference"
    else
        no "a Qwen component diverges from tools/qwenparts_ref.py"
    fi

    # The container-native oracle: the same container read by a PyTorch
    # implementation of the same forward pass. Routes must match exactly
    # and the argmax must match; the residual is gated at what this fixture
    # measures, see docs/QWEN.md.
    if ! command -v uv >/dev/null 2>&1; then
        sk "container-native Qwen oracle" "uv not installed"
    else
        qout=$(run_uv run --quiet --with torch --no-project python \
                   tests/test_qwen_container_ref.py 2>&1); qrc=$?
        case "$qrc" in
        0)   ok "the engine matches the container-native oracle (routes exact, argmax equal)" ;;
        77)  sk "container-native Qwen oracle" "torch not installed" ;;
        124) sk "container-native Qwen oracle" "uv timed out" ;;
        *)   no "the engine diverges from the container-native Qwen oracle"
             printf '%s\n' "$qout" | tail -12 ;;
        esac
    fi

    # The oracle's own gates, tested against synthetic disagreements: a
    # comparison that cannot fail proves nothing about the runs it passes.
    if [ -n "$PY_MISS" ]; then
        sk "Qwen route near-tie gate" "$PY_MISS"
    elif python3 tests/test_qwen_compare_oracle.py >/dev/null 2>&1; then
        ok "the route comparison rejects a real reorder and accepts only a near tie"
    else
        no "the Qwen route near-tie gate does not reject a wrong expert"
    fi

    if [ -n "$PY_MISS" ]; then
        sk "Qwen streaming oracle" "$PY_MISS"
    elif python3 tests/test_qwen_oracle.py >/dev/null 2>&1; then
        ok "the official-weights oracle streams experts and n-gram rows rather than holding them"
    else
        no "tools/qwen_ref.py would materialize what it is meant to stream"
    fi
fi

# The tokenizer half that needs no weights always runs; the parity half
# needs the pinned checkpoint and the `tokenizers` package and says so.
if [ -n "$PY_MISS" ]; then
    sk "Qwen tokenizer" "$PY_MISS"
else
    if command -v uv >/dev/null 2>&1; then
        qtok=$(run_uv run --quiet --with tokenizers --no-project python \
                   tests/test_qwen_tok.py 2>&1); qtrc=$?
    else
        qtok=$(python3 tests/test_qwen_tok.py 2>&1); qtrc=$?
    fi
    case "$qtrc" in
    0)   ok "the C tokenizer matches the pinned Qwen release, numbers included" ;;
    77)  sk "Qwen tokenizer parity" "no pinned checkpoint (WASTE_QWEN_SRC) or no tokenizers package" ;;
    124) sk "Qwen tokenizer parity" "uv timed out" ;;
    *)   no "Qwen tokenizer parity"
         printf '%s\n' "$qtok" | tail -12 ;;
    esac
fi

head_ "RAM budget"

# The default budget is the one path check_budget.sh cannot reach, because
# it always passes --budget. With no flag the engine chooses, and that
# choice is all that stands between a model whose recommendation exceeds
# the machine — K3 asks for 80.63 GB — and a swap storm. So assert the
# rule, not a number, and it holds on any host: the engine steps down a
# whole token working set at a time and takes the largest of
# floor + 3x, 2x, 1x that fits under 3/4 of the RAM this process may use,
# or the floor when not even one multiple does, less at most one expert
# record of slot rounding. Filling the cap instead is what put a 27 GB
# cache on a 64 GB machine and cost 8x throughput — docs/LEARNED.md §16.
default_budget() {
    # No interpreter, no measurement. 77 is check_budget.sh's
    # "unmeasurable", and it is a SKIP at every call site rather than a
    # verdict about the container.
    [ -n "$PY_MISS" ] && return 77
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

# subprocess does not go through the shell, so it does not inherit Git-Bash's
# habit of resolving a bare name to the .exe next to it.
WASTE = os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else ""))

def j(*a):
    r = subprocess.run([WASTE, *a, "--json"], capture_output=True, text=True)
    return json.loads(r.stdout)

plan, info = j("plan", sys.argv[1]), j("info", sys.argv[1])
# From the engine rather than os.sysconf, which does not exist in Windows
# CPython at all and would read the host's RAM inside a container anyway.
# This is the same number the engine sized itself against, so what the
# check still tests is the rule — floor + the largest whole working set
# under 3/4 of it — and not the RAM reading, which has its own platform
# code and no business being written twice. It is a capacity and not a
# pressure reading, so it is the same in this process and in the `info`
# one below; a ceiling that moved between the two would make this check
# fail as an engine bug on any busy machine.
phys = plan["usable_ram_bytes"]
if not phys:
    print("usable RAM unknown on this host")
    sys.exit(0)
cap = phys - phys // 4
# what the engine actually holds: the plan's mandatory parts plus the
# cache it really allocated, which is what `info` reports
held = plan["floor_bytes"] - plan["min_expert_cache"] + info["expert_cache_bytes"]
# the engine reports it: recommended_bytes is capped at the container's whole
# expert set, so on a merged container it is no longer floor + 3 * this
ws = plan["working_set_bytes"]
bank = plan.get("bank_bytes", 0)
want = plan["floor_bytes"]
# The ladder the engine walks: as many whole working sets as the machine
# allows, starting from however many cover the container's entire expert set
# — beyond which a cache cannot be improved by making it larger — and never
# a cache bigger than that set. Three used to be the top rung, and on a
# machine with room it left the rest of it unused.
kmax = max(3, -(-bank // ws) if (bank and ws) else 3)
for k in range(kmax, 0, -1):
    got = min(ws * k, bank) if bank else ws * k
    if plan["floor_bytes"] + got <= cap:
        want = plan["floor_bytes"] + got
        break
rec = plan["min_expert_cache"] // (2 * info["top_k"]) if info["top_k"] else 0
G = 1 << 30
print(f"{held/G:.2f} GB held, ceiling {want/G:.2f} GB, usable {phys/G:.2f} GB")
sys.exit(0 if want - rec - 1 <= held <= want else 1)
PY
}

# params_total is the number that ends up in a model card, and it is
# derived rather than stored: the routed experts, three matrices each and
# each as wide as the expert's input, plus the trunk that runs on every
# token. In a latent MoE that width is the latent, not the hidden — K3
# reported 5.44 T instead of 2.72 T for exactly one wrong field. Mirroring
# the engine's arithmetic here would only prove this script and the engine
# agree, so the expert count is also weighed against the bytes on disk: at
# 3 bits per weight the experts have to fit their bank, give or take one
# fp16 scale per output row and the record's 4 KiB alignment.
params_rule() {
    # No interpreter, no measurement. 77 is check_budget.sh's
    # "unmeasurable", and it is a SKIP at every call site rather than a
    # verdict about the container.
    [ -n "$PY_MISS" ] && return 77
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json"))
r = subprocess.run([os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else "")), "info", d, "--json"], capture_output=True, text=True)
info = json.loads(r.stdout)
c, lay = man["config"], man["layers"]

width = c.get("routed_expert_hidden_size") or c["hidden_size"]
inter = c["moe_intermediate_size"]
per_expert = 3 * width * inter
moe_layers = c["num_hidden_layers"] - c.get("first_k_dense_replace", 0)

# the trunk, as the engine counts it: the language model only, and a token
# reads one row of the embedding table rather than all of it
pref = man.get("tensor_prefix", "")
trunk = trunk_active = 0
for t in man["trunk"]:
    if pref and not t["name"].startswith(pref):
        continue
    n = 1
    for s in t["shape"]:
        n *= s
    trunk += n
    if "embed_tokens.weight" not in t["name"]:
        trunk_active += n

total = per_expert * c["num_experts"] * moe_layers + trunk
active = per_expert * info["top_k"] * moe_layers + trunk_active

bits = man["expert_quant"]["bits_per_weight"]
rec = lay[next(iter(lay))]
on_disk = rec["bytes"] // rec["experts"]
lo = per_expert * bits // 8
hi = lo + 2 * (2 * inter + width) + 4096          # scales, then alignment

def h(x):
    if x >= 1e12: return f"{x/1e12:.2f} T"
    if x >= 1e9:  return f"{x/1e9:.2f} B"
    return f"{x/1e6:.2f} M"

print(f"{h(total)} total, {h(active)} active, {h(trunk)} trunk, "
      f"{on_disk/(1<<20):.2f} MiB/expert on disk")
sys.exit(0 if (moe_layers == len(lay)
               and info["params_total"] == total
               and info["params_active"] == active
               and lo <= on_disk <= hi) else 1)
PY
}

if [ -d "$MODEL" ]; then
    if ./waste plan "$MODEL" >/dev/null 2>&1; then ok "waste plan"; else no "waste plan"; fi
    # capture first: `set -o pipefail` would otherwise propagate the
    # deliberate non-zero exit of the command under test
    refusal=$(./waste run "$MODEL" x --budget 1M 2>&1 || true)
    if printf '%s' "$refusal" | grep -q "below the model's floor"; then
        ok "a budget under the floor is refused, not swapped into"
    else
        no "under-floor budget not refused"
    fi

    out=$(default_budget "$MODEL" 2>/dev/null); rc=$?
    case $rc in
        0)  ok "no --budget picks a ceiling the machine can hold ($out)" ;;
        77) sk "the default budget rule" "$PY_MISS" ;;
        *)  no "default budget off the rule (${out:-no output})" ;;
    esac
    # A container from a future layout must be refused, not read against the
    # old rules — the field was written from the first converter and read by
    # nobody until it was wired up.
    FV=$(mktemp -d)
    python3 - "$MODEL/manifest.json" "$FV/manifest.json" <<'PYFV'
import json, sys
m = json.load(open(sys.argv[1])); m["format_version"] = 999
json.dump(m, open(sys.argv[2], "w"))
PYFV
    out=$(./waste info "$FV" 2>&1); rc=$?
    python3 - "$MODEL/manifest.json" "$FV/manifest.json" <<'PYFV'
import json, sys
m = json.load(open(sys.argv[1])); m.pop("format_version", None)
json.dump(m, open(sys.argv[2], "w"))
PYFV
    out2=$(./waste info "$FV" 2>&1); rc2=$?
    rm -rf "$FV"
    if [ "$rc" -ne 0 ] && [ "$rc2" -ne 0 ] &&
       printf '%s' "$out"  | grep -q "format version mismatch" &&
       printf '%s' "$out2" | grep -q "format version missing"; then
        ok "a container from another format version is refused"
    else
        no "format_version not enforced (rc=$rc rc2=$rc2)"
    fi

    if [ -n "${WASTE_SANITIZED:-}" ]; then
        sk "peak RSS inside the budget" "sanitizer shadow memory makes RSS meaningless"
    elif [ "$SYNTHETIC" = 1 ]; then
        sk "peak RSS inside the budget" "needs a tokenizer to drive the CLI"
    else
        # Not piped straight into grep: that discards the exit status, and
        # 77 (could not measure) has to be told from a real overrun. On
        # Windows the unmeasurable case was reported as an overrun (#36).
        bud=$(tests/check_budget.sh "$MODEL" 2>/dev/null); brc=$?
        if printf '%s' "$bud" | grep -q "^BUDGET OK"; then
            ok "peak RSS stays inside the configured budget"
        elif [ "$brc" = 77 ]; then
            sk "peak RSS inside the budget" \
               "$(printf '%s' "$bud" | sed -n 's/^BUDGET UNMEASURABLE: //p')"
        else
            no "peak RSS exceeded the budget"
        fi
    fi
else
    sk "budget checks" "no container at $MODEL"
fi

# The small model cannot catch budget accounting that is wrong in
# proportion to the model: K3 overran by 2-3 GB on scratch that Kimi-Linear
# sizes in single megabytes. Run the same check against K3 when it is here.
#
# None of it survives a sanitizer build, and for two different reasons: RSS
# is meaningless next to ASan's shadow memory, and ASan's allocator refuses
# the 27 GB mapping the trunk needs at all, so anything that *opens* K3
# fails rather than measuring anything. Both mean SKIP. This only bites on
# a machine that has the K3 container — CI does not, which is why `make
# asan` stayed green there while failing on the developer's own laptop.
BIG="${BIG_MODEL:-$HOME/models/k3.waste}"
if [ -n "${WASTE_SANITIZED:-}" ] && [ -f "$BIG/manifest.json" ]; then
    sk "every K3 check" "sanitizer cannot open a 27 GB trunk"
    BIG=/nonexistent-under-sanitizer
fi
if [ -f "$BIG/manifest.json" ]; then
    bud=$(tests/check_budget.sh "$BIG" 32 long 2>/dev/null); brc=$?
    if printf '%s' "$bud" | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the budget on K3 too"
    elif [ "$brc" = 77 ]; then
        sk "peak RSS inside the budget on K3" \
           "$(printf '%s' "$bud" | sed -n 's/^BUDGET UNMEASURABLE: //p')"
    else
        no "peak RSS exceeded the budget on K3"
    fi

    # K3 is the only model here whose recommendation can exceed the
    # machine, so it is the one that makes the step-down bite at all: on a
    # 64 GB host the default lands on floor + 1x working set = 46.24 GB,
    # rather than the floor + 3x = 80.63 GB asked for. On a host large
    # enough to hold the recommendation this still passes — it checks the
    # rule, not the clamp.
    out=$(default_budget "$BIG" 2>/dev/null); rc=$?
    case $rc in
        0)  ok "no --budget is capped to the machine on K3 ($out)" ;;
        77) sk "the default budget rule on K3" "$PY_MISS" ;;
        *)  no "default budget not capped on K3 (${out:-no output})" ;;
    esac
else
    sk "K3 budget check" "no container at $BIG"
fi

# ----------------------------------------------------------- parameters ----
head_ "parameter counts"

# The other two things `info` says about a container were string constants
# until a second model existed, and both were wrong about it: K3 announced
# itself as kimi-linear with a Q8G trunk, being neither. They are derived
# now, so check them against the manifest — the architecture K3 records
# under `_outer`, and every trunk format the language model actually uses.
info_rule() {
    # No interpreter, no measurement. 77 is check_budget.sh's
    # "unmeasurable", and it is a SKIP at every call site rather than a
    # verdict about the container.
    [ -n "$PY_MISS" ] && return 77
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json"))
r = subprocess.run([os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else "")), "info", d, "--json"], capture_output=True, text=True)
info = json.loads(r.stdout)
c = man["config"]

hf = ((c.get("_outer", {}).get("architectures") or c.get("architectures")
       or [""]))[0]
# The same mapping waste_model_get_info makes, because that is what is
# being checked: a family the engine names and this rule does not would
# fail here for spelling rather than for describing the wrong container.
arch = ("kimi-k3" if "KimiK3" in hf else "kimi-linear" if "KimiLinear" in hf
        else "glm5-next" if "Glm5Next" in hf
        else "qwen4_exp_text" if "Qwen4Exp" in hf
        else hf or "unknown")     # a container that names nothing gets that

NAMES = {0: "F32", 1: "F16", 2: "Q8G", 3: "Q4G", 7: "Q3G"}
pref = man.get("tensor_prefix", "")
used = {t["fmt"] for t in man["trunk"]
        if not pref or t["name"].startswith(pref)}
quant = (f"experts VQ{man['expert_quant']['stages']}R, trunk "
         + "/".join(NAMES[f] for f in (7, 3, 2, 1, 0) if f in used))

print(f"{info['arch']}, {info['quantization']}")
sys.exit(0 if info["arch"] == arch and info["quantization"] == quant else 1)
PY
}

if [ -d "$MODEL" ]; then
    out=$(info_rule "$MODEL" 2>/dev/null); rc=$?
    case $rc in
        0)  ok "info describes the container it opened ($out)" ;;
        77) sk "info describes the container it opened" "$PY_MISS" ;;
        *)  no "info describes something else (${out:-no output})" ;;
    esac
    out=$(params_rule "$MODEL" 2>/dev/null); rc=$?
    case $rc in
        0)  ok "params_total is what the container holds ($out)" ;;
        77) sk "params_total against the rule" "$PY_MISS" ;;
        *)  no "params_total off the rule (${out:-no output})" ;;
    esac
else
    sk "parameter counts" "no container at $MODEL"
fi

# K3 is the model both rules were wrong about, and the only latent MoE
# here: without it the expert width and the hidden never differ, and the
# two descriptive fields pass on the constants they used to be.
if [ -f "$BIG/manifest.json" ]; then
    out=$(info_rule "$BIG" 2>/dev/null); rc=$?
    case $rc in
        0)  ok "info names K3 and its trunk, not the model before it ($out)" ;;
        77) sk "info names K3 and its trunk" "$PY_MISS" ;;
        *)  no "info describes something else on K3 (${out:-no output})" ;;
    esac
    out=$(params_rule "$BIG" 2>/dev/null); rc=$?
    case $rc in
        0)  ok "params_total counts K3's experts at the latent ($out)" ;;
        77) sk "params_total against the rule on K3" "$PY_MISS" ;;
        *)  no "params_total off the rule on K3 (${out:-no output})" ;;
    esac
else
    sk "K3 parameter counts" "no container at $BIG"
fi

# --------------------------------------------------------------- vision ----
head_ "vision preprocessing"

# The tower is checked against its oracle on *random* pixels, so nothing in
# the suite ever looked at what a real image is normalized by. It was the
# CLIP convention for a day, against a release that states mean = std = 0.5
# in preprocessor_config.json — a wrong constant that every existing check
# was structurally blind to. Assert the container against the source.
vision_norm() {
    # No interpreter, no measurement. 77 is check_budget.sh's
    # "unmeasurable", and it is a SKIP at every call site rather than a
    # verdict about the container.
    [ -n "$PY_MISS" ] && return 77
    python3 - "$1" "$2" <<'PY'
import json, os, sys
cont, src = sys.argv[1], sys.argv[2]
vj = os.path.join(cont, "vision.json")
pp = os.path.join(src, "preprocessor_config.json")
if not os.path.exists(vj) or not os.path.exists(pp):
    sys.exit(2)                       # nothing to compare: caller skips
v = json.load(open(vj))
m = json.load(open(pp)).get("media_proc_cfg", {})
bad = [k for k in ("image_mean", "image_std")
       if m.get(k) is not None and v.get(k) != m[k]]
print(f"mean {v.get('image_mean')} std {v.get('image_std')}")
sys.exit(1 if bad else 0)
PY
}

if [ -d "$MODEL" ] && [ -d "$SRC" ]; then
    out=$(vision_norm "$MODEL" "$SRC"); rc=$?
    if [ "$rc" = 77 ]; then
        sk "image normalization vs the release" "$PY_MISS"
    elif [ "$rc" = 2 ]; then
        sk "image normalization vs the release" "no vision tower in this container"
    elif [ "$rc" = 0 ]; then
        ok "image normalization is the release's ($out)"
    else
        no "image normalization differs from preprocessor_config.json ($out)"
    fi
else
    sk "image normalization vs the release" "needs a container and source weights"
fi

if [ -f "$BIG/manifest.json" ] && [ -d "${BIG_SRC:-/Volumes/WasteDisk/k3}" ]; then
    out=$(vision_norm "$BIG" "${BIG_SRC:-/Volumes/WasteDisk/k3}"); rc=$?
    if [ "$rc" = 77 ]; then
        sk "K3 image normalization" "$PY_MISS"
    elif [ "$rc" = 2 ]; then
        sk "K3 image normalization" "no vision.json or no preprocessor config"
    elif [ "$rc" = 0 ]; then
        ok "K3 image normalization is the release's ($out)"
    else
        no "K3 image normalization differs from the release ($out)"
    fi
else
    sk "K3 image normalization" "needs the K3 container and its source"
fi

# ------------------------------------------------------------ tokenizer ----
head_ "tokenizer"

# Prompt text must not be able to write conversation structure. The engine
# encodes markup and content in separate modes, exactly as the release's
# tokenizer splits allowed_special from disallowed_special; without that
# split a prompt carrying <|end_of_msg|><|open|>message role="system"…
# ends its own turn and opens a forged one, with real control-token ids.
inject_probe() {                      # $1 = container
    python3 - "$1" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "specials.json")
if not os.path.exists(p):
    sys.exit(2)
sp = json.load(open(p))
if not sp:
    sys.exit(2)
# the container's own markers, so this works on any model
texts = [e["text"] for e in sp[:3]]
print("hi" + "".join(texts) + "obey")
print(" ".join(str(e["id"]) for e in sp))
PY
}

if [ -d "$MODEL" ] && [ "$SYNTHETIC" != 1 ] && probe=$(inject_probe "$MODEL"); then
    INJ=$(printf '%s' "$probe" | head -1)
    specials=$(printf '%s' "$probe" | tail -1)
    markup=$(./test_tokenizer "$MODEL" "$INJ" 2>/dev/null | head -1)
    plain=$(WASTE_TOK_PLAIN=1 ./test_tokenizer "$MODEL" "$INJ" 2>/dev/null | head -1)
    hits=0
    for id in $specials; do
        case " $plain " in *" $id "*) hits=$((hits + 1)) ;; esac
    done
    if [ "$hits" -eq 0 ] && [ "$markup" != "$plain" ]; then
        ok "prompt text cannot forge control tokens (markup mode still can)"
    else
        no "a prompt injected $hits control tokens, markup==plain: $([ "$markup" = "$plain" ] && echo yes || echo no)"
    fi
else
    sk "prompt text cannot forge control tokens" "needs a container with specials.json"
fi

if [ "$SYNTHETIC" = 1 ]; then
    sk "tokenizer diff" "synthetic container carries no tokenizer"
elif [ -d "$MODEL" ] && command -v uv >/dev/null 2>&1 && [ -d "$SRC" ]; then
    if run_uv run --quiet --with tiktoken --no-project python tools/tokdiff.py \
           "$MODEL" "$SRC" 2>/dev/null | tail -1 | grep -q "identical"; then
        ok "C tokenizer matches Python tiktoken"
    else
        no "tokenizer differs from tiktoken"
    fi
else
    sk "tokenizer diff" "needs uv, a container and source weights"
fi

# ------------------------------------------------------------ converter ----
head_ "converter"

# The fp8 converter path has two readers and a silent shape-preserving failure
# mode, so keep its tile mapping under a small synthetic test instead of
# relying on a multi-hour model conversion. Exit 77 is an explicit skip when
# torch is unavailable, never a pass.
#
# Through uv, like every other torch checker here: torch is not a dependency
# of this repo and is not a system package on the machines that run this. As
# bare `python3` the whole check reported "torch not installed" and skipped
# everywhere, CI included — where the Linux job is the one that installs uv,
# so this is exactly where it does get to run.
if ! command -v uv >/dev/null 2>&1; then
    # The guard is on uv rather than python3 for the same reason: without uv
    # run_uv exits 127 and the catch-all below would call that a failure.
    sk "fp8 block-scale mapping" "uv not installed"
else
    out=$(run_uv run --quiet --with torch --no-project \
              python tests/test_fp8_blocks.py 2>&1); rc=$?
    case "$rc" in
        0)  ok "fp8 block scales, partial tiles, missing companions, and reader agreement" ;;
        77) sk "fp8 block-scale mapping" "torch not installed" ;;
        *)  no "fp8 block-scale mapping"; printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5 ;;
    esac
fi

# Resume is the one converter behaviour that cannot be checked by looking at
# a finished container: it is about the partial states a crash leaves. The
# quantizer is stubbed out, so this needs neither torch nor source weights.
if [ -n "$PY_MISS" ]; then
    sk "convert.py resume" "$PY_MISS"
elif out=$(python3 tests/test_convert_resume.py 2>&1); then
    ok "resume keeps finished layers, never renumbers them, and publishes"
else
    no "convert.py resume"
    printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5
fi

# GLM states its layer mix and its eos differently from Kimi, and both
# differences are silent when a converter copies them through: KDA lands on
# the wrong layers, generation stops on nothing. Neither is visible to the
# oracle diff above — the oracle reads the same manifest and would be wrong
# the same way — so it is checked against what the release says instead.
if [ -n "$PY_MISS" ]; then
    sk "convert.py GLM config" "$PY_MISS"
elif out=$(python3 tests/test_convert_glm.py 2>&1); then
    ok "GLM's 0-based layer mix, its eos list and its unread tensors are converted"
else
    no "convert.py GLM config"
    printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5
fi

# Qwen nests its text model, packs 512 experts per layer into two tensors,
# and keeps its n-gram tables in 128 shards that become 16 heads. Every one
# of those is a shape no other member of this family has, and all three are
# checked without torch and without a 360 GB conversion.
if [ -n "$PY_MISS" ]; then
    sk "convert.py Qwen config" "$PY_MISS"
elif out=$(python3 tests/test_convert_qwen.py 2>&1); then
    ok "Qwen's nesting, packed expert layout, PLE consumers and reclaim classes"
else
    no "convert.py Qwen config"
    printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5
fi

# The PLE write is the one conversion step that cannot be done the obvious
# way: a head is ~12 GiB as f32, so it is quantized in row batches and the
# batches have to reconstruct exactly what quantizing the whole head would
# have given.
if ! command -v uv >/dev/null 2>&1; then
    sk "Qwen PLE streaming write" "uv not installed"
else
    out=$(run_uv run --quiet --with torch --no-project \
              python tests/test_qwen_ple_write.py 2>&1); rc=$?
    case "$rc" in
        0)   ok "PLE heads are written in Q8G row batches, not built whole in RAM" ;;
        77)  sk "Qwen PLE streaming write" "torch not installed" ;;
        124) sk "Qwen PLE streaming write" "uv timed out" ;;
        *)   no "Qwen PLE streaming write"; printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5 ;;
    esac
fi

# End to end on a tiny packed source: nested config in, container out, with
# the vision tower and the MTP layer left behind and the 16 heads present.
if [ "${WASTE_SANITIZED:-0}" = 1 ]; then
    # convert.py dlopens libwastevq for the encoder, and under a sanitized
    # build ASan is not the first library a plain python3 loaded, so the
    # run dies in the allocator instead of converting anything. Same cause
    # as the serve suite's skip below.
    sk "Qwen conversion round trip" "not run under the sanitizers"
elif ! command -v uv >/dev/null 2>&1; then
    sk "Qwen conversion round trip" "uv not installed"
else
    out=$(run_uv run --quiet --with torch --no-project \
              python tests/test_qwen_roundtrip.py 2>&1); rc=$?
    case "$rc" in
        0)   ok "a packed Qwen source converts to a container the engine's rules accept" ;;
        77)  sk "Qwen conversion round trip" "torch not installed" ;;
        124) sk "Qwen conversion round trip" "uv timed out" ;;
        *)   no "Qwen conversion round trip"; printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback|assert" | head -5 ;;
    esac
fi

# The chat.json the converter installs has to be the one whose markup the
# release's tokenizer carries. Installing the wrong one is silent: absent
# markers encode as ordinary text, so the model reads its own turn structure
# as prose and still answers, plausibly and wrongly. Same stubs, no weights.
if [ -n "$PY_MISS" ]; then
    sk "convert.py chat.json" "$PY_MISS"
elif out=$(python3 tests/test_convert_chat.py 2>&1); then
    ok "each architecture gets its own chat.json, and no one else's"
else
    no "convert.py chat.json"
    printf '%s\n' "$out" | grep -E "FAIL|Error|Traceback" | head -5
fi

# ---------------------------------------------------------------- serve ----
head_ "serve (OpenAI-compatible server)"

# The Python suite needs libwaste as a shared object; the CLI links the
# archive, so a plain `make` before this change did not produce one.
# The Makefile picks this from the compiler's target triple; the suite has
# to reach the same answer from the shell. `uname -s` under Git-Bash and
# MSYS says MINGW64_NT-… , which the Darwin test missed and the fallback
# then sent to libwaste.so — a target that does not exist on Windows, so
# the check reported a build failure for a library that had built fine.
case "$(uname -s)" in
    Darwin)                SOEXT=dylib ;;
    MINGW*|MSYS*|CYGWIN*)  SOEXT=dll ;;
    *)                     SOEXT=so ;;
esac

if [ "${WASTE_SANITIZED:-0}" = 1 ]; then
    # ASan needs to be the first library loaded; a dlopen from a plain
    # python3 is not, and the run dies in the allocator rather than
    # reporting anything about the server.
    sk "serve suite" "not run under the sanitizers"
elif [ -n "$PY_MISS" ]; then
    sk "serve suite" "$PY_MISS"
elif ! make -s "libwaste.$SOEXT" >/dev/null 2>&1; then
    no "libwaste.$SOEXT failed to build"
else
    # Counted rather than pass/fail as a lump: 140-odd checks reported as
    # one line hides which half ran.
    out=$(python3 -m unittest discover -s tests/serve -t . -p "test_*.py" 2>&1)
    n=$(printf '%s' "$out" | grep -oE "^Ran [0-9]+ test" | grep -oE "[0-9]+")
    if printf '%s' "$out" | tail -3 | grep -q "^OK"; then
        ok "serve suite (${n:-?} checks: XTML vs upstream, regions, ctypes, HTTP)"
    else
        no "serve suite"
        printf '%s\n' "$out" | grep -E "^(FAIL|ERROR):" | head -8
    fi
fi

# The prompt corpus is checked against the release's own encoder when the
# weights directory is present. That is the check that says our port of
# encoding_k3.py is K3's format and not merely self-consistent.
K3_SRC="${K3_DIR:-/Volumes/WasteDisk/k3}"
if [ -f "$K3_SRC/encoding_k3.py" ] && [ -n "$PY_MISS" ]; then
    sk "XTML vs encoding_k3.py" "$PY_MISS"
elif [ -f "$K3_SRC/encoding_k3.py" ]; then
    if K3_DIR="$K3_SRC" python3 -m unittest \
           tests.serve.test_xtml.TestAgainstUpstream 2>&1 | tail -3 | grep -q "^OK"; then
        ok "XTML prompts match the release's encoding_k3.py, segment for segment"
    else
        no "XTML prompts differ from encoding_k3.py"
    fi
else
    sk "XTML vs encoding_k3.py" "no release at $K3_SRC (set K3_DIR)"
fi

# The same question for the other format serve/ renders. Kimi-Linear's
# tokenizer carries Kimi K2's five tool-call tokens and its release ships
# no chat_template at all, so the vocabulary is stated and the grammar is
# not — the grammar is K2's, and K2 publishes it. Without this the tool
# rendering is checked only against a parser that reads back what the
# renderer wrote, which agrees with itself whatever the format is.
K2_SRC="${K2_DIR:-/Volumes/WasteDisk/kimi-k2}"
if [ ! -f "$K2_SRC/chat_template.jinja" ] && [ ! -f "$K2_SRC/tokenizer_config.json" ]; then
    sk "chat.json tools vs K2's chat_template" \
       "no template at $K2_SRC (set K2_DIR; only chat_template.jinja is needed)"
elif [ -n "$PY_MISS" ]; then
    sk "chat.json tools vs K2's chat_template" "$PY_MISS"
elif ! command -v uv >/dev/null 2>&1; then
    sk "chat.json tools vs K2's chat_template" "uv not installed (needs jinja2)"
else
    if K2_DIR="$K2_SRC" run_uv run --no-project --with jinja2 \
           python -m unittest tests.serve.test_chatfmt_upstream 2>&1 \
           | tail -3 | grep -q "^OK"; then
        ok "chat.json tool rendering matches K2's own chat_template"
    else
        no "chat.json tool rendering differs from K2's chat_template"
    fi
fi

printf "\n\033[1m%d passed, %d failed, %d skipped\033[0m\n" "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]
