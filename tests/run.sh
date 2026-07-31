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

head_ "build"
if make -s test >/dev/null 2>&1 && make -s >/dev/null 2>&1; then
    ok "make && make test"
else
    no "build failed"
    make test 2>&1 | grep -E "error" | head -5
    exit 1
fi

# ---------------------------------------------------------------- unit ----
head_ "kernels vs the reference implementations"

if command -v uv >/dev/null 2>&1; then
    ./test_k3parts "$TMP/k3parts.bin" >/dev/null 2>&1
    if uv run --quiet --with torch --no-project python tools/k3parts_ref.py \
           "$TMP/k3parts.bin" 2>/dev/null | grep -q "^PASS"; then
        ok "K3 components (SiTU, both decay gates, AttnRes)"
    else
        no "K3 components"
    fi

    if KDA_T=24 KDA_H=4 KDA_K=32 KDA_V=32 uv run --quiet --with torch \
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
# The path goes in argv, not into the source: MSYS2 rewrites POSIX paths in
# a native program's arguments and cannot rewrite one quoted inside -c, so
# Windows Python was handed /tmp/... verbatim and could not find it.
python3 -c "import sys; open(sys.argv[1],'wb').write(bytes(range(256))*4096)" "$FT/srv/s.bin"

# The worker the downloader generates is curl, so without curl these three
# report the downloader broken when what is missing is a tool — which is the
# one thing this suite says it must never do. Debian slim and a bare MinGW
# both lack it.
if ! command -v curl >/dev/null 2>&1; then
    sk "a truncated shard resumes and verifies against Content-Length" "curl not installed"
    sk "a completed shard is skipped without a request" "curl not installed"
    NO_CURL=1
elif ! start_server; then
    no "range server did not start (resume, state-file skip not run)"
else
    head -c 400000 "$FT/srv/s.bin" > "$FT/dst/s.bin"
    fetch_worker $PORT $SM
    : > "$FT/dst/.st"
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    if cmp -s "$FT/srv/s.bin" "$FT/dst/s.bin" && grep -q "^s.bin$" "$FT/dst/.st"; then
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

if [ -n "${NO_CURL:-}" ]; then
    sk "a server without Range support restarts the shard instead of giving up" "curl not installed"
elif ! start_server --no-range; then
    no "range server did not start (no-range fallback not run)"
else
    rm -f "$FT/dst/s.bin" "$FT/dst/.st" "$FT/dst/log"
    head -c 400000 "$FT/srv/s.bin" > "$FT/dst/s.bin"
    fetch_worker $PORT $SM
    : > "$FT/dst/.st"
    bash "$FT/dst/.worker.sh" s.bin >/dev/null 2>&1
    if cmp -s "$FT/srv/s.bin" "$FT/dst/s.bin"; then
        ok "a server without Range support restarts the shard instead of giving up"
    else
        no "no-range fallback"
    fi
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

    if [ "$SYNTHETIC" = 1 ]; then
        sk "container round-trip" "synthetic container has no source weights"
    elif [ -d "$SRC" ] && command -v uv >/dev/null 2>&1; then
        if uv run --quiet --with torch --no-project python tools/verify_container.py \
               --container "$MODEL" --src "$SRC" --experts 1 2>/dev/null \
               | grep -q "^PASS"; then
            ok "dequantized weights match the source"
        else
            no "container round-trip"
        fi
    else
        sk "container round-trip" "source weights not at $SRC"
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

    ./test_forward "$MODEL" "$IDS" "$TMP/seq.bin" 0 >/dev/null 2>&1
    WASTE_CHUNK=1 ./test_forward "$MODEL" "$IDS" "$TMP/chunk.bin" 0 >/dev/null 2>&1
    if python3 - "$TMP/seq.bin" "$TMP/chunk.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
d = max(abs(x - y) for x, y in zip(a, b))
sys.exit(0 if d < 1e-3 and a.index(max(a)) == b.index(max(b)) else 1)
PY
    then ok "chunked prefill == token-at-a-time"
    else no "chunked prefill diverges"
    fi

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
    # the tower is never made resident by the text path, so it is not part
    # of what this would allocate — src/model.c skips it at load
    if t.get("name", "").startswith(("vision_tower.", "mm_projector.")):
        continue
    n = 1
    for d in t.get("shape", []):
        n *= d
    need += n * 4
r = subprocess.run([WASTE, "plan", sys.argv[1], "--json"],
                   capture_output=True, text=True)
try:
    phys = json.loads(r.stdout)["physical_ram_bytes"]
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

    WASTE_BACKEND=cpu ./test_forward "$MODEL" "$IDS" "$TMP/cpu.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cpu.bin"; then
        ok "SIMD backend bit-identical to the CPU baseline"
    else
        # a difference here is allowed to be tiny, but must be tiny
        if python3 - "$TMP/seq.bin" "$TMP/cpu.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "SIMD backend matches the CPU baseline (within fp noise)"
        else no "SIMD backend diverges from the CPU baseline"
        fi
    fi

    WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" "$TMP/cache.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cache.bin"; then
        ok "expert cache is bit-identical to no cache"
    else
        no "expert cache changes results"
    fi

    # Read-ahead is on by default, so the synchronous path — the fallback,
    # and the thing every earlier measurement was made on — is the one no
    # check would otherwise run. It shipped broken for exactly one build:
    # a synchronous claim took a pin that never expired, the victim sampler
    # ran out of slots, and the forward pass answered with the experts it
    # had instead of failing.
    WASTE_IO_THREADS=0 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/sync.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/cache.bin" "$TMP/sync.bin"; then
        ok "read-ahead is bit-identical to synchronous reads"
    else
        no "read-ahead changes results"
    fi

    # A purged slot reads back as zeros, so the whole prototype rests on the
    # engine noticing before it multiplies one. This does not create memory
    # pressure — it checks that the volatile/nonvolatile traffic itself does
    # not disturb a record. Vacuously true off macOS, where the flag is a
    # no-op and says so.
    WASTE_PURGEABLE=1 WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" \
        "$TMP/purge.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/cache.bin" "$TMP/purge.bin"; then
        ok "purgeable slots are bit-identical to ordinary ones"
    else
        no "purgeable slots change results"
    fi

    ORACLE="${WASTE_ORACLE:-tests/fixtures/oracle_kimilinear_16tok.bin}"
    if [ "$SYNTHETIC" = 1 ]; then
        sk "engine vs the PyTorch oracle" "synthetic container has no reference logits"
    elif [ -f "$ORACLE" ]; then
        if python3 - "$TMP/seq.bin" "$ORACLE" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "engine matches the PyTorch oracle"
        else no "engine diverges from the oracle"
        fi
    else
        sk "oracle diff" "no fixture; regenerate with tools/kimi_ref.py --dump"
    fi

    if ./test_state "$MODEL" 2>/dev/null | grep -q "^STATE OK"; then
        ok "saved session resumes identically"
    else
        no "session state does not round-trip"
    fi

    # learned hotlist: a second run should start warmer than the first
    if [ "$SYNTHETIC" = 1 ]; then
        sk "learned hotlist" "synthetic container carries no tokenizer"
    else
    # Empty, not 0, when the count cannot be read: a run that printed no
    # stat line and a run that missed nothing are different outcomes, and
    # `|| echo 0` used to collapse them. On a machine whose budget holds the
    # whole working set the warm run reaches 0 misses — the best result this
    # check can produce — and the old `warm -gt 0` guard reported it as
    # "hotlist did not reduce misses (584 -> 0)".
    misses() { grep -oE "[0-9]+ miss" | grep -oE "[0-9]+"; }
    rm -f "$MODEL/usage.waste"
    cold=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G --learn 2>&1 \
           | misses)
    warm=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G 2>&1 \
           | misses)
    rm -f "$MODEL/usage.waste"
    if [ -z "$cold" ] || [ -z "$warm" ]; then
        no "hotlist: no miss count in the output of waste run"
    elif [ "$cold" -eq 0 ]; then
        # Nothing to warm, so nothing is demonstrated either way. Saying
        # PASS here would be passing by luck.
        sk "learned hotlist" "the cold run already missed nothing"
    elif [ "$warm" -lt "$cold" ]; then
        ok "learned hotlist warms the cache ($cold -> $warm misses)"
    else
        no "hotlist did not reduce misses ($cold -> $warm)"
    fi
    fi
else
    sk "engine checks" "no container at $MODEL"
fi

# --------------------------------------------------------------- budget ----
head_ "RAM budget"

# The default budget is the one path check_budget.sh cannot reach, because
# it always passes --budget. With no flag the engine chooses, and that
# choice is all that stands between a model whose recommendation exceeds
# the machine — K3 asks for 80.63 GB — and a swap storm. So assert the
# rule, not a number, and it holds on any host: the engine steps down a
# whole token working set at a time and takes the largest of
# floor + 3x, 2x, 1x that fits under 7/8 of physical RAM, or the floor
# when not even one multiple does, less at most one expert record of slot
# rounding. Filling the cap instead is what put a 27 GB cache on a 64 GB
# machine and cost 8x throughput — docs/LEARNED.md §16.
default_budget() {
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
# CPython at all. This is the same number the engine sized itself against,
# so what the check still tests is the rule — floor + the largest whole
# working set under 7/8 of RAM — and not the RAM reading, which has its
# own platform code and no business being written twice.
phys = plan["physical_ram_bytes"]
if not phys:
    print("physical RAM unknown on this host")
    sys.exit(0)
cap = phys - phys // 8
# what the engine actually holds: the plan's mandatory parts plus the
# cache it really allocated, which is what `info` reports
held = plan["floor_bytes"] - plan["min_expert_cache"] + info["expert_cache_bytes"]
# recommended_bytes is floor + 3 * one token's working set, by construction
ws = (plan["recommended_bytes"] - plan["floor_bytes"]) // 3
want = plan["floor_bytes"]
for k in (3, 2, 1):
    if plan["floor_bytes"] + ws * k <= cap:
        want = plan["floor_bytes"] + ws * k
        break
rec = plan["min_expert_cache"] // (2 * info["top_k"]) if info["top_k"] else 0
G = 1 << 30
print(f"{held/G:.2f} GB held, ceiling {want/G:.2f} GB, machine {phys/G:.2f} GB")
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

    if out=$(default_budget "$MODEL" 2>/dev/null); then
        ok "no --budget picks a ceiling the machine can hold ($out)"
    else
        no "default budget off the rule (${out:-no output})"
    fi
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
    elif tests/check_budget.sh "$MODEL" 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the configured budget"
    else
        no "peak RSS exceeded the budget"
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
    if tests/check_budget.sh "$BIG" 32 long 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the budget on K3 too"
    else
        no "peak RSS exceeded the budget on K3"
    fi

    # K3 is the only model here whose recommendation can exceed the
    # machine, so it is the one that makes the step-down bite at all: on a
    # 64 GB host the default lands on floor + 1x working set = 46.24 GB,
    # rather than the floor + 3x = 80.63 GB asked for. On a host large
    # enough to hold the recommendation this still passes — it checks the
    # rule, not the clamp.
    if out=$(default_budget "$BIG" 2>/dev/null); then
        ok "no --budget is capped to the machine on K3 ($out)"
    else
        no "default budget not capped on K3 (${out:-no output})"
    fi
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
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json"))
r = subprocess.run([os.path.join(os.curdir, "waste" + (".exe" if os.name == "nt" else "")), "info", d, "--json"], capture_output=True, text=True)
info = json.loads(r.stdout)
c = man["config"]

hf = ((c.get("_outer", {}).get("architectures") or c.get("architectures")
       or [""]))[0]
arch = ("kimi-k3" if "KimiK3" in hf else "kimi-linear" if "KimiLinear" in hf
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
    if out=$(info_rule "$MODEL" 2>/dev/null); then
        ok "info describes the container it opened ($out)"
    else
        no "info describes something else (${out:-no output})"
    fi
    if out=$(params_rule "$MODEL" 2>/dev/null); then
        ok "params_total is what the container holds ($out)"
    else
        no "params_total off the rule (${out:-no output})"
    fi
else
    sk "parameter counts" "no container at $MODEL"
fi

# K3 is the model both rules were wrong about, and the only latent MoE
# here: without it the expert width and the hidden never differ, and the
# two descriptive fields pass on the constants they used to be.
if [ -f "$BIG/manifest.json" ]; then
    if out=$(info_rule "$BIG" 2>/dev/null); then
        ok "info names K3 and its trunk, not the model before it ($out)"
    else
        no "info describes something else on K3 (${out:-no output})"
    fi
    if out=$(params_rule "$BIG" 2>/dev/null); then
        ok "params_total counts K3's experts at the latent ($out)"
    else
        no "params_total off the rule on K3 (${out:-no output})"
    fi
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
    if [ "$rc" = 2 ]; then
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
    if [ "$rc" = 2 ]; then
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
    if uv run --quiet --with tiktoken --no-project python tools/tokdiff.py \
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

# Resume is the one converter behaviour that cannot be checked by looking at
# a finished container: it is about the partial states a crash leaves. The
# quantizer is stubbed out, so this needs neither torch nor source weights.
if ! command -v python3 >/dev/null 2>&1; then
    sk "convert.py resume" "python3 not installed"
elif out=$(python3 tests/test_convert_resume.py 2>&1); then
    ok "resume keeps finished layers, never renumbers them, and publishes"
else
    no "convert.py resume"
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
elif ! command -v python3 >/dev/null 2>&1; then
    sk "serve suite" "python3 not installed"
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
if [ -f "$K3_SRC/encoding_k3.py" ]; then
    if K3_DIR="$K3_SRC" python3 -m unittest \
           tests.serve.test_xtml.TestAgainstUpstream 2>&1 | tail -3 | grep -q "^OK"; then
        ok "XTML prompts match the release's encoding_k3.py, segment for segment"
    else
        no "XTML prompts differ from encoding_k3.py"
    fi
else
    sk "XTML vs encoding_k3.py" "no release at $K3_SRC (set K3_DIR)"
fi

printf "\n\033[1m%d passed, %d failed, %d skipped\033[0m\n" "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]
