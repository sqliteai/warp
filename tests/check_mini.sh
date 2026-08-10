#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
#
# check_mini.sh — what a merged ("-mini") container costs and what it buys.
#
#   tests/check_mini.sh ORIGINAL MERGED [n_tokens]
#
# Prints one row per container: tok/s, peak RSS, the planned floor, the
# expert bank size and the cache hit rate, then the logit divergence of the
# merged one against the original on the same prompt.
#
# The two halves are deliberately in one script. A merge is only ever a
# trade — it deletes the streaming path, so it is faster and smaller by
# construction, and the only question worth asking is what the model gives
# up for it. Reporting the speedup without the divergence next to it would
# be reporting the half that cannot lose.
#
# Missing prerequisites SKIP loudly rather than passing quietly, the same
# rule the rest of tests/ follows.

set -u
ORIG=${1:-}
MINI=${2:-}
N=${3:-16}
PROMPT="The capital of France is Paris. The capital of Italy is"

if [ -z "$ORIG" ] || [ -z "$MINI" ]; then
    echo "usage: $0 ORIGINAL MERGED [n_tokens]" >&2
    exit 2
fi
for d in "$ORIG" "$MINI"; do
    if [ ! -f "$d/manifest.json" ]; then
        echo "SKIP: $d is not a container"
        exit 77
    fi
done
if [ ! -x ./waste ] || [ ! -x ./test_forward ]; then
    echo "SKIP: build first (make && make test)"
    exit 77
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

row() {                                   # name dir
    local name=$1 dir=$2 bank out rss line
    bank=$(du -ck "$dir"/experts-*.bin 2>/dev/null | tail -1 |
           awk '{printf "%.2f", $1/1048576}')
    out=$TMP/$name.log
    /usr/bin/time -l ./waste run "$dir" --temp 0 -n "$N" "$PROMPT" \
        >"$out" 2>"$TMP/$name.time"
    rss=$(awk '/maximum resident/{printf "%.2f", $1/1073741824}' "$TMP/$name.time")
    # the throughput line goes to stderr, next to time's own output
    line=$(grep -ho '[0-9.]* tok/s' "$out" "$TMP/$name.time" | head -1)
    printf "  %-10s %-12s peak RSS %6s GB   expert bank %8s GB\n" \
        "$name" "${line:-?}" "${rss:-?}" "${bank:-?}"
    grep -ho 'experts [0-9]* hit.*%' "$TMP/$name.time" | head -1 | sed 's/^/    cache: /'
    head -1 "$out" | cut -c1-116 | sed 's/^/    text:  /'
}

echo "== throughput, memory, and what comes out"
row original "$ORIG"
row merged   "$MINI"

# Same token ids through both, then compare the next-token distributions.
IDS=$(./waste tokenize "$ORIG" "$PROMPT" --json 2>/dev/null |
      sed 's/.*"ids":\[//; s/\].*//' | tr -d ' ')
if [ -z "$IDS" ]; then
    echo "SKIP: could not tokenize the prompt (logit divergence not run)"
    exit 77
fi
WASTE_CACHE_MB=${WASTE_CACHE_MB:-4096} ./test_forward "$ORIG" "$IDS" "$TMP/a.bin" 0 >/dev/null 2>&1
WASTE_CACHE_MB=${WASTE_CACHE_MB:-4096} ./test_forward "$MINI" "$IDS" "$TMP/b.bin" 0 >/dev/null 2>&1
if [ ! -s "$TMP/a.bin" ] || [ ! -s "$TMP/b.bin" ]; then
    echo "SKIP: a forward pass produced no logits"
    exit 77
fi

echo "== next-token distribution, merged against original"
uv run --with torch --no-project python - "$TMP/a.bin" "$TMP/b.bin" <<'PY' 2>/dev/null ||
import sys, torch
def load(p):
    return torch.frombuffer(bytearray(open(p, "rb").read()), dtype=torch.float32)
a, b = load(sys.argv[1]), load(sys.argv[2])
lpa = torch.log_softmax(a, 0); pa = lpa.exp()
kl = (pa * (lpa - torch.log_softmax(b, 0))).sum().item()
t10 = len(set(a.topk(10).indices.tolist()) & set(b.topk(10).indices.tolist()))
print(f"  top-1 agrees: {int(a.argmax()) == int(b.argmax())}")
print(f"  top-10 overlap: {t10}/10")
print(f"  KL(original || merged): {kl:.4f} nats")
print(f"  logit rel L2: {((b - a).norm() / a.norm()).item():.4f}")
print(f"  pearson: {torch.corrcoef(torch.stack([a, b]))[0, 1].item():.4f}")
PY
    echo "SKIP: uv/torch unavailable (logit divergence not run)"
