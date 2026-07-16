#!/usr/bin/env bash
# sting performance benchmark.
#
# Runs the query set (scripts/golden/queries.txt) through the release binary in
# both retrieval (top-6) and all-tools (--top-k 0) modes and reports prefill /
# decode timing. Pure measurement — no pass/fail. For the correctness gate, see
# scripts/regression.sh.
#
#   bash scripts/bench.sh              # default query set, both modes
#   bash scripts/bench.sh -n 8         # only the first 8 queries (quicker)
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="target/release/sting"
QUERIES="scripts/golden/queries.txt"
LIMIT=0
[ "${1:-}" = "-n" ] && LIMIT="${2:-0}"

say() { printf '\033[1;36m[bench]\033[0m %s\n' "$*"; }

# --- ensure the binary and model weights are present -------------------------
if [ ! -x "$BIN" ]; then
  say "building release binary"
  cargo build --release
fi
if [ ! -f model/model.safetensors ] && ls model/model.safetensors.b64.part* >/dev/null 2>&1; then
  say "reassembling model weights from base64 parts"
  cat model/model.safetensors.b64.part* | base64 -d > model/model.safetensors
fi

# warm the retrieval embedding cache so timings are steady-state
"$BIN" --raw "warm up the cache" >/dev/null 2>&1 || true

run_mode() {
  local label="$1"; shift
  local flags=("$@")
  printf '\n=== %s ===\n' "$label"
  printf '%-46s %12s %14s\n' "query" "prefill(ms)" "decode(tok/s)"
  local n=0
  # timing is printed to stderr by --time; capture stderr, discard stdout
  while IFS= read -r q; do
    [ -z "$q" ] && continue
    n=$((n + 1))
    if [ "$LIMIT" -gt 0 ] && [ "$n" -gt "$LIMIT" ]; then break; fi
    local se pf dc
    se="$("$BIN" --raw --time "${flags[@]}" "$q" 2>&1 1>/dev/null)"
    pf="$(printf '%s\n' "$se" | sed -n 's/.*prefill [0-9]* tok \/ \([0-9]*\) ms.*/\1/p')"
    dc="$(printf '%s\n' "$se" | sed -n 's/.*(\([0-9.]*\) tok\/s).*/\1/p')"
    printf '%-46.46s %12s %14s\n' "$q" "${pf:-?}" "${dc:-?}"
    echo "${pf:-0} ${dc:-0}" >> /tmp/.bench_acc
  done < "$QUERIES"
}

summarize() {
  awk '{p+=$1; d+=$2; n++} END {if(n>0) printf "  avg prefill %.0f ms | avg decode %.1f tok/s over %d queries\n", p/n, d/n, n}' /tmp/.bench_acc
  rm -f /tmp/.bench_acc
}

: > /tmp/.bench_acc; run_mode "retrieval top-6" ; summarize
: > /tmp/.bench_acc; run_mode "all tools (--top-k 0)" --top-k 0 ; summarize
