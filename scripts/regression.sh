#!/usr/bin/env bash
# sting correctness regression gate.
#
# Asserts that model output and tokenization are byte-for-byte unchanged against
# committed goldens. Run it after any change that shouldn't alter results (perf
# work, refactors); if it fails, the change moved an output.
#
#   bash scripts/regression.sh            # check against goldens; exit 1 on drift
#   bash scripts/regression.sh --update   # regenerate goldens (after an INTENDED change)
#
# What it covers:
#   * model output  — every query in scripts/golden/queries.txt, in 3 modes
#       (retrieval top-6, all-tools, unconstrained) → scripts/golden/outputs.golden
#   * tokenizer     — every line in scripts/golden/corpus.txt encoded to ids
#       → scripts/golden/tokens.golden
#
# For token-for-token parity against the Python SentencePiece reference (the
# ground truth, not just "unchanged"), use:  sting verify-tokenizer <parity.jsonl>
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="target/release/sting"
G="scripts/golden"
QUERIES="$G/queries.txt"
CORPUS="$G/corpus.txt"
OUT_GOLDEN="$G/outputs.golden"
TOK_GOLDEN="$G/tokens.golden"
UPDATE=0
[ "${1:-}" = "--update" ] && UPDATE=1

say() { printf '\033[1;36m[regression]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[regression] FAIL:\033[0m %s\n' "$*"; }
pass() { printf '\033[1;32m[regression] PASS:\033[0m %s\n' "$*"; }

# --- ensure the binary and model weights are present -------------------------
if [ ! -x "$BIN" ]; then
  say "building release binary"
  cargo build --release
fi
if [ ! -f model/model.safetensors ] && ls model/model.safetensors.b64.part* >/dev/null 2>&1; then
  say "reassembling model weights from base64 parts"
  cat model/model.safetensors.b64.part* | base64 -d > model/model.safetensors
fi

# --- portable "are these two files identical?" -------------------------------
files_equal() { # 0 = identical
  if command -v cmp >/dev/null 2>&1; then cmp -s "$1" "$2"; return $?; fi
  local a b
  a="$(hash_of "$1")"; b="$(hash_of "$2")"; [ "$a" = "$b" ]
}
hash_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else wc -c <"$1"; fi
}
show_diff() {
  if command -v diff >/dev/null 2>&1; then diff -u "$1" "$2" | head -40
  elif command -v git >/dev/null 2>&1; then git --no-pager diff --no-index -- "$1" "$2" | head -40
  else echo "  (install diffutils to see line-level differences)"; fi
}

# --- generate current outputs ------------------------------------------------
gen_outputs() { # $1 = destination file
  local dst="$1"; : > "$dst"
  "$BIN" --raw "warm up the cache" >/dev/null 2>&1 || true
  while IFS= read -r q; do
    [ -z "$q" ] && continue
    printf 'top6\t%s\t%s\n'  "$q" "$("$BIN" --raw "$q" 2>/dev/null)"            >> "$dst"
    printf 'all\t%s\t%s\n'   "$q" "$("$BIN" --raw --top-k 0 "$q" 2>/dev/null)"  >> "$dst"
    printf 'nocon\t%s\t%s\n' "$q" "$("$BIN" --raw --no-constrain "$q" 2>/dev/null)" >> "$dst"
  done < "$QUERIES"
}
gen_tokens() { # $1 = destination file
  "$BIN" tokenize "$CORPUS" > "$1"
}

if [ "$UPDATE" -eq 1 ]; then
  say "regenerating goldens from the current binary"
  gen_outputs "$OUT_GOLDEN"
  gen_tokens  "$TOK_GOLDEN"
  say "wrote $OUT_GOLDEN ($(wc -l <"$OUT_GOLDEN") rows) and $TOK_GOLDEN ($(wc -l <"$TOK_GOLDEN") rows)"
  say "review the diff, then commit the updated goldens."
  exit 0
fi

rc=0
tmp_out="$(mktemp)"; tmp_tok="$(mktemp)"
trap 'rm -f "$tmp_out" "$tmp_tok"' EXIT

say "checking model output over $(grep -c . "$QUERIES") queries x 3 modes"
gen_outputs "$tmp_out"
if [ ! -f "$OUT_GOLDEN" ]; then
  fail "no golden yet — run: bash scripts/regression.sh --update"; rc=1
elif files_equal "$tmp_out" "$OUT_GOLDEN"; then
  pass "model output byte-identical to golden"
else
  fail "model output drifted from golden:"; show_diff "$OUT_GOLDEN" "$tmp_out"; rc=1
fi

say "checking tokenizer over $(grep -c . "$CORPUS") corpus lines"
gen_tokens "$tmp_tok"
if [ ! -f "$TOK_GOLDEN" ]; then
  fail "no golden yet — run: bash scripts/regression.sh --update"; rc=1
elif files_equal "$tmp_tok" "$TOK_GOLDEN"; then
  pass "tokenizer byte-identical to golden"
else
  fail "tokenizer drifted from golden:"; show_diff "$TOK_GOLDEN" "$tmp_tok"; rc=1
fi

[ "$rc" -eq 0 ] && say "all checks passed" || fail "regressions detected (see above)"
exit "$rc"
