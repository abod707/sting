# scripts

Dev and CI helpers. Run them from the repo root; each will build the release
binary and reassemble the model weights (from the base64 parts) if needed.

## `regression.sh` — correctness gate

Asserts that model output **and** tokenization are byte-for-byte unchanged
against committed goldens. Run it after any change that shouldn't alter results
(performance work, refactors); a failure means the change moved an output.

```bash
bash scripts/regression.sh            # check against goldens; exits non-zero on drift
bash scripts/regression.sh --update   # regenerate goldens (after an INTENDED change)
```

Coverage:

- **Model output** — every query in `golden/queries.txt`, in three modes
  (retrieval top-6, all-tools `--top-k 0`, and unconstrained `--no-constrain`) →
  `golden/outputs.golden`. Running all three modes catches regressions in
  retrieval, the decoder, and the constrained-decoding grammar independently.
- **Tokenizer** — every line in `golden/corpus.txt` (ASCII, code/JSON, Arabic,
  CJK, emoji, whitespace and byte-fallback edge cases) encoded to token ids →
  `golden/tokens.golden`.

The goldens are produced by a build verified byte-identical to the original
pre-optimization port, so "matches golden" means "matches the reference
implementation." When you intentionally change outputs, run `--update`, eyeball
the diff, and commit the regenerated goldens alongside the code.

> For token-for-token parity against the **Python SentencePiece** reference (the
> upstream ground truth, not just "unchanged since last commit"), use
> `sting verify-tokenizer <parity.jsonl>` — see `EVAL.md`.

## `bench.sh` — performance benchmark

Runs the query set in both modes with `--time` and reports prefill / decode
timing plus per-mode averages. Pure measurement, no pass/fail.

```bash
bash scripts/bench.sh          # full query set, both modes
bash scripts/bench.sh -n 8     # first 8 queries only (quick)
```

## `golden/`

| file | what it is |
|---|---|
| `queries.txt` | representative requests (all 16 tools, multi-arg, multilingual, no-tool) |
| `corpus.txt` | tokenizer stress lines (stable; not derived from repo files) |
| `outputs.golden` | expected model output, 3 modes/query — regenerate with `regression.sh --update` |
| `tokens.golden` | expected token ids per corpus line — regenerate with `regression.sh --update` |

## `tokenize` subcommand

`regression.sh` leans on a small dev subcommand that prints token ids per input
line (deterministic, so its dump doubles as the tokenizer fixture):

```bash
sting tokenize golden/corpus.txt      # one line of space-joined ids per input line
```
