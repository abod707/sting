# sting3

Fine-tuning **Needle 3** (Cactus Compute's 8–29 MB on-device automation model) on the
**Termux:API** tool set — the gen-3 successor to [sting](https://github.com/abod707/sting),
which did the same thing for Needle 1.

The point is unchanged: type plain English, get a validated tool call, execute it locally.
No cloud, no API bill, no data leaving the device.

```
sting3 "turn on the flashlight"
→ termux_torch({"state":"on"})

sting3 --dry-run "set brightness to 200"
→ would run: termux-brightness 200
```

## Why this exists

Needle 3 beats the gen-1 model on the same tool set out of the box, but only after
fine-tuning is it usable — same conclusion as gen 1, measured rather than assumed.
Base Needle 3 (20L, unfinetuned) scores **70.0% name accuracy / 55.0% exact match** on
the 40-query Termux ground truth in `ground_truth.py`.

## Layout

| path | what |
|---|---|
| `gen_data.py` | synthetic corpus generator, 4,810 examples over 30 tools (16 Termux + 14 generic distractors) |
| `data/data.jsonl` | the generated corpus |
| `fit_dataset.py` | fits the sequence budget by trimming distractor tools, never by truncating answers |
| `data/data384.jsonl` | corpus that fits seq 384 with zero truncated targets |
| `measure_lens.py` | rendered length distribution (p50/p90/p99/max) |
| `run.sh` | the whole pipeline: `data` → `train` → `build` → `eval` |
| `sting3.py` | the runtime: query → call → argv → execute (C API via `needle`, not `--serve`) |
| `agent_tools.json` | tool schemas in the shape the engine and trainer both use (JSON Schema) |
| `tools.json` | the same tools plus their `exec` specs (argv arrays, no shell) |
| `ground_truth.py` | hand-derived correct answers for the 40-query eval set |
| `eval_model.py` | scores any `.cact` against that set |
| `checkpoints/` | `needle3.safetensors` base (242 MB) + the LoRA adapter |

## Pipeline

```bash
python3 -m venv ~/.venvs/needle && ~/.venvs/needle/bin/pip install "cactus-needle[train]==3.0.1"
bash run.sh data      # generate + fit the corpus
bash run.sh train     # LoRA r16 on the frozen 20L base, 1 epoch
bash run.sh build     # export rungs as .cact
bash run.sh eval      # base vs tuned, same 40 queries
```

## Compute reality (1-OCPU ARM VPS)

Measured, not estimated:

- **~90 s per step** at batch 8, seq 384 → 541 steps ≈ **13 h per epoch**. LoRA trains
  at the full 20 layers regardless of the rung you intend to ship, because
  `needle finetune --layers` is advertised in the help text but not implemented
  ([#139](https://github.com/cactus-compute/needle/issues/139)).
- **~5.6 GB peak RSS** for the trainer (JAX keeps the base in float32). Hermes'
  background-process scope caps memory at ~4 GB, so the run must go in its own
  `systemd-run --user` unit or it gets OOM-killed. `run.sh` does this, with
  `CPUWeight=25` so the rest of the box stays responsive.
- Inference is cheap by comparison: ~750 tok/s decode on the same hardware.

## Things you have to know (measured on this build)

- **Locally built rungs are W4A8, not the 2-bit CQ2 of the shipped model.** The 4-layer
  export was **15.4 MB**, not the 8 MB the release page quotes for a 4L rung — the 2-bit
  post-training is Platform-only. Size claims in Cactus's docs describe *their* quantisation,
  not what `needle build` produces locally.
- **Tuned weights report `confidence: null`.** The confidence head isn't trained by
  `needle finetune` (the CLI says so itself). A tuned model therefore cannot use the
  "act above threshold / escalate below it" contract without additional work.
- **Don't use `--serve`.** In this build it returns the last declared tool for *every*
  input at confidence 1.00, while the same query through the one-shot CLI is correct.
  `sting3.py` drives the C API instead
  ([#136](https://github.com/cactus-compute/needle/issues/136)).
- **`--depth N` is inert** — every ladder depth produces byte-identical output
  ([#145](https://github.com/cactus-compute/needle/issues/145)), so you cannot validate a rung
  by running it at that depth through the CLI.
- **Base-model failure modes worth knowing**, since the tuned model inherits their shape:
  every non-English query collapsed to `termux_clipboard_get` at up to confidence 1.00, and
  numbers got mangled (`brightness 200` → `2`, `"one and a half seconds"` → `150 ms`,
  `"maximum"` → `100` on a 0–15 scale).

## Known upstream issues (found while building this)

| # | what | impact |
|---|---|---|
| [#142](https://github.com/cactus-compute/needle/issues/142) | `pip install cactus-needle` (3.0.5) fetches engine wheels that were never published | fresh install fails on first call; pin `==3.0.1` |
| [#136](https://github.com/cactus-compute/needle/issues/136) | `--serve` returns the last declared tool for every input, at confidence 1.00 | do not deploy the HTTP path; this repo uses the C API |
| [#145](https://github.com/cactus-compute/needle/issues/145) | `--depth N` is inert: every ladder depth gives byte-identical output | a rung can only be validated by building and running *it* |
| [#139](https://github.com/cactus-compute/needle/issues/139) | help advertises `needle finetune --layers`, the parser has no such flag | LoRA always trains at 20 layers; rung choice is `build`-time |

## Safety

`sting3.py` builds commands as **argv arrays, never shell strings**; only tools with an
`exec` spec can run; everything asks for confirmation unless `--yes`; `--dry-run` prints
what would run. SMS, contacts and call logs are deliberately absent from `tools.json`.
