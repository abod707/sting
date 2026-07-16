# sting 🪡

**Talk to your phone's hardware from Termux — through a 26M-parameter model that runs entirely on-device.**

```
$ sting "turn on the flashlight"
→ termux_torch({"state":"on"})

$ sting "notify me with title 'Build done' and message 'compiled fine'"
→ termux_notification({"title":"Build done","content":"compiled fine"})

$ sting "read the gyroscope, 5 readings"
→ termux_sensor({"sensor":"gyroscope","limit":5})
```

`sting` is a small Rust CLI that wraps [Needle](https://github.com/cactus-compute/needle) —
cactus-compute's 26M-parameter "Simple Attention Network" for single-shot function
calling — finetuned on the Termux:API command set. You type a request in plain
English; the model picks the right `termux-*` command and fills in its arguments;
sting executes it (with your confirmation, or `--yes`).

No cloud, no server, no Python runtime on the phone. One binary + 52MB of weights.

## How it works

```
"vibrate for 2 seconds"
        │
        ▼
┌─────────────────────┐   top-6 by cosine similarity
│ retrieval head      │──────────────────────────────┐
│ (128-d contrastive) │                              │
└─────────────────────┘                              ▼
                                    ┌──────────────────────────────┐
                                    │ Needle 26M encoder-decoder   │
                                    │ (pure attention, no FFN)     │
                                    │ + trie-constrained decoding  │
                                    └──────────────────────────────┘
                                                     │
                                                     ▼
                                [{"name":"termux_vibrate",
                                  "arguments":{"duration_ms":2000}}]
                                                     │
                                                     ▼
                                    std::process::Command
                                    ["termux-vibrate", "-d", "2000"]
```

- **Pure-Rust inference** via [candle](https://github.com/huggingface/candle) — the
  model architecture (encoder-decoder, zero feed-forward layers, GQA + RoPE,
  gated residuals, ZCRMSNorm) is implemented in `src/model.rs`.
- **KV-cached decoding**: self-attention K/V are cached and grown one token per
  step, and cross-attention K/V are computed once from the encoder output (they
  don't change during decoding). Decoding holds a steady ~120-160 tok/s on CPU
  instead of reprocessing the whole prefix every token. See `EVAL.md` for the
  before/after.
- **Pure-Rust SentencePiece BPE tokenizer** (`src/tokenizer.rs`), verified
  token-for-token against the Python reference on 8,253 test strings.
- **Tool retrieval**: the model's contrastive head embeds your query and every
  tool schema into a shared 128-d space; only the top-k (default 6) tools enter
  the prompt. This keeps prefill ~3x faster and matches how the model was
  finetuned. Embeddings are cached beside your tools config.
- **Constrained decoding**: a character-trie over tool names and argument keys
  guarantees the model can only emit tools that exist and keys they actually have.
- **Safety**: commands are built as argv arrays (no shell, no injection), only
  tools with an `exec` entry can run, and each command asks for confirmation
  unless you pass `--yes`. Nothing sensitive (SMS, contacts, call log) ships in
  the default config.

## Install (Termux)

```bash
pkg install rust git binutils
git clone https://github.com/abod707/sting
cd sting
./scripts/termux-install.sh
```

The script builds the release binary (~5-10 min on a phone), installs it to
`$PREFIX/bin`, and sets up `~/.sting` with the model and default tools config.

You also need the **Termux:API app** and package for the commands themselves:

```bash
pkg install termux-api
# + install the Termux:API app from F-Droid (same source as your Termux install)
```

## Usage

```bash
sting "how much battery do I have"          # one-shot
sting --repl                                # interactive
sting --dry-run "set brightness to 200"     # show the command, don't run it
sting --yes "vibrate for 500 ms"            # skip confirmation
sting --raw "what's the weather in Riyadh"  # print the model's JSON only
sting --time "torch on"                     # timing breakdown
sting --top-k 0 "..."                       # disable retrieval (all tools in prompt)
```

## Your own tools

Everything sting knows comes from `tools.json`. Add any CLI as a tool:

```json
{
  "name": "taskforge_add",
  "description": "Add a task to the Taskforge task manager.",
  "parameters": {
    "title": { "type": "string", "description": "Task title.", "required": true }
  },
  "exec": { "cmd": "taskforge", "args": [ { "lit": "add" }, { "arg": "title" } ] }
}
```

MCP-style JSON Schema (`{"type":"object","properties":{...},"required":[...]}`)
is accepted too and converted automatically. Tools without an `exec` entry are
model-only: sting prints the parsed call instead of executing.

The model generalizes to unseen tools reasonably well (it was trained on a mix of
Termux and generic schemas), but for heavy use of custom tools, finetuning on
~120 examples per tool gives the best results — see `finetune/README.md`.

## For AI agents

sting doubles as a deterministic device-control skill for bigger agents — a
local LLM running in Termux, or anything that can shell out. The big model
does the reasoning; sting turns one line of intent into exact, validated
termux-api argv. No flag hallucinations (constrained decoding can't emit a
flag that doesn't exist), no prompt bloat, sub-second on CPU for a typical
query (retrieval mode), fully offline.

`scripts/sting_tool.py` wraps the CLI in a JSON contract:

```bash
python3 scripts/sting_tool.py "turn on the flashlight"           # plan only
# {"ok":true,"calls":[{"name":"termux_torch","arguments":{"state":"on"}}],...}
python3 scripts/sting_tool.py --execute "vibrate for 2 seconds"  # act
python3 scripts/sting_tool.py --list-tools
```

Plan mode never executes anything. An empty `calls` list is meaningful — the
query matched no available tool or was missing required info — so surface it
to the user instead of retrying. Full contract in the script docstring.

## Model

| | |
|---|---|
| Base | [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle) (26M, MIT) |
| Finetune data | 4,810 synthetic examples: 16 Termux:API tools + 14 generic tools, no-tool and multi-call cases |
| Format | f16 safetensors (52MB), upcast to f32 at load — stored in-repo as base64 parts (GitHub API limits); the installer reassembles + sha256-checks it |
| Runtime | candle (CPU), KV-cached decode |

Performance knobs: the Termux installer builds with `RUSTFLAGS="-C
target-cpu=native"` so the tensor kernels use your phone's exact CPU features;
override `RUSTFLAGS` to change it. Prefill's heaviest step (the attention
softmax) is parallelized across cores with rayon, and candle's matmuls
parallelize too — set `RAYON_NUM_THREADS` to cap or raise the thread count.
Prefill benefits most; per-token decode is small and stays effectively
single-threaded.

Eval numbers, the finetuning recipe, and the data generator are in
[`EVAL.md`](EVAL.md) and [`finetune/`](finetune/).

## For Rust learners

This codebase doubles as a worked example for early Rust Book chapters —
ownership moves in the BPE merger, borrows walking a trie, enums for JSON
decoding states. [`LEARNING.md`](LEARNING.md) maps each file to the chapters
it exercises.

## Credits & license

- [cactus-compute/needle](https://github.com/cactus-compute/needle) — the model,
  architecture, and training pipeline (MIT)
- [huggingface/candle](https://github.com/huggingface/candle) — tensor ops (MIT/Apache-2.0)
- sting itself: MIT
