---
license: mit
base_model: Cactus-Compute/needle
library_name: jax
pipeline_tag: text-generation
tags:
- function-calling
- tool-use
- termux
- android
- on-device
- candle
- rust
- needle
language:
- en
- ar
---

# needle-termux-sting 🪡

[Needle](https://huggingface.co/Cactus-Compute/needle) (26M-parameter
"Simple Attention Network" for single-shot function calling, by
cactus-compute) **finetuned on the Termux:API command set** — the model behind
[sting](https://github.com/abod707/sting), a pure-Rust CLI that gives Termux
natural-language device control, fully offline.

```
"vibrate for 2 seconds"
  → [{"name":"termux_vibrate","arguments":{"duration_ms":2000}}]
```

## What's different from base needle

1. **Finetuned decoder** — 4,810 synthetic examples over 16 Termux:API tools +
   14 generic tools (single-call, multi-call, missing-argument, and no-tool
   cases; EN + some Arabic values). Recipe and data generator:
   [sting/finetune](https://github.com/abod707/sting/tree/main/finetune).
2. **Working retrieval head** — the released base checkpoint ships its
   contrastive (retrieval) head as **all zeros**, and zero weights + ReLU is a
   gradient fixed point, so ordinary finetuning can never revive it. This
   checkpoint's head was re-initialized and then trained on frozen encoder
   features (softmax-over-tools). Retrieval over the 16-tool Termux pack:
   **hit@3 = 99.2%, hit@6 = 100%** (400 queries).

## Eval (held-out test set, 300 examples, 30 tools)

| metric | base needle | this model |
|---|---|---|
| call_f1 (name+args exact) | 75.0% | **99.7%** |
| name_f1 | 94.6% | **100.0%** |
| exact_match | 72.7% | **99.7%** |
| args_acc | 79.2% | **99.7%** |
| parse_rate | 99.3% | **100.0%** |

Held-out but same-distribution synthetic data — treat as an upper bound for
wild phrasing. Methodology + per-tool tables:
[sting/EVAL.md](https://github.com/abod707/sting/blob/main/EVAL.md).

## Files

| file | format | use with |
|---|---|---|
| `needle_sting_final.pkl` | needle checkpoint (JAX/Flax, f16) | the official [needle](https://github.com/cactus-compute/needle) pipeline: `needle run --checkpoint needle_sting_final.pkl --query "..." --tools '[...]'` |
| `model.safetensors` + `config.json` + `tokenizer_spec.json` | f16 safetensors + JSON specs | [sting](https://github.com/abod707/sting)'s pure-Rust candle runtime |

## Usage (Python / needle)

```python
from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer

params, config = load_checkpoint("needle_sting_final.pkl")
model = SimpleAttentionNetwork(config)
result = generate(
    model, params, get_tokenizer(),
    query="read the gyroscope, 5 readings",
    tools='[{"name":"termux_sensor","description":"Read values from a hardware sensor on the device.","parameters":{"sensor":{"type":"string","description":"Sensor name: accelerometer, gyroscope, light, proximity, pressure, magnetic_field or gravity.","required":true},"limit":{"type":"integer","description":"Number of readings to take.","required":false}}}]',
    stream=False,
)
# [{"name":"termux_sensor","arguments":{"sensor":"gyroscope","limit":5}}]
```

## Usage (Termux / sting)

```bash
pkg install rust git binutils termux-api
git clone https://github.com/abod707/sting
cd sting && ./scripts/termux-install.sh
sting "turn on the flashlight"
```

## Scope & limitations

Single-shot function calling over a provided toolset. Not conversational, no
multi-step planning; underspecified requests ("set an alarm" with no time)
correctly return `[]`. Custom tools work zero-shot via the generic schemas it
saw in training; for production use of your own tools, finetune with ~120
examples per tool (recipe in the sting repo).

## Credits

Base model, architecture, and training pipeline:
[cactus-compute/needle](https://github.com/cactus-compute/needle) (MIT).
Finetune, retrieval-head fix, and Rust runtime: [abod707](https://github.com/abod707) (MIT).
