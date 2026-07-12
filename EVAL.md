# Evaluation

Held-out test set: 10 examples per tool (30 tools) + no-tool cases = 300
scoreable examples, split per-tool **before** training (the model never sees
them). Metrics follow needle's own eval: **call_f1** (exact name+arguments
match, set-based), **name_f1** (right tool, any args), **exact_match** (whole
answer identical), **args_acc** (arguments exactly right when the tool name is
right), **parse_rate** (output is valid JSON).

Honest framing: the test split is held out but comes from the same synthetic
generator as training (different phrasings/values, same template families).
Expect somewhat lower accuracy on fully wild phrasing — though the generator
includes politeness noise, casing noise, typos, implicit intent, and
distractor-heavy prompts on purpose.

## Base vs finetuned (greedy + constrained decoding, 300 examples)

| metric | base needle | finetuned | Δ |
|---|---|---|---|
| call_f1 | 75.0% | **99.7%** | +24.7 |
| name_f1 | 94.6% | **100.0%** | +5.4 |
| exact_match | 72.7% | **99.7%** | +27.0 |
| args_acc | 79.2% | **99.7%** | +20.5 |
| parse_rate | 99.3% | **100.0%** | +0.7 |

Training: 2 epochs, batch 8, lr 3e-5 + Muon 0.02, weighted loss on tool-name/
key/value tokens, tool-order shuffling — needle's own finetune pipeline. ~1.5h
on a 2-vCPU / 4GB sandbox (minutes on any GPU).

## Where the base model struggled (per-tool exact-args accuracy)

| tool | base | finetuned |
|---|---|---|
| termux_volume | 2/11 | **11/11** |
| toggle_lights | 2/10 | **10/10** |
| termux_torch | 4/10 | **10/10** |
| create_note | 4/10 | **10/10** |
| send_message | 4/10 | **10/10** |
| create_calendar_event | 4/10 | **10/10** |
| get_directions | 5/10 | **10/10** |
| termux_sensor | 6/10 | **10/10** |
| termux_vibrate | 6/10 | **10/10** |

Base-model failure pattern: picks a *plausible* tool but fumbles argument
values on multi-argument or enum-valued tools (`"state":"flashlight"` instead
of `"on"`), and over-calls (a correct call plus a spurious second one). Every
per-tool bucket is perfect after finetuning.

## Context crowding & retrieval

With all 16 termux tools in the prompt (~850 tokens), the **base** model
collapses entirely ("turn on the flashlight" → `termux_tts_speak("the
flashlight")`). The **finetuned** decoder handles the full 16-tool prompt
correctly, but retrieval still pays: prefill drops ~850 → ~300 tokens.

Two findings worth knowing if you build on needle:

1. **The released checkpoint ships an all-zero contrastive (retrieval) head.**
   Zero weights + ReLU is a gradient fixed point — no amount of joint
   finetuning can revive it. We randomize the head before training.
2. **Joint training barely trains it anyway** (contrastive weight 0.1, 2
   epochs → 62.5% hit@6 over the 16-tool pack). The fix: freeze everything,
   extract mean-pooled encoder features once, and train just the
   512→128→128 head with softmax-over-all-tools (`finetune/train_head.py`,
   seconds of compute). Result, measured over 400 held-out-style queries:

| retrieval over 16 termux tools | hit@3 | hit@6 |
|---|---|---|
| head after joint finetune only | 40.8% | 62.5% |
| head after dedicated training | **99.2%** | **100.0%** |

sting defaults to top-6 retrieval; `--top-k 0` puts every tool in the prompt.

## Runtime cross-check (Rust vs JAX)

The candle/Rust port is verified against the JAX reference:
- tokenizer: **8,253/8,253** strings encode+decode identically
- generation: **50/50** prompts produce byte-identical constrained greedy
  outputs (40 on the base checkpoint, 10 on the shipped finetuned one)

## Timing (sandbox x86-64, 2 vCPU — expect different numbers on your phone)

| stage | all 16 tools | retrieval top-6 |
|---|---|---|
| prefill | ~850 tok / 1.5-2.0s | ~300 tok / 0.24-0.40s |
| decode | ~7 tok/s | ~13-15 tok/s |
| typical query end-to-end | 4-8s | **1.5-2.5s** |

Notes: decode recomputes the full prefix per token (no KV cache yet); typical
answers are 15-30 tokens. Long non-ASCII string values (e.g. Arabic TTS text)
decode via byte-fallback tokens and can take noticeably longer (~120 tokens for
a short sentence). Both are known optimization targets.
