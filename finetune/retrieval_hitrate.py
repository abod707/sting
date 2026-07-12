#!/usr/bin/env python3
"""Measure retrieval hit-rate of the finetuned contrastive head.

For each test-ish query whose gold tool is a termux tool, embed the query and
all 16 termux tool schemas, and check whether every gold tool lands in the
top-k by cosine. Reports hit@3 and hit@6.

Usage: retrieval_hitrate.py <checkpoint.pkl> [n_samples]
"""
import json
import random
import sys

sys.path.insert(0, "/agent/workspace/needle")

import numpy as np
from needle import SimpleAttentionNetwork, load_checkpoint, get_tokenizer
from needle.model.run import encode_for_retrieval

ckpt = sys.argv[1]
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 400

params, config = load_checkpoint(ckpt)
model = SimpleAttentionNetwork(config)
tok = get_tokenizer()

termux_tools = json.load(open("/agent/workspace/ftdata/termux_tools.json"))
tool_names = [t["name"] for t in termux_tools]
tool_texts = [json.dumps(t, separators=(",", ":")) for t in termux_tools]

rows = []
for line in open("/agent/workspace/ftdata/data.jsonl"):
    ex = json.loads(line)
    calls = json.loads(ex["answers"])
    names = [c["name"] for c in calls if isinstance(c, dict)]
    if names and all(n in tool_names for n in names):
        rows.append((ex["query"], set(names)))

rng = random.Random(11)
sample = rng.sample(rows, min(n_samples, len(rows)))

t_emb = encode_for_retrieval(model, params, tok, tool_texts)          # (16, 128)
q_emb = encode_for_retrieval(model, params, tok, [q for q, _ in sample])  # (N, 128)
scores = q_emb @ t_emb.T                                               # (N, 16)

hit3 = hit6 = 0
misses = []
for i, (q, gold) in enumerate(sample):
    order = np.argsort(-scores[i])
    top3 = {tool_names[j] for j in order[:3]}
    top6 = {tool_names[j] for j in order[:6]}
    if gold <= top3:
        hit3 += 1
    if gold <= top6:
        hit6 += 1
    elif len(misses) < 8:
        misses.append((q, sorted(gold), [tool_names[j] for j in order[:6]]))

n = len(sample)
print(f"retrieval over 16 termux tools, {n} queries:")
print(f"  hit@3: {hit3/n:.1%}   hit@6: {hit6/n:.1%}")
if misses:
    print("sample misses (query, gold, top6):")
    for m in misses:
        print("  ", m)
