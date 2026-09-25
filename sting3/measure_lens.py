"""Measure rendered sequence lengths for the training data.

fit_max_len() buckets to 128/256/512... and CAPS at --max-len, truncating anything
longer; a truncated target means the answer tokens are dropped from the loss, so we
want the cap to be >= the true max, or the long examples filtered out deliberately.

Render logic mirrors needle/model/finetune.render_example.
"""
import json
import sys

sys.path.insert(0, "/home/ubuntu/.venvs/needle/lib/python3.12/site-packages")
from needle.model.tokenizer import get_tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
TOOLS_START, TOOLS_END = "<|tools_start|>", "<|tools_end|>"

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/sting3/data/data.jsonl"
tok = get_tokenizer()

lens, tool_counts, over = [], [], {256: 0, 320: 0, 384: 0, 448: 0, 512: 0, 640: 0, 768: 0}
longest = (0, None)
with open(path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        tools = ex.get("tools", [])
        tools_json = tools if isinstance(tools, str) else json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
        answers = ex.get("answers", [])
        answers_json = answers if isinstance(answers, str) else json.dumps(answers, separators=(",", ":"), ensure_ascii=False)
        prompt = (IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
                  + ex["query"] + IM_END + "\n" + IM_START + "assistant\n")
        target = "<|tool_call_start|>" + answers_json + "<|tool_call_end|>" + IM_END
        n = 2 + len(tok.encode(prompt)) + len(tok.encode(target))
        lens.append(n)
        tool_counts.append(len(json.loads(tools_json)) if isinstance(tools_json, str) else len(tools_json))
        if n > longest[0]:
            longest = (n, ex["query"][:60])
        for cap in over:
            if n > cap:
                over[cap] += 1

lens.sort()
n = len(lens)
def pct(p):
    return lens[min(int(n * p / 100), n - 1)]

print(f"examples: {n}")
print(f"min {lens[0]}  p50 {pct(50)}  p75 {pct(75)}  p90 {pct(90)}  p95 {pct(95)}  p99 {pct(99)}  max {lens[-1]}")
print(f"longest example: {longest[1]!r}")
print("\nexamples exceeding each cap (those get their target truncated):")
for cap, c in sorted(over.items()):
    print(f"  > {cap:4}: {c:5}  ({c / n * 100:5.1f}%)")
print(f"\ntools per example: min {min(tool_counts)} max {max(tool_counts)} "
      f"mean {sum(tool_counts) / n:.2f}")
