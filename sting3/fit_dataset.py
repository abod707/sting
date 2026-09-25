"""Filter/trim the training set to fit a sequence budget without truncating answers.

Truncation (what `--max-len` does when the data is longer) silently drops target
tokens from the loss, which is worse than dropping inputs. So instead of capping the
tokenizer, this shortens the *prompt*: for examples over budget it drops distractor
tool schemas first (never the tools named in the answer), and only drops the example
if it still does not fit.

Lengths come from the same render + tokenizer the trainer uses.
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/ubuntu/.venvs/needle/lib/python3.12/site-packages")
from needle.model.tokenizer import get_tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
TOOLS_START, TOOLS_END = "<|tools_start|>", "<|tools_end|>"


def length(tok, tools, query, answers):
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    answers_json = json.dumps(answers, separators=(",", ":"), ensure_ascii=False)
    prompt = (IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
              + query + IM_END + "\n" + IM_START + "assistant\n")
    target = "<|tool_call_start|>" + answers_json + "<|tool_call_end|>" + IM_END
    return 2 + len(tok.encode(prompt)) + len(tok.encode(target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--budget", type=int, default=384)
    args = ap.parse_args()

    tok = get_tokenizer()
    kept, trimmed, dropped = 0, 0, 0
    with open(args.src) as fin, open(args.dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            tools = ex["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
                ex["tools"] = tools
            answers = ex["answers"]
            if isinstance(answers, str):
                answers = json.loads(answers)
                ex["answers"] = answers

            n = length(tok, tools, ex["query"], answers)
            if n > args.budget:
                # Keep every tool the answer names; drop distractors longest-first.
                # Track removals by index into the ORIGINAL list (rebuilding `tools`
                # while iterating its own indices is how you get an IndexError).
                targets = {c["name"] for c in answers}
                keep = list(range(len(tools)))
                order = sorted(range(len(tools)),
                               key=lambda i: length(tok, [tools[i]], ex["query"], answers),
                               reverse=True)
                for i in order:
                    if n <= args.budget:
                        break
                    if i not in keep or tools[i]["name"] in targets:
                        continue
                    candidate = [tools[j] for j in keep if j != i]
                    new_n = length(tok, candidate, ex["query"], answers)
                    if new_n < n:
                        keep.remove(i)
                        n = new_n
                if n > args.budget:
                    dropped += 1
                    continue
                ex["tools"] = [tools[j] for j in keep]
                trimmed += 1
            ex["tools"] = json.dumps(ex["tools"], separators=(",", ":"), ensure_ascii=False)
            ex["answers"] = json.dumps(ex["answers"], separators=(",", ":"), ensure_ascii=False)
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept {kept}  (trimmed distractors on {trimmed}, dropped {dropped} over budget)")
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
