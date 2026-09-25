"""Evaluate any Needle 3 .cact against the Termux ground-truth set.

Generalises eval_cli.py: pass the engine, the weights, the tool schemas and where
to write results, then compare base vs tuned side by side.
"""
import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, "/home/ubuntu/sting3")
from ground_truth import canon, expected  # noqa: E402


def run(engine, model, tools, depth=None, timeout=300):
    base = [engine, "--model", model, "--tools", tools]
    if depth:
        base += ["--depth", str(depth)]
    exp, rows, t0 = expected(), [], time.time()
    for q, want in exp.items():
        p = subprocess.run(base + ["--prompt", q], capture_output=True, text=True, timeout=timeout)
        got = err = None
        try:
            r = json.loads(p.stdout.strip().splitlines()[-1])
            got = r.get("function_calls") or []
            conf = r.get("confidence")
        except Exception as e:
            err, conf = f"{type(e).__name__}: {(p.stderr or p.stdout)[-160:]}", None
        rows.append({"query": q, "expected": want, "got": got, "error": err,
                     "name_ok": got is not None and [c["name"] for c in got] == [c["name"] for c in want],
                     "exact_ok": got is not None and canon(got) == canon(want),
                     "confidence": conf})
    n = len(rows)
    return rows, {
        "model": model, "depth": depth or "full", "n": n,
        "name_acc": round(sum(r["name_ok"] for r in rows) / n, 4),
        "exact": round(sum(r["exact_ok"] for r in rows) / n, 4),
        "parse_failures": sum(1 for r in rows if r["error"]),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/home/ubuntu/.hermes/cache/scratch/n3/needle")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tools", default="/home/ubuntu/sting3/agent_tools.json")
    ap.add_argument("--depth", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, stats = run(args.engine, args.model, args.tools, args.depth)
    stats["label"] = args.label or args.model
    print(json.dumps(stats))
    if args.out:
        json.dump({"stats": stats, "rows": rows}, open(args.out, "w"), indent=1)
        fails = [r for r in rows if not r["exact_ok"]]
        print(f"\n{len(fails)} misses:")
        for r in fails:
            print(f"  {r['query'][:42]!r:44} want {json.dumps(r['expected'])[:58]}")
            print(f"  {'':44} got  {json.dumps(r['got'])[:58]}")


if __name__ == "__main__":
    main()
