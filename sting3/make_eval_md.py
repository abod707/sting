#!/usr/bin/env python3
"""Render sting3's EVAL.md from the eval_model.py result files.

Reads results_*.json in the project dir and writes a human-readable report: the
base-vs-rung comparison, rung sizes, and the remaining misses for the smallest rung.
Numbers come from the files, so the report cannot drift from the measurements.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["base", "sting3-4L", "sting3-8L", "sting3-20L"]


def load():
    runs = {}
    for path in glob.glob(os.path.join(HERE, "results_*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        label = data["stats"].get("label") or os.path.basename(path)
        runs[label] = data
    return runs


def size_of(label):
    """Rung file size in MB, when the rung maps to a file we built."""
    match = {"sting3-4L": "sting3-4L.cact", "sting3-8L": "sting3-8L.cact",
             "sting3-20L": "sting3-20L.cact"}.get(label)
    if not match:
        return "-"
    path = os.path.join(HERE, "engines", match)
    if not os.path.exists(path):
        return "-"
    return f"{os.path.getsize(path) / 1e6:.1f} MB"


def main():
    runs = load()
    if not runs:
        sys.exit("no results_*.json found — run the evals first")

    labels = [l for l in ORDER if l in runs] + [l for l in runs if l not in ORDER]
    out = ["# Evaluation — sting3 (Needle 3, Termux:API tool set)", "",
           "Same 40-query held-out set as `ground_truth.py`, run through the shipped engine",
           "binary (one process per query; the `--serve` path is broken, see Known issues).",
           "Tool calling is exact match on names, order **and** every argument value.", "",
           "| model | rung size | name accuracy | exact match | parse failures | queries |",
           "|---|---|---|---|---|---|"]
    for label in labels:
        s = runs[label]["stats"]
        out.append(f"| {label} | {size_of(label)} | {s['name_acc'] * 100:.1f}% | "
                   f"{s['exact'] * 100:.1f}% | {s['parse_failures']} | {s['n']} |")

    # The smallest tuned rung is the one that ships on a phone, so its misses matter most.
    focus = next((l for l in labels if l.endswith("4L")), None)
    if focus:
        misses = [r for r in runs[focus]["rows"] if not r["exact_ok"]]
        out += ["", f"## Remaining misses — {focus} ({len(misses)}/{runs[focus]['stats']['n']})", ""]
        if not misses:
            out.append("None: every query in the set is exact.")
        for r in misses:
            out.append(f"- `{r['query']}`")
            out.append(f"  - expected `{json.dumps(r['expected'], ensure_ascii=False)}`")
            out.append(f"  - got `{json.dumps(r['got'], ensure_ascii=False)}`")
    out += ["", "## Notes", "",
            "- Base Needle 3 is not usable as shipped on this tool set; every documented",
            "  parity claim for the ladder assumes fine-tuning, and this run agrees.",
            "- Locally built rungs are W4A8, not the 2-bit CQ2 of the distributed model,",
            "  so they are larger than the sizes quoted on the release page.",
            "- Tuned weights report `confidence: null`: `needle finetune` does not train the",
            "  confidence head, so it cannot gate execution without separate calibration.",
            ""]
    path = os.path.join(HERE, "EVAL.md")
    open(path, "w").write("\n".join(out))
    print("wrote", path)
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
