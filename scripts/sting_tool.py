#!/usr/bin/env python3
"""sting_tool — agent-friendly wrapper around the sting CLI.

For AI agents (local LLMs, assistants, automation) that want reliable
Termux:API access without memorizing flag syntax: describe the intent in
plain English, get structured JSON back.

Two modes:
  plan    (default) run the model only, return the parsed tool calls —
          nothing is executed. Safe to call always.
  execute run the calls through sting's dispatcher (equivalent of --yes).

CLI:
  python3 sting_tool.py "turn on the flashlight"            # plan
  python3 sting_tool.py --execute "vibrate for 2 seconds"   # act
  python3 sting_tool.py --list-tools

Python:
  from sting_tool import run_sting, list_tools
  plan = run_sting("set media volume to 11")           # {"ok":true,"calls":[...]}
  done = run_sting("set media volume to 11", execute=True)

Output contract (JSON on stdout, one object):
  {"ok": bool,            # process-level success
   "calls": [{"name": str, "arguments": {...}}],   # [] = model chose no action
   "executed": bool,
   "output": str,         # command stdout when executed
   "error": str|null}

An empty calls list is MEANINGFUL: the query didn't match any available tool
or was missing required information — ask the user, don't retry blindly.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

CALL_LINE = re.compile(r"^→ ([a-zA-Z0-9_.-]+)\((.*)\)$")


def _sting_bin():
    return os.environ.get("STING_BIN") or shutil.which("sting")


def list_tools(tools_path=None):
    """Return the tool schemas sting knows about (from tools.json)."""
    path = tools_path or os.path.join(
        os.environ.get("STING_HOME", os.path.expanduser("~/.sting")), "tools.json"
    )
    try:
        with open(path, encoding="utf-8") as f:
            root = json.load(f)
        tools = root.get("tools", root) if isinstance(root, dict) else root
        return {"ok": True, "tools": [
            {"name": t.get("name"), "description": t.get("description", ""),
             "executable": "exec" in t} for t in tools
        ]}
    except OSError as e:
        return {"ok": False, "error": f"cannot read tools config: {e}"}


def run_sting(query, execute=False, tools_path=None, top_k=None, timeout=120):
    exe = _sting_bin()
    if not exe:
        return {"ok": False, "calls": [], "executed": False, "output": "",
                "error": "sting binary not found on PATH (run scripts/termux-install.sh)"}

    cmd = [exe]
    if tools_path:
        cmd += ["--tools", tools_path]
    if top_k is not None:
        cmd += ["--top-k", str(top_k)]
    cmd += (["--yes"] if execute else ["--raw"]) + [query]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "calls": [], "executed": False, "output": "",
                "error": f"sting timed out after {timeout}s"}

    out = proc.stdout.strip()
    if proc.returncode != 0:
        return {"ok": False, "calls": [], "executed": False, "output": out,
                "error": (proc.stderr.strip() or f"sting exited {proc.returncode}")}

    if not execute:
        # --raw prints the model's JSON (or [] / free text on no-call)
        try:
            parsed = json.loads(out) if out else []
        except json.JSONDecodeError:
            parsed = []
        calls = parsed if isinstance(parsed, list) else [parsed]
        calls = [c for c in calls if isinstance(c, dict) and c.get("name")]
        return {"ok": True, "calls": calls, "executed": False, "output": "", "error": None}

    # execute mode: parse the "→ tool({...})" lines sting prints per call
    calls, output_lines = [], []
    for line in out.splitlines():
        m = CALL_LINE.match(line.strip())
        if m:
            try:
                args = json.loads(m.group(2)) if m.group(2).strip() else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": m.group(1), "arguments": args})
        else:
            output_lines.append(line)
    return {"ok": True, "calls": calls, "executed": bool(calls),
            "output": "\n".join(output_lines).strip(), "error": None}


def main():
    ap = argparse.ArgumentParser(description="agent-friendly wrapper for sting")
    ap.add_argument("query", nargs="?", help="natural-language device request")
    ap.add_argument("--execute", action="store_true", help="run the command(s), not just plan")
    ap.add_argument("--tools", default=None, help="custom tools.json path")
    ap.add_argument("--top-k", type=int, default=None, help="retrieval shortlist size")
    ap.add_argument("--list-tools", action="store_true", help="print available tools")
    args = ap.parse_args()

    if args.list_tools:
        print(json.dumps(list_tools(args.tools), ensure_ascii=False, indent=2))
        return
    if not args.query:
        ap.error("query required (or --list-tools)")
    result = run_sting(args.query, execute=args.execute,
                       tools_path=args.tools, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
