#!/usr/bin/env python3
"""sting3 — talk to your device's tooling through a fine-tuned Needle 3, on-device.

Same contract as sting (the Needle 1 / Rust sibling): you type a request in plain
English, the model picks a tool and fills its arguments, and sting3 executes it —
argv arrays only, no shell, confirmation before anything runs.

Why Python here: Needle 3's shipped HTTP server (--serve) is broken in 3.0.x (it
returns the last declared tool for every input, cactus-compute/needle#136), so this
drives the C API through the `needle` package instead, which is correct.

    sting3 "turn on the flashlight"
    sting3 --dry-run "set brightness to 200"
    sting3 --yes "vibrate for 2 seconds"
    sting3 --raw "notify me with title 'Build done' and message 'compiled fine'"
    sting3 --repl
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOOLS = os.path.join(HERE, "tools.json")
DEFAULT_WEIGHTS = os.path.join(HERE, "engines", "sting3-4L.cact")


def load_tools(path):
    """sting's tools.json -> (schemas for the model, exec specs for the runner)."""
    with open(path) as fh:
        raw = json.load(fh)
    tools = raw["tools"] if isinstance(raw, dict) else raw
    schemas, execs = [], {}
    for t in tools:
        properties, required = {}, []
        for name, spec in (t.get("parameters") or {}).items():
            properties[name] = {k: v for k, v in spec.items() if k != "required"}
            if spec.get("required"):
                required.append(name)
        schemas.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": {"type": "object", "properties": properties, "required": required},
        })
        if t.get("exec"):
            execs[t["name"]] = t["exec"]
    return schemas, execs, tools


def build_argv(spec, arguments):
    """Expand an exec spec into an argv array. Never builds a shell string."""
    argv = [spec["cmd"]]
    for part in spec.get("args") or []:
        if "lit" in part:
            argv.append(str(part["lit"]))
            continue
        key = part.get("arg")
        if key is None:
            continue
        value = arguments.get(key)
        if value is None or value == "":
            if "default" in part:
                value = part["default"]
            elif part.get("optional"):
                continue
            else:
                raise ValueError(f"missing required argument: {key}")
        if part.get("flag"):
            argv.append(str(part["flag"]))
        argv.append(str(value))
    return argv


def execute(name, arguments, spec, assume_yes, dry_run):
    try:
        argv = build_argv(spec, arguments)
    except ValueError as exc:
        return {"error": str(exc)}
    printable = " ".join(shlex.quote(a) for a in argv)
    if dry_run:
        return {"dry_run": printable}
    if not assume_yes:
        try:
            reply = input(f"  run: {printable}\n  execute? [y/N] ").strip().lower()
        except EOFError:
            return {"skipped": "no confirmation available"}
        if reply not in ("y", "yes"):
            return {"skipped": "declined"}
    if shutil.which(argv[0]) is None:
        return {"error": f"{argv[0]} not found on this device"}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return {"error": str(exc)}
    return {"exit": proc.returncode,
            "stdout": proc.stdout.strip()[:400],
            **({"stderr": proc.stderr.strip()[:200]} if proc.stderr.strip() else {})}


def make_agent(weights, tools_path):
    import needle
    schemas, execs, raw_tools = load_tools(tools_path)
    if not os.path.exists(weights):
        sys.exit(f"weights not found: {weights}\n"
                 f"Build them with: needle build <checkpoint> --lora <adapter> --layers 4 --out {weights}")
    agent = needle.Needle(weights=weights, tools=schemas)
    return agent, schemas, execs, raw_tools


def plan(agent, query):
    response = agent.complete(query)
    return response, response.get("function_calls") or []


def handle(agent, query, execs, raw_tools, assume_yes=False, dry_run=False, raw=False):
    response, calls = plan(agent, query)
    if raw:
        print(json.dumps(response, indent=2, ensure_ascii=False))
        return response
    if not calls:
        print("no tool matches that request (the model returns an empty call list, not a guess)")
        return response
    results = []
    for call in calls:
        name, arguments = call.get("name"), call.get("arguments") or {}
        print(f"→ {name}({json.dumps(arguments, ensure_ascii=False)})")
        spec = execs.get(name)
        if spec is None:
            print("   (model-only tool: nothing to execute)")
            results.append({"name": name, "arguments": arguments, "model_only": True})
            continue
        outcome = execute(name, arguments, spec, assume_yes, dry_run)
        if "stdout" in outcome and outcome["stdout"]:
            print(f"   {outcome['stdout']}")
        elif "error" in outcome:
            print(f"   error: {outcome['error']}")
        elif "dry_run" in outcome:
            print(f"   would run: {outcome['dry_run']}")
        elif "skipped" in outcome:
            print(f"   skipped ({outcome['skipped']})")
        results.append({"name": name, "arguments": arguments, **outcome})
    return {"response": response, "results": results}


def main():
    ap = argparse.ArgumentParser(prog="sting3", description=__doc__.splitlines()[0])
    ap.add_argument("query", nargs="*", help="what you want to do, in plain English")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--tools", default=DEFAULT_TOOLS)
    ap.add_argument("--repl", action="store_true", help="interactive loop")
    ap.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--raw", action="store_true", help="print the model's JSON envelope only")
    ap.add_argument("--list-tools", action="store_true")
    args = ap.parse_args()

    if args.list_tools:
        schemas, execs, _ = load_tools(args.tools)
        for s in schemas:
            printable = " (model-only)" if s["name"] not in execs else ""
            print(f"{s['name']:32}{printable}\n    {s['description']}")
        return

    agent, schemas, execs, raw_tools = make_agent(args.weights, args.tools)

    if args.repl or not args.query:
        print(f"sting3 — {len(schemas)} tools, {os.path.basename(args.weights)}. Ctrl-D to exit.")
        while True:
            try:
                query = input("sting3> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            agent.reset()
            handle(agent, query, execs, raw_tools, args.yes, args.dry_run, args.raw)
        return

    handle(agent, " ".join(args.query), execs, raw_tools, args.yes, args.dry_run, args.raw)


if __name__ == "__main__":
    main()
