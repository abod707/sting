"""Convert sting's custom tools.json into the needle3 engine's expected format.

needle3 wants a JSON array of {"name","description","parameters":<JSON Schema>}.
sting used {"name","description","parameters":{k:{"type",...,"required"}},"exec"}.
"""
import json
import sys

TYPES = {"string": "string", "integer": "integer", "number": "number",
         "boolean": "boolean", "array": "array", "object": "object"}


def convert(src, dst):
    raw = json.load(open(src))
    tools = raw["tools"] if isinstance(raw, dict) else raw
    out = []
    for t in tools:
        props, required = {}, []
        for name, p in (t.get("parameters") or {}).items():
            schema = {k: v for k, v in p.items() if k != "required"}
            props[name] = schema or {"type": "string"}
            if p.get("required"):
                required.append(name)
        out.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": {"type": "object", "properties": props, "required": required},
        })
    json.dump(out, open(dst, "w"), indent=2)
    print(f"{len(out)} tools -> {dst}")
    return out


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
