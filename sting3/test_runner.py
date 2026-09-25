"""Verify sting3's argv builder: flags, defaults, optional args, and injection safety.

This is the part that actually touches the device, so it gets tested directly rather
than through the model.
"""
import json
import sys

sys.path.insert(0, "/home/ubuntu/sting3")
from sting3 import build_argv, execute  # noqa: E402

with open("/home/ubuntu/sting3/tools.json") as fh:
    tools = {t["name"]: t for t in json.load(fh)["tools"]}

cases = [
    ("termux_torch", {"state": "on"}, ["termux-torch", "on"]),
    ("termux_vibrate", {"duration_ms": 2000}, ["termux-vibrate", "-d", "2000"]),
    ("termux_vibrate", {}, ["termux-vibrate", "-d", "1000"]),               # default
    ("termux_brightness", {"brightness": 200}, ["termux-brightness", "200"]),
    ("termux_sensor", {"sensor": "gyroscope", "limit": 5},
     ["termux-sensor", "-s", "gyroscope", "-n", "5"]),
    ("termux_media_player", {"action": "pause"}, ["termux-media-player", "pause"]),  # optional skipped
    ("termux_notification", {"title": "Build done", "content": "compiled fine"},
     ["termux-notification", "-t", "Build done", "-c", "compiled fine"]),
    ("termux_battery_status", {}, ["termux-battery-status"]),
    ("termux_clipboard_set", {"text": "hello world"},
     ["termux-clipboard-set", "hello world"]),
    # Injection attempt stays ONE argv element: no shell ever sees this.
    ("termux_clipboard_set", {"text": "x; rm -rf / #"},
     ["termux-clipboard-set", "x; rm -rf / #"]),
    ("termux_clipboard_set", {"text": "$(whoami)`id`"},
     ["termux-clipboard-set", "$(whoami)`id`"]),
    ("termux_clipboard_set", {"text": "a b\tc\nd"}, ["termux-clipboard-set", "a b\tc\nd"]),
]

fails = 0
for name, args, want in cases:
    spec = tools[name]["exec"]
    try:
        got = build_argv(spec, args)
    except Exception as exc:
        print(f"FAIL {name}{args}: raised {exc}")
        fails += 1
        continue
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}{json.dumps(args)[:44]:46} -> {got}")

# Missing required argument must be caught, not silently dropped.
spec = tools["termux_brightness"]["exec"]
try:
    build_argv(spec, {})
    print("FAIL missing-required: no error raised")
    fails += 1
except ValueError as exc:
    print(f"ok   missing-required caught: {exc}")

# Execution layer, using a command that exists here.
echo_spec = {"cmd": "/bin/echo", "args": [{"lit": "hello"}, {"arg": "who"}]}
out = execute("demo", {"who": "$(id)"}, echo_spec, assume_yes=True, dry_run=False)
print(f"ok   execute(default) -> {out}")
assert out.get("stdout") == "hello $(id)", out     # not expanded: no shell
print("ok   no shell expansion in executed command")

print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
sys.exit(1 if fails else 0)
