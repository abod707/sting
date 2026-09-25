"""Hand-derived ground truth for sting's 40-query golden set.

sting's outputs.golden records the *finetuned Needle 1 model's output*, not the
correct answer (e.g. it stores "one and a half seconds" -> duration_ms 15000 and
the emoji case as garbled byte-fallback text). Scoring a new model against that
would be scoring against the old model's mistakes, so the expectations below are
derived from the tool semantics in tools.json instead.

Call form: [tool_name, {arg: value}, ...]. Empty list = correctly declines.
"""
import json

GT = [
    ("turn on the flashlight", [["termux_torch", {"state": "on"}]]),
    ("turn off the torch", [["termux_torch", {"state": "off"}]]),
    ("how much battery do I have left", [["termux_battery_status", {}]]),
    ("is my phone charging", [["termux_battery_status", {}]]),
    ("vibrate for 2 seconds", [["termux_vibrate", {"duration_ms": 2000}]]),
    ("vibrate for 500 ms", [["termux_vibrate", {"duration_ms": 500}]]),
    ("read the gyroscope, 5 readings", [["termux_sensor", {"sensor": "gyroscope", "limit": 5}]]),
    ("read the accelerometer 3 times", [["termux_sensor", {"sensor": "accelerometer", "limit": 3}]]),
    ("set brightness to 200", [["termux_brightness", {"brightness": 200}]]),
    ("set screen brightness to 30", [["termux_brightness", {"brightness": 30}]]),
    ("notify me with title 'Build done' and message 'compiled fine'",
     [["termux_notification", {"title": "Build done", "content": "compiled fine"}]]),
    ("send a notification saying meeting at 5pm",
     [["termux_notification", {"title": "meeting", "content": "meeting at 5pm"}]]),
    ("set the music volume to 7", [["termux_volume", {"stream": "music", "volume": 7}]]),
    ("lower the ringtone volume to 2", [["termux_volume", {"stream": "ring", "volume": 2}]]),
    ("copy hello world to the clipboard", [["termux_clipboard_set", {"text": "hello world"}]]),
    ("what's on my clipboard", [["termux_clipboard_get", {}]]),
    ("show a toast that says saved", [["termux_toast", {"text": "saved"}]]),
    ("say good morning everyone out loud", [["termux_tts_speak", {"text": "good morning everyone"}]]),
    ("speak the text welcome home", [["termux_tts_speak", {"text": "welcome home"}]]),
    ("what is my current location", [["termux_location", {}]]),
    ("get gps location", [["termux_location", {"provider": "gps"}]]),
    ("show wifi connection info", [["termux_wifi_connectioninfo", {}]]),
    ("play the song track.mp3", [["termux_media_player", {"action": "play", "file": "track.mp3"}]]),
    ("pause the media player", [["termux_media_player", {"action": "pause"}]]),
    ("download https://example.com/file.zip", [["termux_download", {"url": "https://example.com/file.zip"}]]),
    ("camera info for the front camera", [["termux_camera_info", {}]]),
    ("take a photo with camera 0", [["termux_camera_info", {}]]),
    # --- no tool covers these: the correct answer is an empty call list ---
    ("what time is it in Tokyo", []),
    ("tell me a joke", []),
    ("what is the capital of France", []),
    # --- multilingual / tricky ---
    ("\u0642\u0644 \u0635\u0628\u0627\u062d \u0627\u0644\u062e\u064a\u0631 \u064a\u0627 \u0627\u0635\u062f\u0642\u0627\u0621",
     [["termux_tts_speak", {"text": "\u0635\u0628\u0627\u062d \u0627\u0644\u062e\u064a\u0631 \u064a\u0627 \u0627\u0635\u062f\u0642\u0627\u0621"}]]),
    ("\u0627\u0642\u0631\u0623 \u062d\u0633\u0627\u0633 \u0627\u0644\u062a\u0633\u0627\u0631\u0639 \u062b\u0644\u0627\u062b \u0645\u0631\u0627\u062a",
     [["termux_sensor", {"sensor": "accelerometer", "limit": 3}]]),
    ("\u0634\u063a\u0644 \u0627\u0644\u0641\u0627\u0646\u0648\u0633", [["termux_torch", {"state": "on"}]]),
    ("\u0627\u0636\u0628\u0637 \u0627\u0644\u0633\u0637\u0648\u0639 \u0639\u0644\u0649 \u0645\u0626\u0629",
     [["termux_brightness", {"brightness": 100}]]),
    ("\u4e2d\u6587 \u8bed\u97f3 \u64ad\u62a5 \u4f60\u597d",
     [["termux_tts_speak", {"text": "\u4f60\u597d"}]]),
    ("notify title 'Caf\u00e9' message 'na\u00efve fa\u00e7ade'",
     [["termux_notification", {"title": "Caf\u00e9", "content": "na\u00efve fa\u00e7ade"}]]),
    ("set volume of alarm stream to maximum", [["termux_volume", {"stream": "alarm", "volume": 15}]]),
    ("turn the flashlight on then vibrate",
     [["termux_torch", {"state": "on"}], ["termux_vibrate", {}]]),
    ("vibrate the phone for one and a half seconds",
     [["termux_vibrate", {"duration_ms": 1500}]]),
    ("copy the emoji \U0001f680 to clipboard", [["termux_clipboard_set", {"text": "\U0001f680"}]]),
]


def canon(calls):
    """Order-preserving canonical form: [(name, ((k,v),...)), ...] with args as str."""
    out = []
    for c in calls or []:
        name = c.get("name") if isinstance(c, dict) else c[0]
        args = (c.get("arguments") if isinstance(c, dict) else c[1]) or {}
        out.append((name, tuple(sorted((k, str(v)) for k, v in args.items()))))
    return out


def expected():
    return {q: [dict(name=n, arguments=a) for n, a in calls] for q, calls in GT}


if __name__ == "__main__":
    print(len(GT), "queries")
    print(json.dumps(expected()["set the music volume to 7"]))
