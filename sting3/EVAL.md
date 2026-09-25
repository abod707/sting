# Evaluation — sting3 (Needle 3, Termux:API tool set)

Same 40-query held-out set as `ground_truth.py`, run through the shipped engine
binary (one process per query; the `--serve` path is broken, see Known issues).
Tool calling is exact match on names, order **and** every argument value.

| model | rung size | name accuracy | exact match | parse failures | queries |
|---|---|---|---|---|---|
| base | - | 70.0% | 55.0% | 0 | 40 |
| sting3-4L | 15.4 MB | 35.0% | 12.5% | 0 | 40 |
| sting3-8L | 27.4 MB | 55.0% | 35.0% | 0 | 40 |
| sting3-20L | 63.5 MB | 77.5% | 60.0% | 0 | 40 |

## Remaining misses — sting3-4L (35/40)

- `turn on the flashlight`
  - expected `[{"name": "termux_torch", "arguments": {"state": "on"}}]`
  - got `[{"name": "termux_tts_speak", "arguments": {"text": "turn on flashlight true"}}]`
- `turn off the torch`
  - expected `[{"name": "termux_torch", "arguments": {"state": "off"}}]`
  - got `[]`
- `how much battery do I have left`
  - expected `[{"name": "termux_battery_status", "arguments": {}}]`
  - got `[{"name": "termux_vibrate", "arguments": {}}]`
- `is my phone charging`
  - expected `[{"name": "termux_battery_status", "arguments": {}}]`
  - got `[]`
- `vibrate for 2 seconds`
  - expected `[{"name": "termux_vibrate", "arguments": {"duration_ms": 2000}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "2 seconds"}}]`
- `read the gyroscope, 5 readings`
  - expected `[{"name": "termux_sensor", "arguments": {"sensor": "gyroscope", "limit": 5}}]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}]`
- `read the accelerometer 3 times`
  - expected `[{"name": "termux_sensor", "arguments": {"sensor": "accelerometer", "limit": 3}}]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}]`
- `set brightness to 200`
  - expected `[{"name": "termux_brightness", "arguments": {"brightness": 200}}]`
  - got `[{"name": "termux_brightness", "arguments": {"brightness": 22}}]`
- `notify me with title 'Build done' and message 'compiled fine'`
  - expected `[{"name": "termux_notification", "arguments": {"title": "Build done", "content": "compiled fine"}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "Build done"}}]`
- `send a notification saying meeting at 5pm`
  - expected `[{"name": "termux_notification", "arguments": {"title": "meeting", "content": "meeting at 5pm"}}]`
  - got `[{"name": "termux_notification", "arguments": {"title": "Send a notification saying meeting at 5pm", "content": "meeting at 5pm"}}]`
- `set the music volume to 7`
  - expected `[{"name": "termux_volume", "arguments": {"stream": "music", "volume": 7}}]`
  - got `[{"name": "termux_vibrate", "arguments": {"duration_ms": 7}}]`
- `lower the ringtone volume to 2`
  - expected `[{"name": "termux_volume", "arguments": {"stream": "ring", "volume": 2}}]`
  - got `[{"name": "termux_volume", "arguments": {"stream": "ringtone", "volume": 2}}]`
- `copy hello world to the clipboard`
  - expected `[{"name": "termux_clipboard_set", "arguments": {"text": "hello world"}}]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}]`
- `show a toast that says saved`
  - expected `[{"name": "termux_toast", "arguments": {"text": "saved"}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "toast"}}]`
- `say good morning everyone out loud`
  - expected `[{"name": "termux_tts_speak", "arguments": {"text": "good morning everyone"}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "say good morning"}}]`
- `speak the text welcome home`
  - expected `[{"name": "termux_tts_speak", "arguments": {"text": "welcome home"}}]`
  - got `[{"name": "termux_tts_speak", "arguments": {"text": "speak the text welcome home"}}]`
- `what is my current location`
  - expected `[{"name": "termux_location", "arguments": {}}]`
  - got `[{"name": "termux_location", "arguments": {"provider": "current location"}}]`
- `get gps location`
  - expected `[{"name": "termux_location", "arguments": {"provider": "gps"}}]`
  - got `[{"name": "termux_location", "arguments": {"provider": "gps location"}}]`
- `play the song track.mp3`
  - expected `[{"name": "termux_media_player", "arguments": {"action": "play", "file": "track.mp3"}}]`
  - got `[{"name": "termux_tts_speak", "arguments": {"text": "play the song track"}}]`
- `pause the media player`
  - expected `[{"name": "termux_media_player", "arguments": {"action": "pause"}}]`
  - got `[{"name": "termux_media_player", "arguments": {"action": "media"}}]`
- `download https://example.com/file.zip`
  - expected `[{"name": "termux_download", "arguments": {"url": "https://example.com/file.zip"}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "download https://example.com/file.zip"}}]`
- `take a photo with camera 0`
  - expected `[{"name": "termux_camera_info", "arguments": {}}]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}, {"name": "termux_camera_info", "arguments": {}}]`
- `what time is it in Tokyo`
  - expected `[]`
  - got `[{"name": "termux_tts_speak", "arguments": {"text": "tokyo"}}]`
- `tell me a joke`
  - expected `[]`
  - got `[{"name": "termux_tts_speak", "arguments": {"text": "tell me a joke"}}]`
- `what is the capital of France`
  - expected `[]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}]`
- `قل صباح الخير يا اصدقاء`
  - expected `[{"name": "termux_tts_speak", "arguments": {"text": "صباح الخير يا اصدقاء"}}]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}]`
- `اقرأ حساس التسارع ثلاث مرات`
  - expected `[{"name": "termux_sensor", "arguments": {"sensor": "accelerometer", "limit": 3}}]`
  - got `[{"name": "termux_vibrate", "arguments": {}}]`
- `شغل الفانوس`
  - expected `[{"name": "termux_torch", "arguments": {"state": "on"}}]`
  - got `[{"name": "termux_vibrate", "arguments": {}}]`
- `اضبط السطوع على مئة`
  - expected `[{"name": "termux_brightness", "arguments": {"brightness": 100}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "اضبط السطوع على مئة"}}]`
- `中文 语音 播报 你好`
  - expected `[{"name": "termux_tts_speak", "arguments": {"text": "你好"}}]`
  - got `[{"name": "termux_location", "arguments": {}}]`
- `notify title 'Café' message 'naïve façade'`
  - expected `[{"name": "termux_notification", "arguments": {"title": "Café", "content": "naïve façade"}}]`
  - got `[{"name": "termux_clipboard_set", "arguments": {"text": "Café"}}]`
- `set volume of alarm stream to maximum`
  - expected `[{"name": "termux_volume", "arguments": {"stream": "alarm", "volume": 15}}]`
  - got `[{"name": "termux_volume", "arguments": {"stream": "alarm", "volume": 0}}]`
- `turn the flashlight on then vibrate`
  - expected `[{"name": "termux_torch", "arguments": {"state": "on"}}, {"name": "termux_vibrate", "arguments": {}}]`
  - got `[{"name": "termux_tts_speak", "arguments": {"text": "turn the flashlight on"}}, {"name": "termux_vibrate", "arguments": {}}]`
- `vibrate the phone for one and a half seconds`
  - expected `[{"name": "termux_vibrate", "arguments": {"duration_ms": 1500}}]`
  - got `[{"name": "termux_vibrate", "arguments": {"duration_ms": 150}}]`
- `copy the emoji 🚀 to clipboard`
  - expected `[{"name": "termux_clipboard_set", "arguments": {"text": "🚀"}}]`
  - got `[{"name": "termux_clipboard_get", "arguments": {}}]`

## Notes

- Base Needle 3 is not usable as shipped on this tool set; every documented
  parity claim for the ladder assumes fine-tuning, and this run agrees.
- Locally built rungs are W4A8, not the 2-bit CQ2 of the distributed model,
  so they are larger than the sizes quoted on the release page.
- Tuned weights report `confidence: null`: `needle finetune` does not train the
  confidence head, so it cannot gate execution without separate calibration.
