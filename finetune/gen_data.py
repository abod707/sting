#!/usr/bin/env python3
"""Synthetic dataset generator for finetuning Needle on Termux:API + generic MCP-style tools.

Stdlib only. Deterministic under --seed.

Output: JSONL with fields {query, tools, answers} in needle's finetune format.
  - tools:   JSON-encoded list of tool schemas (needle-style parameters)
  - answers: JSON-encoded list of {"name", "arguments"} calls ([] for no-op)

Design notes:
  * Each tool has a pool of query templates; values are sampled coherently so the
    answer's arguments are literally derivable from the query text (a 26M model
    copies, it doesn't reason). A few curated conversions (seconds->ms,
    "max volume"->15) appear as small, regular slices.
  * Distractor tools are sampled with a bias toward confusable siblings
    (torch vs toggle_lights, toast vs notification, ...) to teach hard contrasts.
  * ~8% no-tool examples (answers []), ~3% underspecified (required arg absent -> []),
    ~4% dual-call examples.
  * Some templates exclude specific distractors that would make the query ambiguous
    (e.g. implicit "it's dark in here, light please" excludes toggle_lights).
"""
import argparse
import json
import random

# ---------------------------------------------------------------------------
# Tool schemas — needle-style: parameters = {name: {type, description, required}}
# ---------------------------------------------------------------------------

TERMUX_TOOLS = {
    "termux_battery_status": {
        "description": "Get the battery status of the device: percentage, temperature, health and charging state.",
        "parameters": {},
    },
    "termux_torch": {
        "description": "Turn the device flashlight (torch) on or off.",
        "parameters": {
            "state": {"type": "string", "description": "Either on or off.", "required": True},
        },
    },
    "termux_brightness": {
        "description": "Set the screen brightness of the device display.",
        "parameters": {
            "brightness": {"type": "integer", "description": "Brightness level from 0 to 255.", "required": True},
        },
    },
    "termux_vibrate": {
        "description": "Vibrate the device.",
        "parameters": {
            "duration_ms": {"type": "integer", "description": "Vibration duration in milliseconds.", "required": False},
        },
    },
    "termux_notification": {
        "description": "Display a system notification with a title and message body.",
        "parameters": {
            "title": {"type": "string", "description": "Notification title.", "required": True},
            "content": {"type": "string", "description": "Notification body text.", "required": True},
        },
    },
    "termux_toast": {
        "description": "Show a short transient popup (toast) message on screen.",
        "parameters": {
            "text": {"type": "string", "description": "Text to show in the toast.", "required": True},
        },
    },
    "termux_clipboard_get": {
        "description": "Read the current contents of the system clipboard.",
        "parameters": {},
    },
    "termux_clipboard_set": {
        "description": "Set the contents of the system clipboard.",
        "parameters": {
            "text": {"type": "string", "description": "Text to place on the clipboard.", "required": True},
        },
    },
    "termux_tts_speak": {
        "description": "Speak text out loud using the device text-to-speech engine.",
        "parameters": {
            "text": {"type": "string", "description": "Text to speak.", "required": True},
            "language": {"type": "string", "description": "Language code such as en or ar.", "required": False},
        },
    },
    "termux_sensor": {
        "description": "Read values from a hardware sensor on the device.",
        "parameters": {
            "sensor": {"type": "string", "description": "Sensor name: accelerometer, gyroscope, light, proximity, pressure, magnetic_field or gravity.", "required": True},
            "limit": {"type": "integer", "description": "Number of readings to take.", "required": False},
        },
    },
    "termux_location": {
        "description": "Get the current geographic location of the device.",
        "parameters": {
            "provider": {"type": "string", "description": "Location provider: gps, network or passive.", "required": False},
            "request": {"type": "string", "description": "Request type: once, last or updates.", "required": False},
        },
    },
    "termux_wifi_connectioninfo": {
        "description": "Get information about the current wifi connection such as network name and signal.",
        "parameters": {},
    },
    "termux_volume": {
        "description": "Set the volume of an audio stream on the device.",
        "parameters": {
            "stream": {"type": "string", "description": "Audio stream: music, ring, alarm, notification, system or call.", "required": True},
            "volume": {"type": "integer", "description": "Volume level from 0 to 15.", "required": True},
        },
    },
    "termux_media_player": {
        "description": "Control media playback on the device: play a file, pause, stop or get info.",
        "parameters": {
            "action": {"type": "string", "description": "One of play, pause, stop or info.", "required": True},
            "file": {"type": "string", "description": "Path of the media file to play.", "required": False},
        },
    },
    "termux_download": {
        "description": "Download a file from a URL using the system download manager.",
        "parameters": {
            "url": {"type": "string", "description": "URL of the file to download.", "required": True},
            "title": {"type": "string", "description": "Title to show for the download.", "required": False},
        },
    },
    "termux_camera_info": {
        "description": "Get information about the cameras available on the device.",
        "parameters": {},
    },
}

GENERIC_TOOLS = {
    "get_weather": {
        "description": "Get current weather for a city.",
        "parameters": {
            "location": {"type": "string", "description": "City name.", "required": True},
        },
    },
    "set_timer": {
        "description": "Set a countdown timer.",
        "parameters": {
            "duration": {"type": "string", "description": "Timer duration, e.g. 10 minutes.", "required": True},
            "label": {"type": "string", "description": "Optional label for the timer.", "required": False},
        },
    },
    "set_alarm": {
        "description": "Set an alarm for a specific time of day.",
        "parameters": {
            "time": {"type": "string", "description": "Alarm time such as 07:30 or 6 am.", "required": True},
            "label": {"type": "string", "description": "Optional alarm label.", "required": False},
        },
    },
    "send_message": {
        "description": "Send a text message to a contact.",
        "parameters": {
            "recipient": {"type": "string", "description": "Name of the contact to message.", "required": True},
            "message": {"type": "string", "description": "The message text to send.", "required": True},
        },
    },
    "create_calendar_event": {
        "description": "Create a calendar event.",
        "parameters": {
            "title": {"type": "string", "description": "Event title.", "required": True},
            "date": {"type": "string", "description": "Event date such as tomorrow or June 3.", "required": True},
            "time": {"type": "string", "description": "Event start time.", "required": False},
        },
    },
    "play_music": {
        "description": "Play a song, artist or playlist on the music streaming app.",
        "parameters": {
            "song": {"type": "string", "description": "Song title to play.", "required": False},
            "artist": {"type": "string", "description": "Artist name.", "required": False},
            "playlist": {"type": "string", "description": "Playlist name.", "required": False},
        },
    },
    "search_web": {
        "description": "Search the web for information.",
        "parameters": {
            "query": {"type": "string", "description": "The search query.", "required": True},
        },
    },
    "get_stock_price": {
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {
            "symbol": {"type": "string", "description": "Ticker symbol such as AAPL.", "required": True},
        },
    },
    "toggle_lights": {
        "description": "Turn smart home room lights on or off.",
        "parameters": {
            "state": {"type": "string", "description": "Either on or off.", "required": True},
            "room": {"type": "string", "description": "Room name such as kitchen.", "required": False},
        },
    },
    "set_thermostat": {
        "description": "Set the target temperature of the smart thermostat.",
        "parameters": {
            "temperature": {"type": "integer", "description": "Target temperature in degrees.", "required": True},
        },
    },
    "get_directions": {
        "description": "Get directions to a destination.",
        "parameters": {
            "destination": {"type": "string", "description": "Where to go.", "required": True},
            "mode": {"type": "string", "description": "Travel mode: driving, walking or transit.", "required": False},
        },
    },
    "create_note": {
        "description": "Save a note with some content.",
        "parameters": {
            "content": {"type": "string", "description": "The note content.", "required": True},
            "title": {"type": "string", "description": "Optional note title.", "required": False},
        },
    },
    "convert_currency": {
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "amount": {"type": "number", "description": "Amount to convert.", "required": True},
            "from_currency": {"type": "string", "description": "Source currency code such as USD.", "required": True},
            "to_currency": {"type": "string", "description": "Target currency code such as EUR.", "required": True},
        },
    },
    "translate_text": {
        "description": "Translate text into a target language.",
        "parameters": {
            "text": {"type": "string", "description": "Text to translate.", "required": True},
            "target_language": {"type": "string", "description": "Language to translate into.", "required": True},
        },
    },
}

ALL_TOOLS = {**TERMUX_TOOLS, **GENERIC_TOOLS}

# Confusable siblings: preferred hard-negative distractors
CONFUSABLE = {
    "termux_torch": ["toggle_lights", "termux_brightness"],
    "toggle_lights": ["termux_torch", "termux_brightness"],
    "termux_toast": ["termux_notification", "termux_tts_speak"],
    "termux_notification": ["termux_toast", "termux_tts_speak", "send_message"],
    "termux_tts_speak": ["termux_toast", "termux_notification", "translate_text"],
    "termux_brightness": ["termux_torch", "termux_volume", "toggle_lights"],
    "termux_volume": ["termux_brightness", "termux_media_player", "play_music"],
    "termux_media_player": ["play_music", "termux_volume"],
    "play_music": ["termux_media_player", "termux_volume"],
    "termux_clipboard_get": ["termux_clipboard_set", "create_note"],
    "termux_clipboard_set": ["termux_clipboard_get", "create_note", "send_message"],
    "create_note": ["termux_clipboard_set", "send_message"],
    "termux_sensor": ["get_weather", "termux_battery_status"],
    "termux_location": ["get_directions", "get_weather", "termux_wifi_connectioninfo"],
    "get_directions": ["termux_location", "search_web"],
    "termux_download": ["search_web", "termux_media_player"],
    "termux_battery_status": ["termux_wifi_connectioninfo", "termux_sensor", "termux_camera_info"],
    "termux_wifi_connectioninfo": ["termux_battery_status", "termux_location"],
    "termux_camera_info": ["termux_battery_status", "termux_sensor"],
    "termux_vibrate": ["termux_notification", "termux_torch"],
    "set_timer": ["set_alarm", "create_calendar_event"],
    "set_alarm": ["set_timer", "create_calendar_event"],
    "create_calendar_event": ["set_alarm", "set_timer", "create_note"],
    "send_message": ["termux_notification", "create_note", "translate_text"],
    "get_weather": ["termux_sensor", "search_web", "termux_location"],
    "search_web": ["get_weather", "get_stock_price", "translate_text"],
    "get_stock_price": ["convert_currency", "search_web"],
    "convert_currency": ["get_stock_price", "search_web"],
    "translate_text": ["termux_tts_speak", "search_web", "send_message"],
    "set_thermostat": ["get_weather", "toggle_lights"],
}

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------

CITIES = ["Riyadh", "Jeddah", "Dammam", "Mecca", "Medina", "Abha", "Dubai", "Cairo",
          "London", "Tokyo", "Paris", "Berlin", "New York", "San Francisco", "Istanbul",
          "Sydney", "Toronto", "Singapore", "Mumbai", "Amman", "Doha", "Kuwait City",
          "Casablanca", "Oslo", "Seoul", "Chicago", "Madrid", "Rome", "Bangkok", "Nairobi"]

NAMES = ["Ahmed", "Sara", "Omar", "Layla", "Ali", "Noura", "Fatima", "Khalid", "Huda",
         "John", "Maria", "Emma", "Mike", "Yuki", "Chen", "Priya", "Carlos", "Anna",
         "mom", "dad", "my brother", "my sister", "the landlord", "coach Salem"]

NOTIF_PAIRS = [
    ("Build done", "Taskforge compiled successfully"),
    ("Build failed", "3 errors in sensor.rs"),
    ("Reminder", "Stand up and stretch"),
    ("Water break", "Drink a glass of water"),
    ("Backup complete", "142 files synced to the server"),
    ("Rust drill", "Chapter 4 ownership drill is due"),
    ("Download finished", "video.mp4 saved to Downloads"),
    ("Meeting", "Standup starts in 5 minutes"),
    ("Laundry", "The washing machine is done"),
    ("Focus over", "Pomodoro session complete"),
    ("Tea time", "The kettle should be ready"),
    ("Deploy", "Version 2.4.1 is live"),
    ("Prayer time", "Maghrib in 10 minutes"),
    ("Charge me", "Battery is getting low"),
    ("Groceries", "Pick up milk and eggs on the way home"),
    ("Server alert", "CPU usage above 90 percent"),
    ("Workout", "Time for the evening run"),
]

SHORT_TEXTS = [
    "hello world", "on my way", "be right back", "meeting moved to 3 pm",
    "the wifi password is sunflower42", "call me when you land",
    "git pull before you start", "dinner at 8", "left the keys under the mat",
    "cargo build passed", "see you at the gym", "invoice sent",
    "الاجتماع غدا الساعة العاشرة", "تم حفظ الملف", "أهلاً وسهلاً",
]

CLIP_TEXTS = [
    "https://github.com/abod707/artificial-horizon",
    "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html",
    "SELECT * FROM tasks WHERE done = 0;",
    "cargo run --release",
    "meet at gate B7, flight SV1021",
    "IBAN SA44 2000 0001 2345 6789 1234",
    "the quick brown fox jumps over the lazy dog",
    "ssh user@192.168.1.44",
    "TODO: refactor sensor parser",
    "شارع الملك فهد، الرياض",
]

TTS_TEXTS = [
    "Dinner is ready", "Time to take a break", "You have a new message",
    "The build has finished", "Focus session starting now", "Good morning",
    "Battery is low, plug in the charger", "It is time to leave for the airport",
    "مرحبا بك", "حان وقت الاستراحة", "تم إنجاز المهمة بنجاح",
]

SENSORS = ["accelerometer", "gyroscope", "light", "proximity", "pressure", "magnetic_field", "gravity"]

MEDIA_FILES = [
    "/sdcard/Music/lofi_beats.mp3", "/sdcard/Music/track01.mp3",
    "/sdcard/Download/podcast_ep12.mp3", "/sdcard/Music/quran/al_fatiha.mp3",
    "/sdcard/Recordings/memo_2026_07_01.m4a", "/sdcard/Music/rock/thunder.mp3",
]

URLS = [
    "https://example.com/report.pdf",
    "https://static.rust-lang.org/dist/rust-1.88.0-aarch64-linux-android.tar.gz",
    "https://speed.hetzner.de/100MB.bin",
    "https://example.org/photos/album.zip",
    "https://mirrors.kernel.org/ubuntu/pool/main/v/vim/vim_9.1.tar.xz",
    "https://cdn.example.net/wallpapers/nebula.jpg",
]

SONGS = ["Bohemian Rhapsody", "Take Five", "Clair de Lune", "Lose Yourself",
         "Hotel California", "Fairuz morning songs", "Blinding Lights"]
ARTISTS = ["Queen", "Fairuz", "Miles Davis", "Adele", "Umm Kulthum", "Hans Zimmer", "Daft Punk"]
PLAYLISTS = ["deep focus", "workout mix", "morning coffee", "road trip", "coding beats"]

STOCKS = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META", "2222.SR"]
CURRENCIES = ["USD", "EUR", "SAR", "GBP", "JPY", "AED", "EGP", "TRY", "INR", "CAD"]
LANGS = ["Arabic", "English", "French", "Spanish", "Japanese", "German", "Turkish"]
ROOMS = ["kitchen", "bedroom", "living room", "office", "hallway", "garage"]
TIMER_DURS = ["5 minutes", "10 minutes", "15 minutes", "25 minutes", "45 minutes",
              "1 hour", "90 seconds", "2 hours", "30 minutes", "20 minutes"]
ALARM_TIMES = ["06:00", "06:30", "07:00", "07:15", "5:45 am", "8 am", "9:30 pm", "22:00", "noon"]
DATES = ["tomorrow", "next Monday", "July 20", "this Friday", "August 3", "next weekend", "the 15th"]
EVENT_TITLES = ["dentist appointment", "team standup", "gym session", "flight to Jeddah",
                "Rust study group", "project review", "car service", "dinner with Omar"]
SEARCHES = ["best mechanical keyboards under 100 dollars", "rust borrow checker explained",
            "how to fix a leaking tap", "flights from Riyadh to Cairo in August",
            "difference between GQA and MQA attention", "ratatui custom widget tutorial",
            "population of Istanbul", "who invented the transistor",
            "capital of Kazakhstan", "how tall is Kilimanjaro"]
NOTES = ["buy AA batteries", "renew car registration before Thursday",
         "idea: attitude indicator should smooth pitch with EMA",
         "book title: The Pragmatic Programmer", "passport expires in October",
         "milestone M6: add due dates to Taskforge", "call the plumber about the heater"]

POLITE_PREFIX = ["", "", "", "", "please ", "can you ", "could you ", "hey, ", "yo, ", "would you "]
SUFFIX = ["", "", "", "", " please", " now", " for me", ", thanks"]


_INTERROGATIVE_STARTS = ("what", "what's", "how", "how's", "where", "which", "who",
                         "is ", "are ", "am ", "do ", "does", "did", "can ", "could",
                         "will", "would", "why", "when", "i ", "i'm", "i need", "i want",
                         "i feel", "i dropped", "too ", "it's", "it is", "the ", "never",
                         "thanks", "hello", "good", "hey", "yo", "stop blinding")

def _case_sensitive_args(args):
    """String arg values whose casing matters (would break under q.lower())."""
    vals = []
    for v in (args or {}).values():
        if isinstance(v, str) and v != v.lower():
            vals.append(v)
    return vals

def dress(rng, core, allow_prefix=True, allow_suffix=True, args=None):
    """Add natural politeness/casing noise around a core query."""
    q = core
    starts_interrogative = q.lower().startswith(_INTERROGATIVE_STARTS)
    cs_args = _case_sensitive_args(args)
    if allow_prefix and not starts_interrogative:
        p = rng.choice(POLITE_PREFIX)
        if p:
            first_is_arg = any(q.startswith(v) for v in cs_args)
            if q and q[0].isupper() and not first_is_arg:
                q = p + q[0].lower() + q[1:]
            else:
                q = p + q
    if allow_suffix and rng.random() < 0.35:
        q = q + rng.choice(SUFFIX)
    r = rng.random()
    # full-lowercase only when no arg value's casing would be corrupted
    if r < 0.18 and not _case_sensitive_args(args):
        q = q.lower()
    if rng.random() < 0.5 and not q.endswith(("?", ".", "!")):
        q += rng.choice(["", "", "?", ".", "!"]) if "?" in q or " what" in q else rng.choice(["", "", "."])
    # rare light typo: swap two adjacent letters inside one word (never inside arg values)
    if rng.random() < 0.03:
        arg_text = " ".join(str(v) for v in (args or {}).values())
        words = q.split(" ")
        cand = [i for i, w in enumerate(words)
                if len(w) >= 5 and w.isalpha() and w not in arg_text]
        if cand:
            i = rng.choice(cand)
            w = list(words[i])
            j = rng.randrange(1, len(w) - 2)
            w[j], w[j + 1] = w[j + 1], w[j]
            words[i] = "".join(w)
            q = " ".join(words)
    return q


# ---------------------------------------------------------------------------
# Per-tool example generators. Each returns (query, args_dict, exclude_set)
# ---------------------------------------------------------------------------

def gen_battery(rng):
    qs = ["How much battery do I have left", "What's my battery level", "Battery status",
          "battery?", "Am I charging", "Check the battery", "Is the phone charging",
          "How hot is the battery", "What percent is the battery at", "Battery health check",
          "how much charge is left", "Give me the battery stats"]
    return rng.choice(qs), {}, set()

def gen_torch(rng):
    state = rng.choice(["on", "off"])
    if state == "on":
        qs = ["Turn on the flashlight", "Turn the torch on", "Torch on", "Flashlight on",
              "Switch on the flashlight", "I need the flashlight", "Enable the torch",
              "Put the flashlight on", "Light up the torch"]
        implicit = ["It's pitch dark in here, give me some light", "Too dark, light please",
                    "I dropped my keys in the dark, help me see"]
    else:
        qs = ["Turn off the flashlight", "Torch off", "Kill the flashlight", "Flashlight off",
              "Switch the torch off", "Turn the torch off", "Disable the flashlight",
              "I'm done with the torch, turn it off"]
        implicit = ["Stop blinding me, turn that light off", "The torch is draining battery, kill it"]
    if rng.random() < 0.2:
        return rng.choice(implicit), {"state": state}, {"toggle_lights", "termux_brightness"}
    return rng.choice(qs), {"state": state}, set()

def gen_brightness(rng):
    if rng.random() < 0.12:
        # curated named levels
        word, val = rng.choice([("max", 255), ("maximum", 255), ("full", 255), ("zero", 0)])
        return f"Set the screen brightness to {word}", {"brightness": val}, set()
    val = rng.choice([10, 20, 30, 40, 50, 60, 80, 100, 120, 128, 150, 180, 200, 220, 240, 255])
    qs = [f"Set brightness to {val}", f"Set the screen brightness to {val}",
          f"Change brightness to {val}", f"Dim the screen to {val}" if val < 100 else f"Brighten the screen to {val}",
          f"screen brightness {val}", f"Make the display brightness {val}",
          f"Adjust the brightness level to {val}"]
    return rng.choice(qs), {"brightness": val}, set()

def gen_vibrate(rng):
    r = rng.random()
    if r < 0.35:
        # no args
        qs = ["Vibrate the phone", "Make the phone vibrate", "Buzz the phone", "vibrate",
              "Give me a vibration", "Make it buzz"]
        return rng.choice(qs), {}, set()
    if r < 0.75:
        ms = rng.choice([200, 300, 500, 700, 800, 1000, 1500, 2000, 2500, 3000])
        qs = [f"Vibrate for {ms} ms", f"Vibrate the phone for {ms} milliseconds",
              f"Buzz for {ms} ms", f"vibrate {ms}ms"]
        return rng.choice(qs), {"duration_ms": ms}, set()
    sec = rng.choice([1, 2, 3, 5])
    ms = sec * 1000
    qs = [f"Vibrate for {sec} second" + ("s" if sec > 1 else ""),
          f"Buzz the phone for {sec} second" + ("s" if sec > 1 else ""),
          f"Make it vibrate {sec} seconds"]
    return rng.choice(qs), {"duration_ms": ms}, set()

def gen_notification(rng):
    title, content = rng.choice(NOTIF_PAIRS)
    qs = [f"Notify me with title '{title}' and message '{content}'",
          f"Send a notification titled '{title}' saying '{content}'",
          f"Show a notification: title {title}, body {content}",
          f"Post a notification '{title}' with the text '{content}'",
          f"Create a notification called '{title}' that says '{content}'",
          f"notification with title '{title}' and content '{content}'"]
    return rng.choice(qs), {"title": title, "content": content}, set()

def gen_toast(rng):
    text = rng.choice(SHORT_TEXTS + [t[1] for t in NOTIF_PAIRS[:8]])
    qs = [f"Show a toast saying '{text}'", f"Pop up a toast with '{text}'",
          f"Toast '{text}'", f"Flash a quick message on screen: {text}",
          f"Show a popup that says '{text}'", f"Display a toast message '{text}'"]
    return rng.choice(qs), {"text": text}, set()

def gen_clip_get(rng):
    qs = ["What's on my clipboard", "Read the clipboard", "Show me what I copied",
          "Get the clipboard contents", "clipboard?", "What did I copy last",
          "Paste out whatever is on the clipboard", "Check the clipboard"]
    return rng.choice(qs), {}, set()

def gen_clip_set(rng):
    text = rng.choice(CLIP_TEXTS)
    qs = [f"Copy '{text}' to the clipboard", f"Put '{text}' on my clipboard",
          f"Set the clipboard to '{text}'", f"Save '{text}' to clipboard",
          f"clipboard set: {text}"]
    return rng.choice(qs), {"text": text}, set()

def gen_tts(rng):
    text = rng.choice(TTS_TEXTS)
    is_ar = any("؀" <= ch <= "ۿ" for ch in text)
    if rng.random() < 0.3:
        lang = "ar" if is_ar else rng.choice(["en", "en", "fr", "es"])
        lang_word = {"ar": "Arabic", "en": "English", "fr": "French", "es": "Spanish"}[lang]
        qs = [f"Say '{text}' in {lang_word}", f"Speak '{text}' in {lang_word}",
              f"Read this out loud in {lang_word}: {text}"]
        return rng.choice(qs), {"text": text, "language": lang}, {"translate_text"}
    qs = [f"Say '{text}'", f"Speak '{text}' out loud", f"Read this out loud: {text}",
          f"Say out loud: {text}", f"Use text to speech to say '{text}'",
          f"Have the phone say '{text}'"]
    return rng.choice(qs), {"text": text}, set()

def gen_sensor(rng):
    sensor = rng.choice(SENSORS)
    natural = {
        "accelerometer": ["Read the accelerometer", "Give me accelerometer values",
                          "What does the accelerometer say", "accelerometer readings"],
        "gyroscope": ["Read the gyroscope", "Get gyroscope data", "gyroscope values",
                      "What's the gyro reading"],
        "light": ["Check the light sensor", "How bright is it around me according to the sensor",
                  "Read the ambient light sensor", "light sensor value"],
        "proximity": ["Read the proximity sensor", "Check the proximity sensor",
                      "Is something near the proximity sensor"],
        "pressure": ["What's the air pressure", "Read the barometer", "Check the pressure sensor",
                     "Current atmospheric pressure from the sensor"],
        "magnetic_field": ["Read the magnetic field sensor", "Check the magnetometer",
                           "magnetic field values"],
        "gravity": ["Read the gravity sensor", "Get gravity sensor values"],
    }
    q = rng.choice(natural[sensor])
    args = {"sensor": sensor}
    if rng.random() < 0.3:
        n = rng.choice([3, 5, 10, 20])
        q += f", {n} readings" if rng.random() < 0.5 else f" and take {n} samples"
        args["limit"] = n
    excl = {"get_weather"} if sensor == "pressure" else set()
    return q, args, excl

def gen_location(rng):
    r = rng.random()
    if r < 0.4:
        qs = ["Where am I", "Get my location", "What's my current location",
              "Locate me", "Find my position", "current location?"]
        return rng.choice(qs), {}, {"get_directions"}
    if r < 0.7:
        qs = ["Get my location using GPS", "GPS fix please", "Get a gps location",
              "Locate me with gps"]
        return rng.choice(qs), {"provider": "gps"}, {"get_directions"}
    qs = ["What was my last known location", "Get the last known location",
          "Show my last recorded position"]
    return rng.choice(qs), {"request": "last"}, {"get_directions"}

def gen_wifi(rng):
    qs = ["What wifi am I connected to", "Show the current wifi connection info",
          "Which network am I on", "wifi details", "What's my wifi signal like",
          "Get wifi connection information", "Am I connected to wifi right now"]
    return rng.choice(qs), {}, set()

def gen_volume(rng):
    stream_words = {
        "music": ["media", "music", "media playback"],
        "ring": ["ring", "ringer", "ringtone"],
        "alarm": ["alarm"],
        "notification": ["notification", "notifications"],
        "system": ["system"],
        "call": ["call", "in-call"],
    }
    stream = rng.choice(list(stream_words))
    word = rng.choice(stream_words[stream])
    if rng.random() < 0.1:
        w, v = rng.choice([("max", 15), ("full", 15), ("mute", 0), ("zero", 0)])
        q = f"Set the {word} volume to {w}"
        return q, {"stream": stream, "volume": v}, set()
    vol = rng.randrange(0, 16)
    qs = [f"Set {word} volume to {vol}", f"Set the {word} volume to {vol}",
          f"Turn the {word} volume to {vol}", f"{word} volume {vol}",
          f"Change {word} volume to level {vol}"]
    return rng.choice(qs), {"stream": stream, "volume": vol}, set()

def gen_media(rng):
    r = rng.random()
    if r < 0.35:
        f = rng.choice(MEDIA_FILES)
        qs = [f"Play {f}", f"Play the file {f}", f"Start playing {f}",
              f"Open {f} in the media player"]
        return rng.choice(qs), {"action": "play", "file": f}, set()
    if r < 0.6:
        qs = ["Pause the music", "Pause playback", "Pause the audio", "pause"]
        return rng.choice(qs), {"action": "pause"}, {"play_music"}
    if r < 0.85:
        qs = ["Stop the music", "Stop playback", "Stop playing", "stop the audio"]
        return rng.choice(qs), {"action": "stop"}, {"play_music"}
    qs = ["What's currently playing", "Show media player status", "What is playing right now"]
    return rng.choice(qs), {"action": "info"}, {"play_music"}

def gen_download(rng):
    url = rng.choice(URLS)
    if rng.random() < 0.3:
        title = rng.choice(["rust toolchain", "monthly report", "wallpaper", "podcast episode", "backup archive"])
        qs = [f"Download {url} as '{title}'", f"Download {url} and call it '{title}'",
              f"Save {url} with the title '{title}'"]
        return rng.choice(qs), {"url": url, "title": title}, set()
    qs = [f"Download {url}", f"Download this file: {url}", f"Save {url} to my phone",
          f"Grab {url}", f"Fetch {url} for me"]
    return rng.choice(qs), {"url": url}, set()

def gen_camera_info(rng):
    qs = ["What cameras does this phone have", "Show camera info", "List the device cameras",
          "Get camera specifications", "How many cameras are on this device",
          "camera details"]
    return rng.choice(qs), {}, set()

# ---- generic tools ----

def gen_weather(rng):
    city = rng.choice(CITIES)
    qs = [f"What's the weather in {city}", f"Weather in {city}", f"How's the weather in {city} today",
          f"Is it hot in {city} right now", f"Give me the current weather for {city}",
          f"What's it like outside in {city}", f"weather {city}"]
    return rng.choice(qs), {"location": city}, set()

def gen_timer(rng):
    dur = rng.choice(TIMER_DURS)
    if rng.random() < 0.3:
        label = rng.choice(["tea", "pasta", "laundry", "workout", "pomodoro", "eggs", "bread in the oven"])
        qs = [f"Set a {dur} timer for {label}", f"Start a {dur} timer called {label}",
              f"Timer for {label}, {dur}"]
        return rng.choice(qs), {"duration": dur, "label": label}, set()
    qs = [f"Set a timer for {dur}", f"Start a {dur} timer", f"Countdown {dur}",
          f"timer {dur}", f"Give me a {dur} timer"]
    return rng.choice(qs), {"duration": dur}, set()

def gen_alarm(rng):
    t = rng.choice(ALARM_TIMES)
    if rng.random() < 0.3:
        label = rng.choice(["work", "gym", "flight", "meds", "school run", "fajr"])
        qs = [f"Set an alarm for {t} labeled {label}", f"Wake me at {t} for {label}",
              f"Alarm at {t} called {label}"]
        return rng.choice(qs), {"time": t, "label": label}, set()
    qs = [f"Set an alarm for {t}", f"Wake me up at {t}", f"Alarm at {t}",
          f"Set my alarm to {t}", f"I need an alarm at {t}"]
    return rng.choice(qs), {"time": t}, set()

def gen_message(rng):
    who = rng.choice(NAMES)
    msg = rng.choice(SHORT_TEXTS)
    qs = [f"Send a message to {who} saying '{msg}'", f"Text {who}: {msg}",
          f"Message {who} that {msg}" if not any(c in msg for c in "'") else f"Message {who}: '{msg}'",
          f"Tell {who} '{msg}'", f"Send '{msg}' to {who}"]
    return rng.choice(qs), {"recipient": who, "message": msg}, set()

def gen_event(rng):
    title = rng.choice(EVENT_TITLES)
    date = rng.choice(DATES)
    if rng.random() < 0.45:
        t = rng.choice(["9 am", "10:30", "2 pm", "16:00", "7 pm", "noon"])
        qs = [f"Add {title} to my calendar on {date} at {t}",
              f"Create a calendar event '{title}' on {date} at {t}",
              f"Schedule {title} for {date} at {t}"]
        return rng.choice(qs), {"title": title, "date": date, "time": t}, set()
    qs = [f"Add {title} to my calendar on {date}", f"Create an event '{title}' for {date}",
          f"Put {title} on the calendar for {date}", f"Schedule {title} on {date}"]
    return rng.choice(qs), {"title": title, "date": date}, set()

def gen_play_music(rng):
    r = rng.random()
    if r < 0.4:
        song = rng.choice(SONGS)
        qs = [f"Play {song}", f"Play the song {song}", f"Put on {song}", f"I want to hear {song}"]
        return rng.choice(qs), {"song": song}, {"termux_media_player"}
    if r < 0.7:
        artist = rng.choice(ARTISTS)
        qs = [f"Play some {artist}", f"Play music by {artist}", f"Put on some {artist} songs",
              f"I feel like listening to {artist}"]
        return rng.choice(qs), {"artist": artist}, {"termux_media_player"}
    pl = rng.choice(PLAYLISTS)
    qs = [f"Play my {pl} playlist", f"Start the {pl} playlist", f"Put on the {pl} playlist"]
    return rng.choice(qs), {"playlist": pl}, {"termux_media_player"}

def gen_search(rng):
    s = rng.choice(SEARCHES)
    qs = [f"Search for {s}", f"Search the web for {s}", f"Look up {s}",
          f"Google {s}", f"Find information about {s}", f"{s}?"]
    return rng.choice(qs), {"query": s}, set()

def gen_stock(rng):
    sym = rng.choice(STOCKS)
    qs = [f"What's the stock price of {sym}", f"Get the current price of {sym}",
          f"{sym} stock price", f"How is {sym} trading right now", f"Check {sym} for me"]
    return rng.choice(qs), {"symbol": sym}, set()

def gen_lights(rng):
    state = rng.choice(["on", "off"])
    if rng.random() < 0.55:
        room = rng.choice(ROOMS)
        qs = [f"Turn {state} the {room} lights", f"Switch the {room} lights {state}",
              f"{room} lights {state}", f"Turn the lights {state} in the {room}"]
        return rng.choice(qs), {"state": state, "room": room}, {"termux_torch"}
    qs = [f"Turn {state} the lights", f"Lights {state}", f"Switch the lights {state}",
          f"Turn the smart lights {state}"]
    return rng.choice(qs), {"state": state}, {"termux_torch"}

def gen_thermostat(rng):
    temp = rng.choice([18, 19, 20, 21, 22, 23, 24, 25, 26])
    qs = [f"Set the thermostat to {temp} degrees", f"Set the temperature to {temp}",
          f"Make it {temp} degrees inside", f"thermostat {temp}",
          f"Change the AC to {temp} degrees"]
    return rng.choice(qs), {"temperature": temp}, set()

def gen_directions(rng):
    dest = rng.choice(["the airport", "King Khalid Airport", "the nearest pharmacy", "downtown",
                       "the office", "Boulevard Riyadh City", "the gym", "the train station",
                       "Granada Mall", "the university"])
    if rng.random() < 0.35:
        mode = rng.choice(["driving", "walking", "transit"])
        qs = [f"Get me {mode} directions to {dest}", f"Directions to {dest} by {mode}",
              f"How do I get to {dest} {('on foot' if mode=='walking' else 'by ' + mode)}"]
        # 'on foot' keeps mode word out; keep args literal -> use mode word in 2 of 3
        q = rng.choice(qs[:2])
        return q, {"destination": dest, "mode": mode}, {"termux_location"}
    qs = [f"Directions to {dest}", f"Navigate to {dest}", f"How do I get to {dest}",
          f"Take me to {dest}", f"Route to {dest}"]
    return rng.choice(qs), {"destination": dest}, {"termux_location"}

def gen_note(rng):
    content = rng.choice(NOTES)
    if rng.random() < 0.25:
        title = rng.choice(["todo", "ideas", "shopping", "rust notes", "errands"])
        qs = [f"Save a note titled {title}: {content}", f"Create a note '{title}' saying {content}",
              f"Note under {title}: {content}"]
        return rng.choice(qs), {"content": content, "title": title}, set()
    qs = [f"Make a note: {content}", f"Note down {content}", f"Save a note that says {content}",
          f"Remember this: {content}", f"Jot down {content}"]
    return rng.choice(qs), {"content": content}, set()

def gen_currency(rng):
    amt = rng.choice([10, 25, 50, 75, 100, 200, 250, 500, 1000, 1500])
    a, b = rng.sample(CURRENCIES, 2)
    qs = [f"Convert {amt} {a} to {b}", f"How much is {amt} {a} in {b}",
          f"{amt} {a} to {b}", f"What's {amt} {a} worth in {b}",
          f"Exchange {amt} {a} into {b}"]
    return rng.choice(qs), {"amount": amt, "from_currency": a, "to_currency": b}, set()

def gen_translate(rng):
    text = rng.choice(["good morning", "where is the train station", "thank you very much",
                       "how much does this cost", "I would like a coffee", "see you tomorrow",
                       "the weather is nice today", "I am learning Rust"])
    lang = rng.choice(LANGS)
    qs = [f"Translate '{text}' to {lang}", f"How do you say '{text}' in {lang}",
          f"Translate this into {lang}: {text}", f"What is '{text}' in {lang}"]
    return rng.choice(qs), {"text": text, "target_language": lang}, {"termux_tts_speak"}

GENERATORS = {
    "termux_battery_status": gen_battery,
    "termux_torch": gen_torch,
    "termux_brightness": gen_brightness,
    "termux_vibrate": gen_vibrate,
    "termux_notification": gen_notification,
    "termux_toast": gen_toast,
    "termux_clipboard_get": gen_clip_get,
    "termux_clipboard_set": gen_clip_set,
    "termux_tts_speak": gen_tts,
    "termux_sensor": gen_sensor,
    "termux_location": gen_location,
    "termux_wifi_connectioninfo": gen_wifi,
    "termux_volume": gen_volume,
    "termux_media_player": gen_media,
    "termux_download": gen_download,
    "termux_camera_info": gen_camera_info,
    "get_weather": gen_weather,
    "set_timer": gen_timer,
    "set_alarm": gen_alarm,
    "send_message": gen_message,
    "create_calendar_event": gen_event,
    "play_music": gen_play_music,
    "search_web": gen_search,
    "get_stock_price": gen_stock,
    "toggle_lights": gen_lights,
    "set_thermostat": gen_thermostat,
    "get_directions": gen_directions,
    "create_note": gen_note,
    "convert_currency": gen_currency,
    "translate_text": gen_translate,
}

# ---------------------------------------------------------------------------
# No-tool and underspecified pools
# ---------------------------------------------------------------------------

# (query, tools to exclude from the sampled list so the query truly has no match)
NO_TOOL_QUERIES = [
    ("What's 45 times 12", set()),
    ("Tell me a joke", set()),
    ("Good morning", set()),
    ("Thanks, that's all", set()),
    ("Write a haiku about the desert", {"create_note"}),
    ("Who won the world cup in 2022", {"search_web"}),
    ("What's the capital of Japan", {"search_web"}),
    ("How do transformers work", {"search_web"}),
    ("Do you think pineapple belongs on pizza", set()),
    ("I'm bored", {"play_music", "search_web"}),
    ("Explain ownership in Rust in one line", {"search_web", "create_note"}),
    ("What should I cook tonight", {"search_web"}),
    ("hello", set()),
    ("Never mind, cancel that", set()),
    ("What day is it tomorrow", {"create_calendar_event", "search_web"}),
    ("Sing me a song", {"play_music", "termux_tts_speak", "termux_media_player"}),
    ("How are you doing", set()),
    ("Recommend a sci-fi movie", {"search_web"}),
    ("Why is the sky blue", {"search_web"}),
    ("Count from one to five", {"termux_tts_speak"}),
]

# Underspecified: required argument genuinely missing -> correct output is []
UNDERSPECIFIED = [
    ("Set the volume", "termux_volume"),
    ("Change the volume for me", "termux_volume"),
    ("Set the brightness", "termux_brightness"),
    ("Adjust my screen brightness", "termux_brightness"),
    ("Toggle the torch", "termux_torch"),
    ("Download the file", "termux_download"),
    ("Download it for me", "termux_download"),
    ("Set an alarm", "set_alarm"),
    ("Set a timer", "set_timer"),
    ("Send a message to Sara", "send_message"),
    ("Text Ahmed", "send_message"),
    ("Show me a notification", "termux_notification"),
    ("Translate this", "translate_text"),
    ("Convert some currency", "convert_currency"),
    ("What's the weather", "get_weather"),
    ("Get the stock price", "get_stock_price"),
]

# Dual-call patterns: (builder(rng) -> (query, [call1, call2], exclude))
def dual_notify_speak(rng):
    title, content = rng.choice(NOTIF_PAIRS)
    q = f"Notify me with title '{title}' and message '{content}', and read the message out loud"
    return q, [
        {"name": "termux_notification", "arguments": {"title": title, "content": content}},
        {"name": "termux_tts_speak", "arguments": {"text": content}},
    ], set()

def dual_torch_vibrate(rng):
    ms = rng.choice([500, 1000, 2000])
    q = f"Turn on the flashlight and vibrate for {ms} ms"
    return q, [
        {"name": "termux_torch", "arguments": {"state": "on"}},
        {"name": "termux_vibrate", "arguments": {"duration_ms": ms}},
    ], {"toggle_lights"}

def dual_battery_wifi(rng):
    q = rng.choice(["Check the battery and the wifi connection",
                    "Give me battery status and wifi info",
                    "Quick device check: battery and wifi"])
    return q, [
        {"name": "termux_battery_status", "arguments": {}},
        {"name": "termux_wifi_connectioninfo", "arguments": {}},
    ], set()

def dual_timer_music(rng):
    dur = rng.choice(TIMER_DURS)
    pl = rng.choice(PLAYLISTS)
    q = f"Set a timer for {dur} and play the {pl} playlist"
    return q, [
        {"name": "set_timer", "arguments": {"duration": dur}},
        {"name": "play_music", "arguments": {"playlist": pl}},
    ], {"termux_media_player"}

def dual_brightness_volume(rng):
    b = rng.choice([30, 50, 80, 120, 200])
    v = rng.randrange(0, 16)
    q = f"Set brightness to {b} and media volume to {v}"
    return q, [
        {"name": "termux_brightness", "arguments": {"brightness": b}},
        {"name": "termux_volume", "arguments": {"stream": "music", "volume": v}},
    ], set()

DUAL_BUILDERS = [dual_notify_speak, dual_torch_vibrate, dual_battery_wifi,
                 dual_timer_music, dual_brightness_volume]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def schema_of(name):
    t = ALL_TOOLS[name]
    return {"name": name, "description": t["description"], "parameters": t["parameters"]}

def sample_tools(rng, targets, exclude):
    """Build the tools list: target tool(s) + distractors (confusables preferred)."""
    n_extra = rng.choices([0, 1, 2, 3, 4, 5], weights=[8, 22, 26, 22, 14, 8])[0]
    chosen = list(targets)
    pool_all = [t for t in ALL_TOOLS if t not in chosen and t not in exclude]
    sibs = []
    for t in targets:
        sibs += [s for s in CONFUSABLE.get(t, []) if s not in exclude and s not in chosen]
    while n_extra > 0 and pool_all:
        if sibs and rng.random() < 0.5:
            pick = rng.choice(sibs)
        else:
            pick = rng.choice(pool_all)
        if pick not in chosen:
            chosen.append(pick)
            n_extra -= 1
        if pick in sibs:
            sibs.remove(pick)
        if pick in pool_all:
            pool_all.remove(pick)
    rng.shuffle(chosen)
    return [schema_of(t) for t in chosen]

def make_example(rng, query, calls, targets, exclude):
    tools = sample_tools(rng, targets, exclude)
    return {
        "query": query,
        "tools": json.dumps(tools, separators=(",", ":"), ensure_ascii=False),
        "answers": json.dumps(calls, separators=(",", ":"), ensure_ascii=False),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--per-termux", type=int, default=155)
    ap.add_argument("--per-generic", type=int, default=115)
    ap.add_argument("--no-tool", type=int, default=380)
    ap.add_argument("--underspec", type=int, default=150)
    ap.add_argument("--dual", type=int, default=190)
    ap.add_argument("--out", type=str, default="data.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    seen = set()

    def emit(row):
        key = (row["query"], row["tools"], row["answers"])
        if key in seen:
            return False
        seen.add(key)
        rows.append(row)
        return True

    # Single-call per-tool examples
    for name, gen in GENERATORS.items():
        target_n = args.per_termux if name in TERMUX_TOOLS else args.per_generic
        made, attempts = 0, 0
        while made < target_n and attempts < target_n * 30:
            attempts += 1
            q, call_args, excl = gen(rng)
            q = dress(rng, q, args=call_args)
            calls = [{"name": name, "arguments": call_args}]
            if emit(make_example(rng, q, calls, [name], excl)):
                made += 1

    # Dual-call examples
    made, attempts = 0, 0
    while made < args.dual and attempts < args.dual * 30:
        attempts += 1
        b = rng.choice(DUAL_BUILDERS)
        q, calls, excl = b(rng)
        targets = [c["name"] for c in calls]
        if emit(make_example(rng, q, calls, targets, excl)):
            made += 1

    # No-tool examples
    made, attempts = 0, 0
    while made < args.no_tool and attempts < args.no_tool * 30:
        attempts += 1
        q, excl = rng.choice(NO_TOOL_QUERIES)
        q = dress(rng, q, allow_prefix=False)
        n_tools = rng.choices([1, 2, 3, 4, 5], weights=[15, 28, 27, 18, 12])[0]
        pool = [t for t in ALL_TOOLS if t not in excl]
        chosen = rng.sample(pool, n_tools)
        tools = [schema_of(t) for t in chosen]
        row = {"query": q,
               "tools": json.dumps(tools, separators=(",", ":"), ensure_ascii=False),
               "answers": "[]"}
        if emit(row):
            made += 1

    # Underspecified examples (target tool IS present, required arg missing -> [])
    made, attempts = 0, 0
    while made < args.underspec and attempts < args.underspec * 30:
        attempts += 1
        q, tool = rng.choice(UNDERSPECIFIED)
        q = dress(rng, q)
        tools = sample_tools(rng, [tool], set())
        row = {"query": q,
               "tools": json.dumps(tools, separators=(",", ":"), ensure_ascii=False),
               "answers": "[]"}
        if emit(row):
            made += 1

    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    c = Counter()
    n_calls = Counter()
    for r in rows:
        calls = json.loads(r["answers"])
        n_calls[len(calls)] += 1
        if not calls:
            c["__no_tool__"] += 1
        for call in calls:
            c[call["name"]] += 1
    print(f"total examples: {len(rows)}")
    print(f"calls per example: {dict(sorted(n_calls.items()))}")
    print(f"tool coverage ({len(c)} buckets):")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v}")

    # Also write the schema packs for downstream use (Rust CLI config)
    with open("termux_tools.json", "w", encoding="utf-8") as f:
        json.dump([schema_of(t) for t in TERMUX_TOOLS], f, indent=2, ensure_ascii=False)
    with open("generic_tools.json", "w", encoding="utf-8") as f:
        json.dump([schema_of(t) for t in GENERIC_TOOLS], f, indent=2, ensure_ascii=False)
    print("\nwrote termux_tools.json, generic_tools.json")

if __name__ == "__main__":
    main()
