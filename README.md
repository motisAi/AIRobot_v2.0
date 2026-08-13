# Stella — Smart Self‑Learning AI Robot 🤖

Stella runs on a **Raspberry Pi 5 + Hailo‑10H**, with a USB camera (+ mic) and a
**5‑finger robotic hand** driven by an **ESP32‑S3 + PCA9685**. She recognises faces,
holds a natural spoken conversation, thinks with cloud + on‑device LLMs, sees through a
vision model, plays music, guards your home, gestures and mirrors your hand, has a
cheeky personality (and a memory for grudges), and talks to you on Telegram.

Everything is driven by one human‑editable file — **`config/config.yaml`** — and all
secrets live in **`.env`**.

---

## Table of contents
- [Capabilities](#capabilities)
- [Everyday commands](#everyday-commands)
- [The robotic hand](#the-robotic-hand)
- [Home guard](#home-guard)
- [Configuration](#configuration-configconfigyaml)
- [Secrets (.env)](#secrets-env)
- [Hardware setup](#hardware-setup)
- [Run & ops](#run--ops)
- [Known quirks](#known-quirks--notes)

---

## Capabilities

### 🎙️ Voice & conversation
- **Wake word** ("Stella" / "hey Stella"), fully offline (Vosk).
- **Two microphones, two roles** — the camera mic listens for the wake word; the USB
  mic captures your command (voice‑activity detection ends the sentence).
- **Multi‑turn conversation** — greets you, chats, and after a few seconds of silence
  asks "anything else?", then says goodbye and returns to listening.
- **Natural neural voice** — Piper (offline). Speech‑to‑text: Google online with an
  offline Vosk fallback.
- **Language switch** — `behavior.language: en | he` flips listening + speaking between
  **English and Hebrew** (her brain understands both).
- **Clean pages** — each conversation starts fresh (no old topics bleeding in); the
  transcript is still saved to her memory DB.

### 🧠 Brain (LLM) with graceful fallback
Tried in order, so she stays responsive:
1. **Groq Llama‑3.3‑70B** — smartest (free tier; daily token cap).
2. **Groq GPT‑OSS‑20B** — kicks in when the 70B hits its rate‑limit (fresh, separate limit).
3. **On‑device Hailo‑10H NPU** — free/offline fallback *(currently unstable on this box —
   see [Known quirks](#known-quirks--notes))*.

Extras: a **60s cooldown** skips a rate‑limited provider instead of retrying it every
turn; **per‑user memory** (names, preferences) in a SQLite DB; and **anti‑fabrication**
rules so she won't invent facts or fire tools on vague "yes/ok" filler.

### 🤖 Agent tools (she chooses when to use them, and chains them)
| Tool | What it does |
|------|--------------|
| `get_time` | Current time; pass a timezone for other cities (e.g. Tokyo → `Asia/Tokyo`) |
| `get_weather` | Live weather for a place (wttr.in) |
| `web_search` | Look things up on the web (DuckDuckGo, free) |
| `look` | Describe what the camera sees / "what am I holding?" (Moondream VLM) |
| `set_reminder` | "Remind me in 10 minutes to…" (spoken when due; survives restarts) |
| `control_device` | "Turn on the light" → microcontroller (ESP32/Pi Zero) |
| `set_guard_mode` | Arm / disarm home guard |
| `send_telegram` | Send you a text message |
| `send_photo` | Snap the camera and send it to your phone |
| `play_music` / `stop_music` | Play/stop a song from YouTube |
| `do_gesture` | Hand gestures: wave, thumbs_up, point, peace, fist, count, middle_finger… |

### 👁️ Vision
- **Face recognition + enrollment** — recognises people by face, asks a new person's
  name and remembers them; master authentication for privileged actions. Tunable
  threshold + an "identity memory" window so she doesn't forget you mid‑chat.
- **Scene / object understanding** — Moondream cloud VLM answers "what do you see?".
- **Auto white‑balance / exposure** for accurate colours.

### ✋ Robotic hand (5 fingers)
- **Gestures:** `fist`, `open`, `hello` wave, `point`, `peace`, `thumbs_up`, `count N`,
  and `middle_finger` (plus a **hold‑until‑told** version).
- **Copy‑my‑hand mirror** — say *"copy my hand"* and she mirrors your finger positions
  in real time (MediaPipe hand‑landmarks → her servos). *"stop copying"* ends it.
- **Homes to a closed fist** on boot (its known start position).

### 🎭 Personality
- **Greets with a wave**, waves goodbye, **thumbs‑up** when pleased.
- **Insult her** → 🖕 + a random sassy comeback ("Back at you", "Go look in the mirror"…),
  and she **holds a grudge** (saved to that person's memory) — turning cold and reluctant
  until they **apologise**, then she forgives.
- Ask *"what would you do if I called you stupid?"* → she **demonstrates** (no real grudge).

### 🛡️ Home guard / security mode
- **Motion & body detection** — while armed, alerts on **movement or a human body**
  (no clear face needed). Face recognition only *suppresses* alerts (so she ignores you),
  and she **auto‑disarms + greets** when she recognises you.
- **Interactive phone alert:** 📸 photo → *"Do you recognise this person? YES/NO"* →
  if **NO** → *"Sound the alarm? YES"* → she **screams** a loud, distorted alarm voice:
  *"THIEF! GET OUT NOW!"* 🚨
- Arm/disarm by **voice** (arming goes quiet after), **Telegram**, or dashboard. Tunable
  sensitivity; disarming is master‑only (recent face, pass‑phrase, or the phone).

### 🎵 Music
- **"Play \<song\>"** → she finds it on YouTube, confirms, and plays it out the speaker
  (yt‑dlp + ffmpeg — no mpv needed).
- Live control: **louder / quieter / set volume to N / pause / resume / stop / what's playing?**
- **Auto‑ducks** and frees the speaker while she talks, then resumes.

### 📱 Telegram (two‑way)
- Guard alerts (photo + text), and **chat with her from anywhere** — questions,
  "what do you see?", reminders, device control, music, gestures, `guard on/off`, `status`.
- Fresh context per chat (no stale‑topic bleed).

### 🔌 Real‑world & 🖥️ ops
- **Microcontroller/hand bridge** over USB‑serial (ESP32‑S3). Navigation hooks ready for
  future wheels/sensors.
- **Web dashboard** at `http://<pi-ip>:5000` (camera + type/read the conversation).
- **Autostart** as a systemd service (`airobot`) on boot.

---

## Everyday commands
- "Stella… what's the weather in Tel Aviv?" · "what time is it in Tokyo?"
- "What do you see?" / "What am I holding?"  (she points 👉 then looks)
- "Play *Bohemian Rhapsody*" → then "louder" / "pause" / "stop the music"
- "Remind me in 15 minutes to take out the laundry."
- "Copy my hand" → mirror → "stop copying"
- "Give me a thumbs up" · "count to three" · (insult her → 🖕 → "sorry" → forgiven)
- "Guard on" (goes quiet, armed) / "I'm home" / "Guard off"
- Telegram: "status", "send me a picture", "what do you see?"

---

## The robotic hand
**Channels (PCA9685):** `0=pinky 1=ring 2=middle 3=index 4=thumb`.
Calibrated pulses — open `{1500,1500,1500,1500,2000}` · closed `{2600,2600,2600,2600,500}`.

**Firmware** (`firmware/hand/hand.ino`, ESP32‑S3, I²C on GPIO 8/9). Serial @115200:
```
fist | open | home | hello | middle | middlehold | point | peace | thumbs | count N
f <ch> open|close      one finger        m <ch> <us>   smooth move
s <ch> <us>            instant            set <5bits>   all fingers at once (mirror)
lower | rest           back to fist       pos           report tracked positions
speed slow|med|fast    off <ch> | off all
```
Stella holds one persistent serial link on **`/dev/ttyACM0`** (`hand.enabled: true`).
**Reflashing:** `sudo systemctl stop airobot`, upload, `sudo systemctl start airobot`.
Build target: `esp32:esp32:esp32s3`.

---

## Home guard
1. Arm: *"Stella, guard on"* (she goes quiet), Telegram `guard on`, or dashboard.
2. On movement/a body while armed → 📸 + *"Do you recognise this person? YES/NO"* to Telegram.
3. **NO** → *"Sound the alarm? YES"* → **screamed alarm** on the speaker.
4. Disarm: your face (auto), *"I'm home"* (recent‑face/pass‑phrase gated), Telegram, or dashboard.

---

## Configuration (`config/config.yaml`)
- `behavior.robot_name`, `behavior.language` (`en`/`he`)
- `ai.mode`, `ai.online_provider`, `ai.groq_model`, `ai.groq_fast_model`, `ai.fallback_order`
- `security.guard_*` — motion detection, sensitivity, alert cooldown, disarm rules
- `hand.enabled`, `hand.serial_port`, `hand.wave_on_greeting`, `hand.middle_finger_on_insult`
- `music.default_volume`, `music.duck_volume`, `music.volume_step`
- `hardware.audio_output_*` — where she speaks (HDMI auto‑detected)
- `model.face_recognition_threshold`, `model.piper_*`, `model.wake_word_*`
- `system.enable_hailo` (leave `false` so the NPU is free for the offline LLM)

## Secrets (`.env`)
One documented file with every key: `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `MOONDREAM_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.
Edit a value → `sudo systemctl restart airobot`.

---

## Hardware setup
- **Pi 5 + Hailo‑10H**, USB camera (+ mic), USB mic, HDMI monitor for audio.
- **Hand:** ESP32‑S3 → PCA9685 `3V3→VCC, GND→GND, GPIO8→SDA, GPIO9→SCL`. Servos on
  channels 0–4. **External 5–6 V** into the PCA9685 **V+** screw terminal (never from the
  ESP32/Pi), with a **common ground**. Firm, soldered power wiring (loose jumpers cause shorts).
- **Deps:** `arduino-cli` + `esp32` core (flashing), `yt-dlp`+`ffmpeg` (music),
  `mediapipe 0.10.18`+`opencv 4.11`+`numpy<2` (hand mirror), `face_recognition`, `vosk`, `piper`.

## Run & ops
```bash
sudo systemctl restart airobot     # start/restart (autostarts on boot)
journalctl -u airobot -f           # watch logs   (app log: data/logs/gonzo.log)
# manual: source venv/bin/activate && python main.py
```

---

## Known quirks & notes
- **Offline Hailo LLM is unstable** — the NPU reports `OUT_OF_PHYSICAL_DEVICES` and stays
  jammed even after a reboot (a Hailo gen‑AI‑stack quirk). She relies on Groq; a small
  CPU model (llama.cpp) is the reliable offline alternative if needed.
- **HDMI audio card numbers reshuffle** across reboots — the TTS **auto‑detects** the live
  HDMI port, but it can throw a transient "device busy". A USB audio dongle would be rock‑solid.
- **USB device order reshuffles** on reboot — if the wake mic fails to open at boot, a
  restart re‑opens it; the ESP32 hand is expected on `/dev/ttyACM0`.
- **Groq free tier** rate‑limits (per‑minute + daily). GPT‑OSS‑20B adds headroom; a paid
  Claude/Anthropic key removes the limits entirely.
- Reflashing the ESP32 requires stopping `airobot` first (it holds the serial port).
