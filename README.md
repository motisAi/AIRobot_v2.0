# Stella — Smart Self‑Learning AI Robot 🤖

Stella runs on a **Raspberry Pi 5 + Hailo‑10H NPU** (Ubuntu Server 24.04, headless), with a
USB camera (+ mic), a USB mic, a **5‑finger robotic hand** (ESP32‑S3 + PCA9685), an on‑demand
animated **face** on a PC/phone screen, and a two‑way **Telegram** link. She recognises faces,
holds a spoken conversation, thinks with cloud LLMs and falls back to an on‑device LLM when
offline, sees through a vision model, detects objects, plays music, guards the home, controls
smart‑home devices, gestures and mirrors your hand, and has a cheeky personality with a memory.

Everything is driven by one human‑editable file — **`config/config.yaml`** — and all secrets
live in **`.env`** (never in git).

> **This file is the map.** Each folder below has its own focused README where the detail lives.

---

## Where everything lives

| Path | What is in it | Read more |
|---|---|---|
| `main.py` | The entry point. Wires every part and module together, runs the systemd service. | — |
| `config/` | `config.yaml` (behaviour, the ONE file you edit), `settings.py` (typed config), `platforms/` (Pi 5 overrides). | comments inside `config.yaml` |
| `core/` | `robot_brain.py` (event bus + robot state + memory), `watchdog.py` (restarts her when stuck). | — |
| **`parts_used/`** | **One file per physical part**: USB camera, audio devices, Hailo‑10H, ESP32 hand, microcontroller bridge, 4G modem, WiFi, RC toy. | [parts_used/README.md](parts_used/README.md) |
| **`modules/`** | Capabilities grouped by what they do: `ai/`, `audio/`, `vision/`, `conversation/`, `comms/`, `smart_home/`, `media/`, `navigation/`, `connectivity/`, `web/`. | [modules/README.md](modules/README.md) |
| `face_bridge/` | The screen face: WebSocket server + web app (`http://<pi>:8080`). | [docs/hardware/face-subsystem.md](docs/hardware/face-subsystem.md) |
| `firmware/` | ESP32 sketches (hand, tests, I²C scan). | [docs/firmware/hand-esp32.md](docs/firmware/hand-esp32.md) |
| **`evolution/`** | Stella's self‑improvement system: **Guardian** (health + rollback), manifest, Scout, reports. | [evolution/README.md](evolution/README.md) |
| **`bug_report/`** | One file per bug ever fixed: symptom → root cause → fix → how to verify. **Check here before debugging anything.** | [bug_report/README.md](bug_report/README.md) |
| `tests/` | Smoke tests run by Guardian (`tests/test_imports.py`). | [tests/README.md](tests/README.md) |
| `tools/` | Operator scripts: enrol the master face, manage faces, check deps, hand control. | [tools/README.md](tools/README.md) |
| `deploy/` | systemd unit, **`deploy.sh`** (restart → Guardian → auto‑rollback), sudoers/WiFi helpers. | [evolution/README.md](evolution/README.md#deploying-safely) |
| `docs/` | Architecture, dated decision log, hardware/wiring, briefs. | [docs/README.md](docs/README.md) |
| `data/` | Runtime state, gitignored: models, faces DB, logs, memory DB, TTS cache. | — |

**Architecture:** [docs/architecture/stella-architecture-2026-09-19.md](docs/architecture/stella-architecture-2026-09-19.md)
(layout, safety net, self‑evolution) builds on
[stella-architecture-2026-09-13.md](docs/architecture/stella-architecture-2026-09-13.md)
(capability stack online → offline, Hailo/NVIDIA plan, phased roadmap).

---

## Capabilities

### 🎙️ Voice & conversation
- **Wake word** ("Stella" / "hey Stella"), offline (Vosk; openWakeWord "Hey Stella" planned).
- **Two microphones, two roles** — camera mic for the wake word, USB mic for the command
  (voice‑activity detection ends the sentence). No mic → she stays quiet instead of talking to herself.
- **Multi‑turn conversation** — greets, chats, asks "anything else?" after silence, says goodbye.
- **Speech‑to‑text chain:** Groq Whisper → Google → offline Vosk, with a hallucination filter.
- **Natural neural voice** — Piper (offline). HDMI output auto‑detected, non‑HDMI fallback.
- **Language switch** — `behavior.language: en | he`.

### 🧠 Brain (LLM) with graceful fallback
Tried in order per question; on error/rate‑limit she falls to the next:
1. **Groq `openai/gpt-oss-120b`** — smartest free model.
2. **Groq `openai/gpt-oss-20b`** — separate rate limit.
3. **Gemini `gemini-3.6-flash`** — separate provider, separate quota.
4. **On‑device Hailo‑10H NPU (`qwen2.5-instruct:1.5b` via hailo‑ollama)** — free, offline.

When she is offline (network monitor), cloud providers are skipped entirely, so an offline
answer takes ~0.2 s to start instead of ~90 s. Per‑user memory lives in a SQLite DB;
anti‑fabrication rules stop her inventing facts or firing tools on filler.

### 🤖 Agent tools (she picks them herself)
| Tool | What it does |
|------|--------------|
| `get_time` / `get_weather` | Time in any timezone; live weather (wttr.in) |
| `web_search` | DuckDuckGo (free) |
| `look` | Describe the camera view / "what am I holding?" (Moondream VLM → NVIDIA NIM fallback) |
| `set_reminder` | "Remind me in 10 minutes to…" (survives restarts) |
| `control_device` | Smart‑home: Tuya plug → MQTT → wired microcontroller |
| `set_guard_mode` | Arm / disarm home guard |
| `send_telegram` / `send_photo` | Message or camera snapshot to your phone |
| `play_music` / `stop_music` | YouTube playback |
| `do_gesture` | Hand gestures: wave, thumbs_up, point, peace, fist, count, middle_finger… |

### 👁️ Vision
- **Face recognition + enrolment** (dlib, CPU) with master authentication for privileged actions.
- **Object detection** — YOLOv8n via OpenCV‑DNN on CPU (`data/models/yolov8n.onnx`), 1 inference/s,
  scene‑change events; Hailo NPU path available once HailoRT ≥ 5.2 is installed.
- **Scene understanding** — cloud VLM (Moondream, NVIDIA NIM llama‑3.2‑11b‑vision).
- **Motion guard** for the home‑guard mode; **hand mirror** (MediaPipe landmarks → servos).

### ✋ Robotic hand · 🎭 Personality · 🛡️ Home guard · 🎵 Music · 📱 Telegram
Unchanged from before and documented in detail below. Sensibo AC, Tuya plugs and MQTT
devices are controlled by voice or Telegram (master‑only).

---

## Everyday commands
- "Stella… what's the weather in Tel Aviv?" · "what time is it in Tokyo?"
- "What do you see?" / "What am I holding?"
- "Play *Bohemian Rhapsody*" → "louder" / "pause" / "stop the music"
- "Remind me in 15 minutes to take out the laundry."
- "Copy my hand" → mirror → "stop copying" · "Give me a thumbs up" · "count to three"
- "Guard on" (goes quiet, armed) / "I'm home" / "Guard off" · "Turn on the AC, make it cold"
- "Show your face" / "show your face on phone" · Telegram: "status", "send me a picture", or **send a 🎤 voice note** — she transcribes it (Groq Whisper), runs it like a typed command, and answers in text **and** in her own voice

## The robotic hand
**Channels (PCA9685):** `0=pinky 1=ring 2=middle 3=index 4=thumb`. Open `{1500,1500,1500,1500,2000}`,
closed `{2600,2600,2600,2600,500}`. Firmware `firmware/hand/hand.ino` (ESP32‑S3, I²C GPIO 8/9,
serial 115200 on **`/dev/ttyACM0`**). Stella holds one persistent link (`hand.enabled: true`);
**stop `airobot` before reflashing**. Full protocol: [docs/firmware/hand-esp32.md](docs/firmware/hand-esp32.md).

## Home guard
1. Arm: *"Stella, guard on"*, Telegram `guard on`, or dashboard.
2. Movement / a body while armed → 📸 + *"Do you recognise this person? YES/NO"* on Telegram.
3. **NO** → *"Sound the alarm? YES"* → screamed alarm on the speaker.
4. Disarm: your face (auto, after you were away >60 s), *"guard off"* (master‑gated), Telegram, dashboard.

---

## Configuration (`config/config.yaml`)
Sections: `behavior`, `ai` (provider chain), `conversation`, `music`, `web_search`, `hand`,
`microcontroller`, `rc_toy`, `navigation`, `hardware` (camera, mics, audio out, Hailo),
`model` (face threshold, Piper voice, STT, wake word), `system`, `security` (guard), `mqtt`.
Every key is commented in the file.

## Secrets (`.env`, mode 600, gitignored)
`GROQ_API_KEY`, `GEMINI_API_KEY`, `MOONDREAM_API_KEY`, `NVIDIA_API_KEY`, `TELEGRAM_TOKEN`,
`TELEGRAM_CHAT_ID`, `SENSIBO_API_KEY` (+ optional `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
Edit a value → `sudo systemctl restart airobot`.

---

## Run & ops
```bash
sudo systemctl restart airobot            # start/restart (autostarts on boot)
journalctl -u airobot -f                  # live log        (app log: data/logs/gonzo.log)
venv/bin/python evolution/guardian.py     # health check: syntax, imports, config, service, log, hardware, brain, power
deploy/deploy.sh                          # after a commit: restart → Guardian → auto-rollback if unhealthy
venv/bin/python tools/reenroll.py         # re-enrol the master face
```
**Rule for every change:** commit → `deploy/deploy.sh` → push. Rollback is always
`git reset --hard <previous commit>` + restart. Every fixed bug gets a file in `bug_report/`.

## Hardware setup
- **Pi 5 + Hailo‑10H**, USB camera (+ mic), USB mic, HDMI monitor or USB speaker for audio.
  Use the official 27 W PSU; the Pi has powered off when the Ethernet cable was plugged in on a
  weaker supply. No fan is fitted yet and she reaches the 80 °C soft limit under load.
- **Hand:** ESP32‑S3 → PCA9685 `3V3→VCC, GND→GND, GPIO8→SDA, GPIO9→SCL`, servos on channels 0–4,
  **external 5–6 V** into PCA9685 **V+** with a common ground. See [docs/hardware/wiring.md](docs/hardware/wiring.md).
- **Hailo driver:** rebuilt automatically after kernel upgrades once `linux-headers-raspi` is
  installed; if `/dev/hailo0` is missing see [bug_report/bug_011](bug_report/bug_011_hailo-driver-missing-after-kernel-upgrade.md).
- **Deps:** `arduino-cli` + `esp32` core, `yt-dlp` + `ffmpeg`, `mediapipe 0.10.18` + `opencv 4.11` +
  `numpy<2`, `face_recognition`, `vosk`, `piper`. Python deps: `requirements.txt`.

## Known quirks
- HDMI audio card numbers and USB device order reshuffle across reboots — devices are resolved
  by name; a restart re‑opens a mic that failed at boot.
- Groq free tier rate‑limits (per‑minute and daily); Gemini and the NPU catch the overflow.
- Reflashing the ESP32 requires stopping `airobot` first (it holds the serial port).
- Everything else that ever went wrong, and how it was fixed: [bug_report/](bug_report/README.md).
