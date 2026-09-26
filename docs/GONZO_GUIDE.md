# Gonzo — Setup & Usage Guide

The single source of truth for how this robot is wired **right now** on your
Raspberry Pi 5 + **Hailo-10H**. Everything below reflects the current code.

## The one file you edit: `config/config.yaml`
All behaviour is controlled from `config/config.yaml`. Change a value, save,
restart (`python main.py`). Secrets (API keys) live in `.env`, never in YAML.

Sections: `behavior`, `ai`, `web_search`, `microcontroller`, `navigation`,
`hardware`, `model`, `system`, `security`.

## How Gonzo thinks (free, on-device)
- Default `ai.mode: local` → the **Hailo-10H NPU** runs a local LLM via the
  `hailo-ollama` service (port 8000). No internet, no tokens, no cost.
- Installed NPU model: `qwen2.5-instruct:1.5b`. Check what's installed:
  `curl localhost:8000/api/tags`. To use a bigger/smarter model, pull it first:
  `curl -X POST localhost:8000/hailo/v1/pull -d '{"model":"llama3.2:3b"}'`
  then set `ai.hailo_ollama_model` to it.
- `ai.mode: hybrid` keeps local as default but escalates hard questions to the
  cloud (needs `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` in `.env` and
  `allow_cloud_escalation: true`). `ai.mode: online` uses the cloud first.

## Web search (like Claude/Copilot)
`web_search.enabled: true` (default). When you ask a factual/current question,
Gonzo searches **DuckDuckGo** (free, no key) and answers using the results.

## Talking to Gonzo
1. Say **"Gonzo"** (or "Hey Gonzo"). Wake detection uses **Vosk** — free,
   offline, and detects the actual word (`model.wake_word_engine: auto`).
2. Gonzo greets you: *"How can I help you, <name>?"* if it recognises your face,
   otherwise it asks your name and **enrolls a new face**.
3. You speak a question/command; Gonzo replies through the **HDMI monitor**
   speakers (`hardware.audio_output_device: hdmi`, card 3 = the connected port).
   To use a USB speaker later: set `audio_output_device: usb`,
   `audio_output_card: null`.

Two microphones, one job each:
- Wake word → camera mic (`Auto Focus Camera`, card 1)
- Commands/conversation → `USB PnP Sound Device` (card 0)

## Faces, memory & learning
- Master (**Moti**) is already enrolled in `data/faces/face_db.pkl`.
- New people are enrolled on the fly ("What's your name?").
- Conversations + preferences ("call me…", "I like…") are stored per-user in
  `data/robot_memory.db` (SQLite) and recalled in later chats.

## Controlling real devices ("turn on the light")
`microcontroller.connected: false` by default → device commands are recognised
and **logged only**, so you can build everything before wiring hardware.
When you connect an ESP32 / Pi Zero:
1. Set `microcontroller.connected: true`.
2. Choose `transport: serial` (USB) or `network` (Wi-Fi) and set the port/host.
3. Map spoken names to firmware ids under `microcontroller.device_map`
   (e.g. `light: relay1`).
Firmware just needs to accept one JSON line per command:
`{"action":"on","target":"relay1"}` and reply `{"ok":true}`.
Only the **master** can issue device commands.

## Future: wheels, sensors, mapping the house
`navigation.*` is off by default. When you add wheels + an ultrasonic/LiDAR
sensor to the microcontroller, enable `navigation.enabled`, `has_wheels`,
`has_ultrasonic`, `mapping_enabled`, etc. The `Navigator` already drives and
reads sensors through the same microcontroller bridge — no new plumbing needed.

## Run it
```bash
cd ~/AIRobot_v2.0
venv/bin/python check_deps.py      # health check
venv/bin/python main.py --test     # init + self-test, then exit
venv/bin/python main.py            # normal operation
```

## What needs a key vs. free
| Feature            | Cost | Needs a key? |
|--------------------|------|--------------|
| On-device LLM (NPU)| Free | No |
| Wake word (Vosk)   | Free | No |
| Speech-to-text (Google) | Free | No (needs internet) |
| Text-to-speech (pyttsx3)| Free | No |
| Web search (DuckDuckGo) | Free | No |
| Face recognition   | Free | No |
| Cloud LLM (optional)| Paid | OpenAI/Anthropic key in `.env` |
| Picovoice wake word| Paid | Not used (Vosk replaces it) |
