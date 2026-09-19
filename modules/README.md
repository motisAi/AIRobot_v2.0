# modules — capabilities, grouped by what they do

Each sub‑package is one capability. Modules never open hardware themselves; they get the
`parts_used` objects (camera, audio, hand…) from `main.py`, and they talk to each other
through the event bus in `core/robot_brain.py`.

| Package | Files | Does |
|---|---|---|
| `ai/` | `ai_engine.py` (provider chain groq → groq_fast → gemini → hailo, tools, `online` flag), `learning_db.py` (SQLite memory: people, objects, conversations), `reminders.py`, `web_search.py` (DuckDuckGo) | Thinking, memory, tools |
| `audio/` | `wake_word.py` (Vosk "Stella", energy fallback, backoff when no mic), `speech_recognition.py` (Groq Whisper → Google → Vosk, hallucination filter, `mic_available()`), `text_to_speech.py` (Piper via aplay, HDMI auto‑detect + fallback), `spell_parser.py` | Hearing and speaking |
| `vision/` | `face_recognition.py` (dlib, enrolment, master auth), `object_detection.py` (YOLO via `parts_used/hailo_10h.py`, 1 Hz, scene‑change events), `motion_guard.py` (guard mode), `hand_mirror.py` (MediaPipe → hand), `vlm.py` (Moondream → NVIDIA NIM) | Seeing |
| `conversation/` | `manager.py` — the spoken session: wake → listen → think → speak, plus keyword handlers (guard, AC, devices, music, mirror, face show/hide) all master‑gated | The dialogue loop |
| `comms/` | `telegram_bridge.py` (two‑way chat, alerts, YES/NO flows), `notify.py` (spoken/phone notifications) | Talking to Moti's phone |
| `smart_home/` | `sensibo.py` (AC, cloud API, cached state), `tuya_devices.py` (local plugs), `mqtt_devices.py` (mosquitto on the Pi, RobotNet relays) | Controlling the house |
| `media/` | `music.py` (yt‑dlp + ffmpeg, ducking, volume) | Music |
| `navigation/` | `navigator.py` — hooks for wheels/lidar/IMU (all off in config) | Future driving |
| `connectivity/` | `server_connection.py` — remote server link over WiFi/4G | Remote link |
| `web/` | `dashboard.py` — `http://<pi>:5000` camera + conversation + guard control | Local dashboard |

## Conventions
- Config comes from `config.settings` dataclasses (loaded from `config/config.yaml`); no
  module reads YAML itself.
- Time inside loops and watchdogs uses `time.monotonic()` (the wall clock jumps hours after
  an offline boot syncs NTP — see `bug_report/bug_040`).
- Anything that can block (mic read, HTTP, subprocess) has a timeout; providers are skipped
  when `engine.online` is false.
- A privileged action (devices, guard off, gestures on people, RC toy) goes through the
  master check in `conversation/manager.py` or the equivalent tool gate in `ai/ai_engine.py`.
