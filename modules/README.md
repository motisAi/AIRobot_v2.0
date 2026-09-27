# modules — capabilities, grouped by what they do

Each sub‑package is one capability. Modules never open hardware themselves; they get the
`parts_used` objects (camera, audio, hand…) from `main.py`, and they talk to each other
through the event bus in `core/robot_brain.py`.

| Package | Files | Does |
|---|---|---|
| `ai/` | `ai_engine.py` (provider chain groq → groq_fast → gemini → hailo, tools, `online` flag), `learning_db.py` (SQLite memory `data/robot_memory.db`: people, objects, conversations, per‑person facts), `reminders.py`, `web_search.py` (DuckDuckGo) | Thinking, memory, tools |
| `audio/` | `wake_word.py` (Vosk "hey stella" on the **camera mic**; the wake‑word thread owns that stream; energy fallback, backoff when no mic, dead‑stream reopen), `speech_recognition.py` (`capture_utterance()` on the **USB PnP mic**, single‑thread + hard deadline; STT chain Groq Whisper → Google → Vosk, hallucination filter, `mic_available()`), `text_to_speech.py` (Piper via aplay, HDMI auto‑detect + espeak fallback), `spell_parser.py`. Canonical design: [docs/architecture/stella-architecture.md §4](../docs/architecture/stella-architecture.md#4-audio--the-canonical-two-mic-design) | Hearing and speaking |
| `vision/` | `face_recognition.py` (YuNet detector + dlib embeddings, euclidean 0.60, enrolment, master auth), `object_detection.py` (YOLOv8n via `parts_used/hailo_10h.py` on the CPU, every 4 s, scene‑change events), `motion_guard.py` (guard mode), `hand_mirror.py` (MediaPipe → hand), `vlm.py` (Moondream → NVIDIA NIM) | Seeing |
| `conversation/` | `manager.py` — the spoken session on its own thread: wake → listen → think → speak, plus keyword handlers (guard, AC, devices, music, mirror, face show/hide) all master‑gated; resumes the wake listener once at session end | The dialogue loop |
| `comms/` | `telegram_bridge.py` (two‑way chat, alerts, voice notes, YES/NO flows), `notify.py` (spoken/phone notifications) | Talking to Moti's phone |
| `smart_home/` | `sensibo.py` (AC, cloud API, cached state), `tuya_devices.py` (local plugs), `mqtt_devices.py` (mosquitto on the Pi, RobotNet relays) | Controlling the house |
| `media/` | `music.py` (yt‑dlp + ffmpeg, ducking, volume) | Music |
| `navigation/` | `navigator.py` — hooks for wheels/lidar/IMU (all off in config) | Future driving |
| `web/` | `dashboard.py` — `http://<pi>:5000` camera + conversation + guard control | Local dashboard |

Removed 2026-09-27: `connectivity/` (`server_connection.py`, a dead remote‑server link over the never‑fitted 4G modem).

## Conventions
- Config comes from `config.settings` dataclasses (loaded from `config/config.yaml`); no
  module reads YAML itself (`main.py`'s raw read of the `mqtt` section is listed debt — `MqttConfig` is coming).
- Time inside loops and watchdogs uses `time.monotonic()` (the wall clock jumps hours after
  an offline boot syncs NTP — see `bug_report/bug_040`).
- Anything that can block (mic read, HTTP, subprocess) has a timeout; providers are skipped
  when `engine.online` is false. A blocked mic read is bounded by the hard deadline and the two‑stage watchdog,
  **never by a helper thread** (bug_028, bug_054).
- A PortAudio stream is opened, read and closed by one thread; wake and command are two different devices.
- A privileged action (devices, guard off, gestures on people, RC toy) goes through the
  master check in `conversation/manager.py` or the equivalent tool gate in `ai/ai_engine.py`.
