# AIRobot v2.0

An autonomous AI home-robot stack for **Raspberry Pi 5 + Hailo-8/8L AI
accelerator**.  The robot sees, listens, speaks, thinks (online & offline),
and remembers — all while avoiding camera/microphone conflicts through
shared hardware managers.

## Key Capabilities
- **Hailo-accelerated vision** — YOLO object detection on the Hailo NPU with
  automatic fallback to OpenCV DNN (ONNX) or MobileNet SSD (Caffe) on CPU.
- **Shared camera & audio** — `CameraManager` distributes frames to all vision
  modules; `AudioManager` provides exclusive-lease mic/speaker access so
  wake-word, dialogue, and playback never fight.
- **AI engine (online + offline)** — OpenAI / Anthropic APIs when connected,
  local GGUF model via `llama-cpp-python` when offline, rule-based fallback
  when neither is available.
- **Persistent learning** — SQLite database stores memories, recognised faces,
  detected objects, conversations, and user preferences across reboots.
- **Wake-word → dialogue pipeline** — Picovoice Porcupine listens for "Gonzo",
  Whisper transcribes speech, AI engine generates response, Coqui/pyttsx3
  speaks it back.
- **Face recognition** — DeepFace + OpenCV for real-time identification with a
  persistent face database.
- **Modular brain** — `transitions`-based state machine with event bus,
  short/long-term memory, patrol mode, and pluggable behaviours.
- **ESP32 over UART** — motor/servo control and sensor fusion, toggled via config.
- **SIM7600X 4G** — LTE connectivity for remote access and cloud API calls.

## Repository Structure
```
config/              Global settings + platform overrides (RPi5, Jetson)
core/                Robot brain (state machine, decisions, memory)
modules/
  ai/               AI engine (online/offline LLM) + SQLite learning DB
  audio/             Wake word, speech recognition, text to speech
  hardware/          Camera manager, audio manager, ESP32, SIM7600X controllers
  vision/            Face recognition, Hailo/OpenCV object detection
docs/                Setup guides, architecture notes, API registration list
main.py              Entry point that wires every module together
```

## Hardware
| Component | Purpose |
|-----------|---------|
| **Raspberry Pi 5** (4 GB / 8 GB) | Main compute board |
| **Hailo-8 / 8L** (PCIe M.2) | AI accelerator for YOLO inference |
| **USB camera** | Vision pipeline (shared via CameraManager) |
| **USB microphone(s)** | Wake word + dialogue (managed via AudioManager) |
| **SIM7600X 4G HAT** | LTE data connectivity |
| **ESP32** (UART) | Motor / servo / sensor bridge |

## Software Requirements
- **OS**: Ubuntu Server 24.04 LTS (aarch64) or Raspberry Pi OS (64-bit).
- **Python**: 3.10+.
- **System packages**: `portaudio19-dev python3-dev ffmpeg libssl-dev`.
- **Hailo SDK**: Install from Hailo developer portal (deb packages, NOT pip).
- **Python packages**: `pip install -r requirements.txt`.

## Quick Start
1. Create and activate a virtual environment.
2. Install system libs and Python deps: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and configure:
   - `ROBOT_NAME`, `MASTER_USER_ID`
   - `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` (optional — offline fallback works)
   - `OFFLINE_MODEL_PATH` (path to a `.gguf` model for offline LLM)
   - `PICOVOICE_ACCESS_KEY` (for wake-word detection)
4. Run `python main.py`.

## Configuration Overview
- **Platform detection** in `config/platforms/` auto-detects RPi5 and applies
  Hailo-friendly defaults (model paths, thread counts, camera settings).
- **AI mode** controlled by `AI_MODE` env var: `auto` (default), `online`, or
  `offline`.
- **Shared camera** — all vision modules subscribe to `CameraManager`; no
  module opens its own `cv2.VideoCapture`.
- **Audio leases** — `AudioManager` grants exclusive per-role mic access.
- **ESP32 flag** `hardware.is_esp_connected` gates the serial controller.

## Toggling Features (config/config.json)
Edit `config/config.json` to enable/disable features **without touching code**.
The file is auto-loaded on startup. Key toggles:

| Setting | Section | What it does |
|---------|---------|-------------|
| `is_esp_connected` | hardware | Enable ESP32 motor/servo controller |
| `enable_sim7600x` | system | Enable SIM7600X 4G modem |
| `enable_hailo` | system | Use Hailo accelerator for vision |
| `patrol_mode_enabled` | system | Allow autonomous patrol behaviour |
| `auto_learning_enabled` | system | Auto-learn objects and faces |
| `learn_new_faces` | behavior | Auto-enroll unknown faces |
| `remember_conversations` | behavior | Save chat history per user |
| `remote_access_enabled` | security | Allow remote control (future app) |
| `camera_index` | hardware | Which `/dev/video*` to use |
| `sim7600x_apn` | hardware | Your carrier's APN |
| `log_level` | system | DEBUG / INFO / WARNING / ERROR |

Environment variables (`.env`) override `config.json` for secrets (API keys).

## Self-Learning Features
The robot learns automatically through normal use:
- **Conversations** are stored per-user in SQLite and recalled in future chats.
- **User preferences** ("I like coffee", "call me Dave") are auto-extracted
  and remembered.
- **Faces** are enrolled and recognized across reboots.
- **Objects** detected by the camera are logged with timestamps and counts.
- **Memories** are promoted from short-term to long-term when importance is high.

## Documentation
- `docs/rpi5_setup.md` — RPi5 + Hailo installation guide.
- `docs/architecture.md` — Module diagrams, event flow, data paths.
- `docs/service_accounts.txt` — Required API accounts and registration steps.
