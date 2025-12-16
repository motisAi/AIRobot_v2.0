# AIRobot v2.0

An autonomous home robot stack that combines face recognition, wake-word driven
speech control, and modular hardware integration.  The code base now targets a
Jetson Nano running Ubuntu 18.04.6 LTS (aarch64, Tegra 4.9.253) but it gracefully
falls back to CPU-only execution so you can iterate on any development machine.

## Key Capabilities
- Local-first identity management (faces + voices) so the robot always knows its
  owner without uploading biometric data.
- Always-on wake-word service for the "Gonzo" keyword with dedicated USB-mic
  selection and automatic pause/resume during dialogues.
- Speech pipeline that prefers Whisper for offline transcription and falls back
  to standard `speech_recognition` if Whisper is not installed yet.
- Modular robot brain with event bus, short/long term memory, patrol mode, and
  pluggable behaviors.
- Optional ESP32 over UART for motor control, servo control, and sensor fusion –
  toggled via configuration so you can wire it up later without touching code.

## Repository Structure
```
config/              Global settings + platform overrides
core/                Robot brain (state machine, decisions, memory)
modules/
  audio/             Wake word, speech recognition, text to speech
  hardware/          ESP32 + SIM7600X controllers
  vision/            Face recognition (DeepFace + OpenCV)
docs/                Jetson setup, architecture notes, API registration list
main.py              Entry point that wires every module together
```

## Hardware Checklist
1. **Jetson Nano** with JetPack (CUDA/cuDNN/TensorRT installed) – currently
   tested on Ubuntu 18.04.6 LTS.
2. **USB camera with microphone** (camera feed only) + a **dedicated USB
   microphone** that handles the wake-word thread.
3. **Optional ESP32** connected through UART. Toggle `is_esp_connected` inside
   `config/settings.py` (or via a JSON override) when you are ready to bring the
   board online.
4. **Optional SIM7600X** module for LTE connectivity. The controller auto-detects
   Jetson UART mappings and stays idle until the module is plugged in.

## Software Requirements
- Python 3.8+ (Jetson Nano images ship with 3.8; see `docs/jetson_setup.md` for
  the recommended tooling stack).
- System packages: `portaudio19-dev`, `python3-dev`, `ffmpeg`, `libssl-dev`.
- Python packages listed in `requirements.txt`. Torch/Whisper wheels are **not**
  included because Jetson users must install the matching CUDA wheels manually –
  the setup guide explains the exact commands.

## Quick Start
1. Create and activate a virtual environment.
2. Install system libs and Python dependencies (`pip install -r requirements.txt`).
3. Copy `.env.example` to `.env` (set `ROBOT_NAME`, `MASTER_USER_ID`, and leave
   API keys blank if you are still evaluating the fallback modes).
4. Run `python main.py --test` to execute the built-in diagnostics.
5. Launch the robot normally with `python main.py` once all modules pass.

## Configuration Overview
- **Platform detection** lives in `config/platforms/`. The Jetson helper applies
  CUDA-friendly defaults (camera indices, FPS caps, TensorRT toggles).
- **Wake-word + mic routing** is configured inside `HardwareConfig` – specify the
  ALSA/PortAudio device name so the always-on thread never conflicts with your
  dialogue microphone.
- **ESP32 flag** `hardware.is_esp_connected` gates the controller so you can ship
  the code without serial drivers on your dev box.
- **Security** settings in `SecurityConfig` keep the master user ID local-only.

## Documentation Set
- `docs/jetson_setup.md` – Driver, CUDA, and package installation cheat sheet.
- `docs/architecture.md` – Module-level diagrams, event flow, and data paths.
- `docs/service_accounts.txt` – Where to register for Picovoice, OpenAI, etc.

## Next Steps
- Wire in SLAM/object tracking modules to extend the "environment learning"
  layer once additional sensors (depth, proximity) are available.
- Integrate the upcoming on-board CSI camera and proximity sensors by enabling
  the ESP32 flag and populating the sensor map in `esp32_controller.py`.
- Connect to your preferred LLM provider once API keys are ready. The
  `docs/service_accounts.txt` file lists every required account.

Please read the docs and configuration comments before altering the code – every
module contains detailed docstrings so you can reason about the execution flow
quickly.
