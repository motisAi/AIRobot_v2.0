# Hailo-10H NPU — how it's used & how to fix it

The Hailo-10H is Stella's **on-device AI accelerator**. Its value here is **free, private,
offline** inference — she keeps working with no internet.

## What runs on it
- **Offline LLM brain:** `qwen2.5-instruct:1.5b` (Hailo-compiled `.hef`), served by the
  `hailo-ollama` service on `http://localhost:8000`. It's the **last fallback** in the AI
  chain, so if Groq (cloud) is down/rate-limited, Stella still thinks — locally.
  - `llama3.2:3b` is also installed but **too slow** on this NPU (~19s+, times out) — do not
    use it as the live model. `qwen2.5-instruct:1.5b` is the responsive one (warm ≈ 2–5 s;
    first/cold load ≈ 40 s, absorbed by a background warmup at startup).
- Vision/object detection is **not** on the Hailo (would contend for the single NPU context
  with the LLM, and the pip `hailort` wheel in the venv is ABI-broken against HailoRT 5.1.1 —
  see "Hailo runtime repair" in the architecture doc §14). Object detection runs on the **CPU**
  (OpenCV-DNN, `data/models/yolov8n.onnx`, `system.enable_hailo: false`); scene understanding uses
  the cloud VLM (Moondream → NVIDIA NIM); face recognition uses CPU YuNet + dlib.

## The API gotcha (important)
`hailo-ollama` 5.1.1 serves **only** the OpenAI-compatible endpoint:
`POST /v1/chat/completions` (response = `choices[0].message.content`).
The ollama-native `/api/chat` and `/api/generate` return **empty / errors** on this build.
The app (`modules/ai/ai_engine.py`) was fixed to use `/v1/chat/completions` everywhere.
`/api/tags` still works (used to list local models).

## The kernel-upgrade gotcha (this WILL recur)
The Hailo PCIe driver (`hailo1x_pci`) is a **DKMS** module. When the Pi's kernel is upgraded
(e.g. `6.8.0-1057` → `6.8.0-1060`), DKMS may not auto-rebuild it, so on reboot there's **no
`/dev/hailo0`** and every Hailo call fails with *"Failed to create VDevice"*.

**Symptoms:** `ls /dev/hailo*` → not found; `lsmod | grep hailo` → empty; offline LLM 500s.

**Fix (needs sudo — run on the Pi):**
```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)   # headers for the NEW kernel
sudo dkms autoinstall                               # rebuild hailo1x_pci for it
sudo modprobe hailo1x_pci                            # load it now
ls -la /dev/hailo*                                   # expect /dev/hailo0
hailortcli fw-control identify                       # expect: Device Architecture: HAILO10H
```
Then restart Stella: `sudo -n systemctl restart airobot` (the NPU warmup will preload qwen).

Since 2026-09-19 `linux-headers-raspi` is installed, so DKMS rebuilds the driver on kernel upgrades
automatically (bug_011). **To prevent silent breakage** you can also hold the kernel:
`sudo apt-mark hold linux-image-raspi linux-headers-raspi` (optional).

## Quick health checks
```bash
ls /dev/hailo*                                        # device present?
lsmod | grep hailo                                    # driver loaded?
curl -s localhost:8000/api/tags                       # models available?
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-instruct:1.5b","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

## Base packages (OS prerequisites)

Moved here from the deleted `docs/rpi5_setup.md` (2026-09-27). Ubuntu Server 24.04 LTS (64-bit) on the Pi 5;
this is the apt baseline the Python venv builds on:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-dev python3-venv python3-pip \
    portaudio19-dev libportaudio2 \
    ffmpeg libssl-dev \
    libopencv-dev \
    git
```

Then the project venv (the service uses `venv/`, not `.venv/`):
```bash
cd ~/AIRobot_v2.0
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

Hailo specifics: install HailoRT and the firmware from the **Hailo Developer Zone `.deb` packages for aarch64**
(`hailort_*.deb`, `hailo-firmware_*.deb`; `hailortcli fw-control identify` must report `HAILO10H`). Do **not**
`apt install hailo-all` (Raspberry-Pi-OS shortcut, wrong for Ubuntu Server) and do **not** `pip install hailort`
from PyPI (the 4.23.0 wheel is what broke the venv bindings) — the only usable Python binding is the matched
pyhailort wheel from the h10-hailort bundle. `hailo-ollama` is a separate system service (`systemctl status hailo-ollama`).
Other system tools the app expects: `arduino-cli` + the `esp32` core (hand firmware), `yt-dlp` + `ffmpeg` (music,
Telegram voice), `alsa-utils` (`aplay`, `amixer`), `espeak-ng` (TTS fallback).
