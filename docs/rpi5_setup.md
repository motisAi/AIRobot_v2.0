# Raspberry Pi 5 + Hailo Setup Guide

## Hardware Assembly
1. **Raspberry Pi 5** (4 GB or 8 GB recommended).
2. **Hailo-8 or Hailo-8L** M.2 module installed on a PCIe HAT/adapter.
3. **USB camera** — any UVC-compatible webcam.
4. **USB microphone(s)** — one for wake-word, optionally a second for dialogue.
5. **SIM7600X 4G HAT** — connected via UART (`/dev/ttyUSB2` or `/dev/ttyAMA0`).
6. **ESP32** — connected via USB-UART for motor/servo/sensor bridge.
7. Adequate power supply (27 W USB-C PD recommended).

## OS Installation
1. Flash **Ubuntu Server 24.04 LTS (64-bit)** or **Raspberry Pi OS Lite
   (64-bit)** using Raspberry Pi Imager.
2. Enable SSH during the flash process.
3. Boot, update:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## System Packages
```bash
sudo apt install -y \
    python3-dev python3-venv python3-pip \
    portaudio19-dev libportaudio2 \
    ffmpeg libssl-dev \
    libopencv-dev \
    git
```

## Hailo SDK Installation
1. Register at https://hailo.ai/developer-zone/ and download the Hailo SDK
   `.deb` packages for aarch64.
2. Install:
   ```bash
   sudo dpkg -i hailort_*.deb
   sudo dpkg -i hailo-firmware_*.deb
   ```
3. Verify:
   ```bash
   hailortcli fw-control identify
   ```
   You should see the Hailo device information (firmware version, board type).
4. Install the Python bindings:
   ```bash
   pip install hailo-platform   # or install the .whl from the SDK
   ```

## Python Environment
```bash
cd ~/AIRobot_v2.0
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional heavy dependencies
```bash
# Local LLM inference (offline mode)
pip install llama-cpp-python

# Coqui TTS (neural voices, ~2 GB download)
pip install TTS
```

## AI Model Files
Place model files in `data/models/`:

| File | Purpose | Source |
|------|---------|--------|
| `yolov8s.hef` | Hailo object detection | Hailo Model Zoo |
| `yolov8n.onnx` | OpenCV DNN fallback | Ultralytics export |
| `MobileNetSSD_deploy.caffemodel` | CPU-only fallback | OpenCV samples |
| `MobileNetSSD_deploy.prototxt` | CPU-only fallback | OpenCV samples |

For offline LLM, download a GGUF model (e.g. `tinyllama-1.1b-chat.Q4_K_M.gguf`)
and set `OFFLINE_MODEL_PATH` in `.env`.

## Environment Variables (`.env`)
```bash
ROBOT_NAME=Gonzo
MASTER_USER_ID=your_face_id

# AI — leave blank to use offline/fallback mode
AI_MODE=auto
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AI_PROVIDER=openai

# Offline LLM
OFFLINE_MODEL_PATH=data/models/tinyllama-1.1b-chat.Q4_K_M.gguf

# Picovoice wake word
PICOVOICE_ACCESS_KEY=your_key_here

# Hardware hints
WAKE_MIC_NAME=USB PnP Sound Device
SPEECH_MIC_NAME=USB PnP Sound Device
```

## Running
```bash
source .venv/bin/activate
python main.py
```

## Running as a systemd Service
```bash
sudo tee /etc/systemd/system/airobot.service << 'EOF'
[Unit]
Description=AIRobot v2.0
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/AIRobot_v2.0
ExecStart=/home/pi/AIRobot_v2.0/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable airobot
sudo systemctl start airobot
```

## Troubleshooting
- **Hailo not detected**: Check PCIe connection with `lspci | grep Hailo`.
  Ensure `hailort` kernel module is loaded: `lsmod | grep hailo`.
- **Camera not found**: `ls /dev/video*` — use `v4l2-ctl --list-devices`.
- **Mic not found**: `arecord -l` to list ALSA capture devices.
- **High temperature**: The Pi 5 needs active cooling. Check with
  `cat /sys/class/thermal/thermal_zone0/temp`.
