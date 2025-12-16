# Jetson Nano Setup Guide

These steps assume Ubuntu 18.04.6 LTS on a Jetson Nano with Linux for Tegra
kernel 4.9.253.

## 1. Update Firmware and JetPack
1. Flash the latest JetPack image (5.x) using NVIDIA SDK Manager.
2. Boot the Nano and finish the Ubuntu onboarding wizard.
3. Update the BSP:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

## 2. Install System Packages
```bash
sudo apt install -y \
    python3-venv python3-dev build-essential \
    libffi-dev libssl-dev cmake git pkg-config \
    libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    ffmpeg libopencv-dev
```

## 3. Create the Python Environment
```bash
python3 -m venv ~/envs/airobot
source ~/envs/airobot/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

> **Note:** Torch/Whisper wheels are not listed in `requirements.txt` because
> Jetson requires CUDA-specific builds. Install them manually when you are ready:
> ```bash
> pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
>     torch torchvision torchaudio
> pip install git+https://github.com/openai/whisper.git
> ```

## 4. Audio Device Mapping
1. List microphones: `aplay -l` and `arecord -l`.
2. Note the card/device numbers for:
   - USB mic dedicated to wake-word detection.
   - USB mic used for dialogue capture (if different).
3. Update `HardwareConfig` (`config/settings.py`) with either the ALSA name or
   an explicit PortAudio device index so the background listener never steals the
   wrong audio stream.

## 5. Optional Hardware
### ESP32 over UART
1. Wire TX/RX to `/dev/ttyTHS1` (Jetson GPIO 8/10) and ground reference.
2. Set `hardware.is_esp_connected = True` and update
   `hardware.uart_device_map['esp32'] = '/dev/ttyTHS1'` if needed.
3. Install pySerial only when you enable the board: `pip install pyserial`.

### SIM7600X LTE
1. Connect the module to `/dev/ttyTHS0` (or use a USB adapter).
2. Populate APN and SIM PIN in `config/settings.py`.
3. Ensure `system.enable_sim7600x = True` if you want the server bridge to dial
   out automatically.

## 6. Wake-Word Custom Model (Picovoice)
1. Create a free Picovoice console account.
2. Generate a custom Porcupine keyword for "Gonzo" (language: English).
3. Download the `.ppn` file into `data/models/wake_word.ppn`.
4. Set `PICOVOICE_ACCESS_KEY` inside `.env`.

## 7. Diagnostics
```bash
python main.py --test   # runs sanity checks
python main.py          # launches the full stack
```

Monitor logs under `data/logs/` if any module reports initialization errors.
