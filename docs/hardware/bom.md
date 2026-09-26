# Hardware — Bill of Materials

_Running list of components in the Stella build. Add rows as parts are added. Microphone roles are canonical in
[../architecture/stella-architecture.md §4](../architecture/stella-architecture.md#4-audio--the-canonical-two-mic-design);
this table only mirrors them._

| Component | Model / spec | Role | Notes |
|-----------|--------------|------|-------|
| SBC | Raspberry Pi 5 | Main compute / orchestrator | Ubuntu Server 24.04.4 aarch64, headless |
| AI accelerator | Hailo-10H NPU (PCIe/M.2) | Offline LLM only (`qwen2.5-instruct:1.5b` via `hailo-ollama` :8000) | Object detection runs on the CPU (OpenCV-DNN `yolov8n.onnx`) because the pip `hailort` wheel is ABI-broken; see [hailo.md](hailo.md) |
| Hand controller | ESP32-S3-WROOM-1 | Drives the servo hand over USB serial `/dev/ttyACM0` | Replaced an earlier classic ESP32 that shorted |
| Servo driver | PCA9685 (16-ch PWM) | I2C → PWM for the finger servos | Addr 0x40, ~50 Hz |
| Servos | 5× hobby servo | One per finger | Channels 0–4 = pinky→thumb |
| Servo PSU | 5–6 V external supply | Powers servos (V+) | Common ground with ESP32; NOT off the Pi |
| Camera | USB UVC webcam "Signo Camera WB-400 Auto Focus Camera" | Vision (face, objects, guard, mirror, VLM) **+ the WAKE mic** | Shared via `CameraManager`; found by name, CSI nodes skipped |
| Mic A — **WAKE** | Camera built-in mic — ALSA card id `Camera` (card 3 today), by-id `usb-Signo_Camera_WB-400_Auto_Focus_Camera*`, PortAudio `Auto Focus Camera: USB Audio (hw:3,0)` | Wake word (Vosk "hey stella") | Native 48 kHz (no 16 k). Config `hardware.wake_word_microphone_name: "Auto Focus Camera"` |
| Mic B — **COMMAND** | "USB PnP Sound Device" (C-Media dongle) — ALSA card id `Device` (card 0 today), by-id `usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*`, PortAudio `USB PnP Sound Device: Audio (hw:0,0)` | Conversation / STT capture | Native 44.1 kHz. Config `hardware.speech_microphone_name: "USB PnP Sound Device"`. Has AGC and weak far-field pickup — never the wake mic (bug_053) |
| Audio out | HDMI audio (`vc4hdmi0` / `vc4hdmi1`, auto-detected) | Speech output (piper → aplay) | Card numbers move across reboots; BT speaker "AR-SPJ" optional |
| WiFi (home) | Pi 5 onboard radio | Home-WiFi client (`wlan0`, sometimes `wlan1` — bug_052) | — |
| WiFi (AP) | TP-Link Archer T2U Plus (RTL8821AU) | "RobotNet" AP 10.0.0.1/24 for peripherals | `8821au` DKMS driver — rebuild after kernel upgrades (bug_044) |
| Face display | PC / tablet browser on `http://<pi>:8080` (face_bridge) | Animated face, shown on voice command | Live on the Surface today; Teclast P30T tablet (Allwinner A523, 4 GB real RAM, 10.1" 1280×800, Android 14) is the planned dedicated screen |
| PSU | USB-C, official 27 W recommended | — | Power-offs seen with a weaker supply when Ethernet is plugged in; no fan fitted (80 °C soft limit under load) |

**Not fitted:** SIM7600X 4G modem, IMU, wheel encoders, wheels, lidar, ultrasonic, INMP441 I²S mics (planned as the
durable replacement for USB audio).

## Notes / cautions

- **Servo power** must come from the dedicated 5–6 V supply, never the Pi's 5V rail.
- **I2C** on the ESP32-S3 is GPIO 8 (SDA) / GPIO 9 (SCL) — the classic-ESP32 GPIO22 pin
  does **not** exist on the S3.
- **Two mics, two roles, two devices.** Wake and command must never resolve to the same ALSA device (single-mic
  handoff failed — bug_055). Devices are resolved by stable identity (by-id / card id), never by index.
- Tablet vendor RAM spec ("10/15 GB") is virtual — treat as **4 GB real**; keep the face app
  lightweight (2D Canvas, vanilla JS).
