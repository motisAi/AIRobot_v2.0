# Hardware — Bill of Materials

_Running list of components in the Stella build. Add rows as parts are added._

| Component | Model / spec | Role | Notes |
|-----------|--------------|------|-------|
| SBC | Raspberry Pi 5 | Main compute / orchestrator | Ubuntu 24.04.4 aarch64 |
| AI accelerator | Hailo-10H NPU | On-device vision / (offline LLM) | Offline LLM currently unstable; using cloud LLMs |
| Hand controller | ESP32-S3-WROOM-1 | Drives the servo hand over USB serial | Replaced an earlier classic ESP32 that shorted |
| Servo driver | PCA9685 (16-ch PWM) | I2C → PWM for the finger servos | Addr 0x40, ~50 Hz |
| Servos | 5× hobby servo | One per finger | Channels 0–4 = pinky→thumb |
| Servo PSU | 5–6 V external supply | Powers servos (V+) | Common ground with ESP32; NOT off the Pi |
| Camera | USB UVC webcam | Vision + one onboard mic | Shared via CameraManager |
| Mic A | USB microphone | Wake-word | Resolved by name (self-healing index) |
| Mic B | USB microphone | Conversation / STT | One mic mounted on the camera |
| Audio out | HDMI audio (vc4hdmi0) | Speech output | BT speaker "AR-SPJ" optional |
| Face display (planned) | Teclast P30T (TLC005) | Tablet "face" — animated eyes/mouth | Allwinner A523, Mali-G57 MC1, 4 GB RAM, 10.1" 1280×800, Android 14 |

## Notes / cautions

- **Servo power** must come from the dedicated 5–6 V supply, never the Pi's 5V rail.
- **I2C** on the ESP32-S3 is GPIO 8 (SDA) / GPIO 9 (SCL) — the classic-ESP32 GPIO22 pin
  does **not** exist on the S3.
- Tablet vendor RAM spec ("10/15 GB") is virtual — treat as **4 GB real**; keep the face app
  lightweight (2D Canvas, vanilla JS).
