# Hardware — Wiring & Connection Map

_Current physical connections of the Stella robot. Update this whenever something is
plugged in, moved, or rewired._

Last verified: 2026-08-15.

## Compute / host

| Item | Detail |
|------|--------|
| SBC | Raspberry Pi 5 |
| AI accelerator | Hailo-10H NPU (PCIe/M.2) |
| OS | Ubuntu Server 24.04.4 LTS, aarch64, headless (SSH) |
| Python | 3.12 (venv at `~/AIRobot_v2.0/venv`) |
| Host user | `moti_ai` — SSH alias `rpi5` |
| Service | systemd `airobot` (Restart=on-failure) |

## Robotic hand (5-finger) — ESP32 + PCA9685

```
Pi 5 ──USB(CH343)──> ESP32-S3-WROOM-1 ──I2C──> PCA9685 ──PWM──> 5x servos (fingers)
                                                   ▲
                                    external 5–6V servo supply (V+)
```

| Link | Detail |
|------|--------|
| Pi → ESP32 | USB serial, device `/dev/ttyACM0` (CH343 USB-serial), **115200 baud** |
| ESP32 → PCA9685 | I2C, address **0x40**. **SDA = GPIO 8, SCL = GPIO 9** on the ESP32-S3 |
| PCA9685 logic | VCC from ESP32 3.3V; PWM freq ~50 Hz (SLEEP bit cleared in firmware) |
| PCA9685 servo power | **V+ from a SEPARATE 5–6V supply**, NOT from the Pi/ESP32. **Common ground** tied between the servo supply and the ESP32/PCA9685 |

### Servo → finger channel map (PCA9685 outputs)

| Channel | Finger |
|:------:|--------|
| 0 | Pinky |
| 1 | Ring |
| 2 | Middle |
| 3 | Index |
| 4 | Thumb |

On serial connect the ESP32 resets and **homes to a closed fist** (~2.5 s settle).

> ⚠️ History: a classic ESP32 shorted/overheated from a loose jumper — replaced with the
> ESP32-S3-WROOM-1. Keep servo power on its own supply with a solid common ground.

## Audio & camera (USB)

| Item | Role |
|------|------|
| USB camera (UVC) | Shared via `CameraManager` (face recognition, motion guard, hand-mirror, VLM, dashboard) |
| Mic A | Wake-word thread (Vosk) |
| Mic B | Conversation / STT thread (one mic is mounted on the camera) |
| Audio out | HDMI — `sysdefault:CARD=vc4hdmi0` |
| Bluetooth speaker (optional) | "AR-SPJ" — used only if stable; no USB BT dongle currently |

Mics are resolved **by name** each open, so a shifted ALSA/USB index self-heals
(see `modules/audio/wake_word.py` + the hardware watchdog).

## Planned / not yet wired

- **Tablet face** (Teclast P30T) over WiFi + WebSocket — see
  [`../briefs/robot-face-tablet-brief.md`](../briefs/robot-face-tablet-brief.md). No physical
  wiring to the Pi; LAN only, powered/charged on its own battery (acts as a UPS for the face).
