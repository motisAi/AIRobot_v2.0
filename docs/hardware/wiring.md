# Hardware — Wiring & Connection Map

_Current physical connections of the Stella robot. Update this whenever something is
plugged in, moved, or rewired._

Last verified: audio, service and camera on 2026-09-26 (commit a2e3229); hand wiring on 2026-08-15.

## Compute / host

| Item | Detail |
|------|--------|
| SBC | Raspberry Pi 5 |
| AI accelerator | Hailo-10H NPU (PCIe/M.2) — offline LLM only; vision is on the CPU |
| OS | Ubuntu Server 24.04.4 LTS, aarch64, headless (SSH) |
| Python | 3.12 (venv at `~/AIRobot_v2.0/venv` — the only interpreter the service uses) |
| Host user | `moti_ai` — SSH alias `rpi5` |
| Service | systemd `airobot` (Restart=on-failure, RestartSec=5, After=hailo-ollama.service) |

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

Canonical mic roles, thread ownership and the wake → command → wake handoff live in
[../architecture/stella-architecture.md §4](../architecture/stella-architecture.md#4-audio--the-canonical-two-mic-design).
This table mirrors that section.

| Item | Identity | Role |
|------|----------|------|
| USB camera (UVC) | "Signo Camera WB-400 Auto Focus Camera" (`/dev/video*`, found by name; CSI nodes skipped) | Shared via `CameraManager` (face recognition, motion guard, hand-mirror, VLM, object detection, dashboard) |
| Mic A — **WAKE** | the camera's built-in mic — ALSA card id `Camera` (card 3 today), `/dev/snd/by-id/usb-Signo_Camera_WB-400_Auto_Focus_Camera*`, PortAudio `Auto Focus Camera: USB Audio (hw:3,0)`, 48 kHz native | Wake-word thread (Vosk); `hardware.wake_word_microphone_name: "Auto Focus Camera"` |
| Mic B — **COMMAND** | "USB PnP Sound Device" C-Media dongle — ALSA card id `Device` (card 0 today), `/dev/snd/by-id/usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*`, PortAudio `USB PnP Sound Device: Audio (hw:0,0)`, 44.1 kHz native | Conversation / STT capture on the conversation thread; `hardware.speech_microphone_name: "USB PnP Sound Device"` |
| Audio out | HDMI — `vc4hdmi0` / `vc4hdmi1`, auto-detected (`hardware.audio_output_card: null`); cards 1–2 are playback-only | TTS: piper → espeak fallback → `aplay` |
| Bluetooth speaker (optional) | "AR-SPJ" — used only if stable; no USB BT dongle currently | — |

Rules (from the architecture doc): the two roles are on two physical devices and must never resolve to the same one
(single-mic handoff failed — bug_055); each stream is opened, read and closed by one thread; devices are resolved by
stable identity (by-id symlink → `controlC<N>` → PortAudio `(hw:N,`), never by index — ALSA card numbers and PortAudio
indices both move across reboots and replugs. PulseAudio runs in the user session with the camera mic as its default
source; the robot never records through Pulse.

## Network

| Item | Detail |
|------|--------|
| Home WiFi | onboard radio, client (`wlan0`; a boot race can name it `wlan1` — dual-name netplan, bug_052) |
| RobotNet AP | TP-Link Archer T2U Plus (RTL8821AU, `8821au` DKMS), 10.0.0.1/24, `robotnet-*.service` |
| mDNS | `motiAi.local` (avahi) — see `../README.md` |
| Ports | face bridge `:8080` (ws `/ws`), dashboard `:5000`, hailo-ollama `:8000`, mosquitto `:1883` |

## Planned / not yet wired

- **INMP441 I²S microphones** — the durable replacement for USB audio (bugs 002, 009, 010, 027, 037, 046, 053).
- **Tablet face** (Teclast P30T) over WiFi — the face already runs in a browser via `face_bridge/` on `:8080`; see
  [`face-subsystem.md`](face-subsystem.md), [`tablet-setup.md`](tablet-setup.md) and the original brief
  [`../briefs/robot-face-tablet-brief.md`](../briefs/robot-face-tablet-brief.md). No physical wiring to the Pi; LAN only.
- **Fan / airflow** — the Pi reaches the 80 °C soft limit under load.
