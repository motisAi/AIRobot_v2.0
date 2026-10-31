# Hardware — Wiring & Connection Map (canonical)

_The single source of truth for every physical connection on Stella: ports, pins, buses,
power and the config keys that depend on them. Update it whenever something is plugged in,
moved or rewired. Other docs (`bom.md`, `../schematics/system-overview.md`,
`../firmware/hand-esp32.md`, the briefs) describe parts and data-flow; **wiring lives here**._

Last verified: camera, mics, fan, eyes pinout, network on **2026-10-27**; hand on 2026-08-15.
Items marked **⚠ verify** were taken from datasheets/overlays, not measured on the robot.

---

## 1. Compute / host

| Item | Detail |
|------|--------|
| SBC | Raspberry Pi 5 (4 GB) |
| AI accelerator | Hailo-10H NPU on PCIe/M.2 (`/dev/hailo0`, `h10-hailort 5.1.1`) — offline LLM only; vision runs on the CPU |
| OS | Ubuntu Server 24.04 (noble, py3.12), headless. **Note:** apt sources also carry the Raspberry Pi Debian repos — never let apt cross-grade core libs (see `../briefs/stella-camera-setup-ubuntu.md`) |
| Python | venv `~/AIRobot_v2.0/venv` (`include-system-site-packages = true`, plus `zz_usrlocal_dist.pth` for the from-source libcamera bindings) |
| Service | systemd `airobot` (Restart=on-failure, RestartSec=5) |
| Access | user `moti_ai`; mDNS `motiai.local`; laptop ssh alias **`stella`** (IPv4 pinned). Old alias `rpi5` points at a dead IP |
| Cooling | 4-wire fan on the **J17 fan header** (5V / GND / PWM / TACH). Controller enabled with `dtparam=cooling_fan=on` in `/boot/firmware/config.txt` (+ `pwm-fan` in modules-load). ~2950 RPM at state 1/4; fan is **off below ~50 °C by design** |
| PSU | USB-C, official 27 W recommended (brown-outs seen with weaker supplies) |

### Pi 5 pin / port allocation

| Pi 5 resource | Used by | Notes |
|---------------|---------|-------|
| **CSI camera connector** (CAM/DISP ⚠ verify which of the two) | Camera Module 3 **Wide NoIR** | see §2 |
| **I²S / PCM** GPIO18 (BCLK) · GPIO19 (LRCLK/WS) · GPIO20 (DIN) · GPIO21 (DOUT) | I²S audio HAT (2× INMP441 + MAX98357A) | overlay `googlevoicehat-soundcard`; see §3 |
| GPIO16 ⚠ verify | MAX98357A **SD_MODE** (amp enable) if driven by the overlay | otherwise tie SD_MODE high |
| **J17** 4-pin fan header | cooling fan | see above |
| PCIe (M.2 HAT) | Hailo-10H | do not disturb |
| `dtparam=i2c_arm=on`, `dtparam=spi=on` | nothing on the Pi side today | **The PCA9685 is NOT on the Pi** — it hangs off the hand's ESP32 (§4) |
| USB-A (×4) | ESP32-S3 hand (`/dev/ttyACM0`), USB PnP mic dongle, RobotNet WiFi dongle, ESP32-C3 eyes (new) | see §4, §3, §6, §5 |
| HDMI | optional monitor; HDMI audio is only the fallback output | |
| `eth0` | unused (no carrier) — plugging Ethernet in has caused brown-outs on weak PSUs | |

---

## 2. Camera — Raspberry Pi Camera Module 3 Wide NoIR (CSI)

```
Camera Module 3 Wide NoIR (IMX708, 120° FOV, AF actuator dw9807) ──ribbon──> Pi 5 CSI (CAM/DISP ⚠ verify 0/1)
```

| Item | Detail |
|------|--------|
| Sensor | `imx708_wide_noir` — **no IR-cut filter** (great in low light / with IR illumination; daylight colours skew warm). Tuning file `imx708_wide_noir.json` auto-selected |
| Mount | module sits **upside-down** in the head → `camera_hflip: true`, `camera_vflip: true` (= 180°). **After ANY re-mount run `venv/bin/python tools/camera_orientation_check.py`** with someone in front of her: on 2026-10-31 she came back inverted after a re-mount and saw nobody (neither YuNet nor dlib finds an upside-down face); turning the module 180° fixed it |
| Driver stack | kernel `imx708` + `rp1_cfe` + `pisp_be`; userspace **libcamera 0.7 built from Raspberry Pi's fork into `/usr/local`** (Ubuntu's apt libcamera cannot drive it); `picamera2 0.3.38` in the venv (optional DRM/Qt previews guarded) |
| In code | `parts_used/camera_usb.py` `CameraManager`, backend `picamera2`, 1280×720 BGR frames, continuous AF; USB/OpenCV path kept as fallback (`camera_backend: auto`) |
| Config | `hardware.camera_backend`, `camera_resolution`, `camera_hflip/vflip`, `camera_swap_rb` (false — colours verified) |
| Has a mic? | **No.** The old USB webcam's built-in mic was the wake mic; see §3 |

---

## 3. Audio — I²S HAT (command mic + speaker) and USB mic (wake)

Two mics, two roles, **two physical devices** (bug_055 — never share one mic for both roles).
Roles are canonical in `../architecture/stella-architecture.md §4`; the physical map is here.

### I²S HAT — 2× INMP441 microphones + MAX98357A amplifier ("voicehat")

ALSA card `snd_rpi_googlevoicehat` (overlay `dtoverlay=googlevoicehat-soundcard`). Role: **COMMAND mic** (STT) and **speaker** (TTS).

| HAT part | Pin | Pi 5 pin (BCM) | Notes |
|----------|-----|----------------|-------|
| INMP441 (both) | VDD | 3V3 | |
| INMP441 (both) | GND | GND | |
| INMP441 (both) | SCK | **GPIO18** (PCM_CLK / BCLK) | shared |
| INMP441 (both) | WS | **GPIO19** (PCM_FS / LRCLK) | shared |
| INMP441 (both) | SD | **GPIO20** (PCM_DIN) | both mics share the data line |
| INMP441 #1 | L/R | GND | = left channel |
| INMP441 #2 | L/R | 3V3 | = right channel (⚠ verify which is which on your board) |
| MAX98357A | VIN | 5V | |
| MAX98357A | GND | GND | |
| MAX98357A | BCLK | **GPIO18** | shared with the mics |
| MAX98357A | LRC | **GPIO19** | shared |
| MAX98357A | DIN | **GPIO21** (PCM_DOUT) | |
| MAX98357A | SD_MODE | GPIO16 ⚠ verify / or tied high | overlay-driven enable |
| MAX98357A | GAIN | per build | |
| Speaker | + / − | amp output | |

Config: `hardware.speech_microphone_name: "voicehat"`, `speech_microphone_rate: 48000`,
`audio_output_name: "voicehat"`, `audio_output_device: "auto"`.

### USB mic dongle — WAKE word (temporary)

| Item | Detail |
|------|--------|
| Device | "USB PnP Sound Device" (C-Media), ALSA card id `Device`, by-id `usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*`, native 44.1 kHz |
| Role | **WAKE** mic (Vosk "stella"/"hey stella") — stand-in since the camera mic went away; weak far-field pickup / AGC (bug_053). Replace with a 2nd INMP441 when it arrives |
| Config | `hardware.wake_word_microphone_name: "USB PnP Sound Device"`, `wake_mic_id: "usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*"`, `microphone_rate: 44100` |

Audio out fallback: HDMI (`vc4hdmi0/1`, auto-detected) when `audio_output_name` is null.

---

## 4. Robotic hand — ESP32-S3 + PCA9685 (unchanged)

```
Pi 5 ──USB (CH343)──> ESP32-S3-WROOM-1 ──I2C 0x40──> PCA9685 ──PWM ~50 Hz──> 5× finger servos
                                                        ▲ V+ from a SEPARATE 5–6 V supply, common GND
```

| Link | Detail |
|------|--------|
| Pi → ESP32-S3 | USB serial `/dev/ttyACM0`, **115200**, newline ASCII (`firmware/hand/hand.ino`, driver `parts_used/esp32_hand.py`) |
| ESP32-S3 → PCA9685 | I²C addr **0x40**, **SDA = GPIO 8, SCL = GPIO 9** (on the S3) |
| PCA9685 power | logic VCC from ESP32 3V3; **servo V+ from an external 5–6 V supply, never the Pi**; common ground |
| Channels | 0 pinky · 1 ring · 2 middle · 3 index · 4 thumb (open `{1500,1500,1500,1500,2000}` · closed `{2600,2600,2600,2600,500}` µs) |
| Boot | homes to a closed fist (~2.5 s). Stop `airobot` before reflashing (it holds the port) |

Protocol and gotchas: `../firmware/hand-esp32.md`.

---

## 5. Eyes — ESP32-C3 + dual GC9D01 round LCDs (new, 2026-10-27)

```
Pi 5 ──USB (power + gaze serial 115200)──> ESP32-C3 ──SPI (shared SCLK/MOSI/DC)──> 2× GC9D01 160×160
                                               └──I2C (GPIO8 SDA / GPIO9 SCL)──> PCA9685 (head/arm servos, planned)
```

Module: bare "0.71 inch Dual-Eye LCD" (two GC9D01, 160×160, each with its own CS, RST and backlight).
Firmware: `firmware/stella_eyes/stella_eyes.ino` (board "ESP32C3 Dev Module", vendor TFT_eSPI dual-reset build).

| Eye module pin | ESP32-C3 GPIO | Role / note |
|----------------|---------------|-------------|
| VCC | **3V3** | **not 5 V** |
| GND | GND | |
| SCL / SCLK / CLK | **GPIO7** | shared SPI clock (`TFT_SCLK 7`) |
| SDA / MOSI / DIN | **GPIO4** | shared SPI data (`TFT_MOSI 4`) |
| DC / RS | **GPIO10** | shared data/command (`TFT_DC 10`, moved off GPIO9) |
| RST1 (left eye) | **GPIO0** | `TFT_RST 0` (moved off GPIO8) |
| RST2 (right eye) | **GPIO5** | `TFT2_RST 5` (vendor default) |
| CS_L (left eye) | **GPIO2** | sketch toggles CS (`TFT_CS -1`) |
| CS_R (right eye) | **GPIO6** | |
| BL_L (left backlight) | **GPIO1** | HIGH = on |
| BL_R (right backlight) | **GPIO3** | HIGH = on |
| — (reserved) | **GPIO8 SDA / GPIO9 SCL** | I²C to the 16-ch PCA9685 — never use for the eyes |
| — (reserved) | GPIO18/19 (USB), GPIO20/21 (UART0) | |

TFT_eSPI `User_Setup.h` (ESP32-C3 block): change only `TFT_DC 10` and `TFT_RST 0`; keep `TFT_MOSI 4`,
`TFT_SCLK 7`, `TFT_CS -1`, `TFT2_RST 5`. Init pattern: both CS low → one `tft.init()` → CS high.

Serial protocol (Pi → eyes): `G <x> <y>` gaze −100..100 · `B` blink · `C` center · `E <n>` emotion 0–4 ·
`P <n>` pupil 0–100. Pi side: `modules/vision/eye_tracker.py` (`EyeController`), config `hardware.eyes_*`.

> ⚠ Port detection: the eye board enumerates as an Espressif USB-JTAG/serial device
> (`/dev/serial/by-id/usb-Espressif_*`), the hand as CH343 (`usb-1a86_*`). Pin `hardware.eyes_serial`
> to the Espressif by-id path so the eyes never grab the hand's port (or vice-versa).

Power: ~100–200 mA from the Pi's USB is fine for the eyes alone.

---

## 6. Network

| Item | Detail |
|------|--------|
| Home WiFi client | onboard radio (MAC `88:a2:9e:ab:1a:d9`), name **`wlan0` or `wlan1`** depending on boot order (bug_052). SSID **"Arik hadash at moti"** (old "Arik -2.4G-ext" kept as fallback). Netplan gives it a **static 10.100.102.133/24**, gw 10.100.102.1, DNS 10.100.102.1 + 1.1.1.1 (the extender does not relay DHCP) and keeps `dhcp4: true`. wpa units have a retry drop-in (`/etc/systemd/system/netplan-wpa-wlan{0,1}.service.d/`) |
| RobotNet AP | TP-Link Archer T2U Plus (RTL8821AU, `8821au` DKMS) = `wlx984827df22bb`, **10.0.0.1/24**, not managed by netplan — **never touch**; it is also the **recovery path** (laptop → RobotNet → `ssh moti_ai@10.0.0.1`) |
| mDNS | `motiai.local` (avahi) — on the laptop use IPv4 (`ssh -4`) or the `stella` alias |
| Ports | dashboard `:5000`, face bridge `:8080` (ws `/ws`), hailo-ollama `:8000`, mosquitto `:1883` |
| Router to-do | add a DHCP reservation `88:a2:9e:ab:1a:d9 → 10.100.102.133` |

---

## 7. Planned / not yet wired

| Subsystem | Wiring plan | Source |
|-----------|-------------|--------|
| **360° LiDAR** WitMotion/Gouke COIN-D6 (UART 230400) | **Option A (recommended):** ZH1.5 cable → CH340 USB adapter → Pi USB → `/dev/ttyUSB0`. **Option B:** direct UART — 5V (yellow) → pin 2/4, GND (white) → pin 6, LiDAR TXD (red) → Pi RXD **GPIO15** (pin 10), LiDAR RXD (black) → Pi TXD **GPIO14** (pin 8); enable UART, free the console. Spin: `AA 55 F0 0F`, stop `AA 55 F5 0A`. Power it from a powered hub / separate 5 V (motor surge) | `../briefs/stella-eyes-lidar/WIRING.md`, driver `modules/navigation/lidar_coin_d6.py` |
| **Head pan/tilt** 2× MG90S | PCA9685 **on the eyes' ESP32-C3** (I²C GPIO8/9), channels 0 pan / 1 tilt; software limits ±60° / ±30°; camera mounted ON the head | `../briefs/stella-eyes-head-arm-brief.md` |
| **Arm (10 DOF)** | same PCA9685 (or a 2nd at 0x41): 2 shoulder pan/3 tilt/4 elbow/5 wrist pitch/6 wrist roll/7–11 fingers; ArUco 2 cm marker on the hand; external servo PSU | brief above |
| 2nd **INMP441** | replaces the USB wake mic (I²S, shares SCK/WS, own SD or L/R pairing) | §3 |
| Wheels / motors, IMU, encoders, ultrasonic | ESP32 motor controller over WiFi (RobotNet) — TBD | `navigation` config (all `false`) |
| SIM7600X 4G | not fitted (`/dev/ttyAMA1` reserved in config) | |

---

## 8. Change log

- **2026-10-31** — Camera found inverted after a re-mount (both face detectors blind) → module turned 180°; `tools/camera_orientation_check.py` and `tools/face_diag.py` added. Master re-enrolled with the CSI camera (`tools/reenroll.py`, 14 samples, faces 104–391 px). Liveness (blink) gate added (`modules/vision/liveness.py`): a photo no longer unlocks master mode, the guard or enrolment.
- **2026-10-27** — USB webcam removed → Camera Module 3 Wide NoIR on CSI (libcamera from source). Wake mic → USB PnP dongle (temporary). Fan on J17 enabled. Home WiFi switched to "Arik hadash at moti" with static IP + wpa retry fix. Eyes (ESP32-C3 + dual GC9D01) pinout finalised; GPIO8/9 reserved for the PCA9685.
- **2026-09-26** — audio/service/camera verification (USB era).
- **2026-08-15** — hand wiring verified; classic ESP32 replaced by ESP32-S3 after a short.
