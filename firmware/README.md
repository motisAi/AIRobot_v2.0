# Stella firmware (everything needed to build lives in this folder)

| Sketch | Board | Drives | Talks to the Pi as |
|---|---|---|---|
| `stella_s3/` | **ESP32-S3** | eyes (dual GC9D01) **+** arm (5× MG996R) **+** hand (5 fingers) | one serial link, CH343 `usb-1a86_*` |
| `hand/hand.ino` | ESP32-S3 | hand only (old reference — **never flash it with the arm connected**: it drives ch0–4 to their ends at boot) | |
| `hand_test/`, `i2c_scan/` | test sketches | | |

`stella_s3/` is the real firmware: `stella_s3.ino` (serial router + setup/loop), `eyes.cpp`
(custom GC9D01 driver + animation, on its own task on core 0) and `servos.cpp` (PCA9685,
smooth arm motion, calibration, finger gestures, on core 1). The protocol is documented at
the top of `stella_s3.ino`; `help` over serial prints it.

- **Eyes:** one UPPERCASE letter — `G x y` (gaze −100..100), `B`, `C`, `E n`, `P n`, plus `T`/`M` for orientation checks.
- **Hand:** the same lowercase words the Pi already sends — `fist|open|hello|middle|middlehold|point|peace|thumbs|count N|set 10110|f/m/s …|speed`.
- **Arm:** `pos`, `home`, `pose <roll> <pitch> <elbow> <shoulder> <base>`, `en/j/n <ch> …`, `spd`, `off <ch>`, `x` (all limp),
  `osc <ch> <us> <ms> <times>` (smooth swing in place, e.g. a handshake), `motion smooth|classic`
  (minimum-jerk, human-like — the default — or the older accelerate/cruise/brake profile).
- **Calibration:** built into `servos.cpp` (user-confirmed 2026-11-22); `lim`/`hm`/`sethome` save overrides in NVS, `dump` prints them, `clearcal` returns to the built-in values.

Pinouts: `docs/hardware/wiring.md` (the single wiring map). Eyes SPI DIN4/CLK7/DC10,
RST1 11 / RST2 5, CS1 2 / CS2 6, BL1 1 / BL2 3. PCA9685 I²C on SDA GPIO12 / SCL GPIO9 @0x40 (SDA moved off GPIO8 on 2026-11-23: damaged on the board in use);
ch0–4 fingers (pinky, ring, middle, index, thumb), ch5 wrist roll, ch6 wrist pitch,
ch7 elbow, ch8 shoulder, ch9 base.

**Safety at power-up:** servo power just switched on → every servo stays limp until commanded.
ESP32 reset only (reflash, the Pi opening the port) → the PCA9685 keeps its pulses and the
firmware reads them back, so the arm keeps holding. Fingers do **not** close at boot anymore.

## Build + flash from the laptop (PlatformIO)

```bash
cd firmware
pio run -t upload --upload-port COM7    # COMx = the board's CH343 UART port on the PC
```
`firmware/platformio.ini` pins **`espressif32@6.5.0`** (arduino-esp32 2.0.x — the eyes were
proven on it) and `-DARDUINO_USB_CDC_ON_BOOT=0` (Serial = the CH343 UART type-C).
Stop anything holding the port first (the `airobot` service on the Pi, a serial monitor).
Opening the port resets the ESP32 — harmless, the servos keep holding.

Quick test (serial @115200): `ver`, `pos`, `G -60 0` (eyes look left), `B`, `E 1`,
`open`/`fist`/`hello` (fingers), `home` (arm, only joints that are ON).

## Why not TFT_eSPI anymore

The vendored `libraries/TFT_eSPI` (supplier's 2.5.0 with the GC9D01 driver) crashes in
`tft.init()` on arduino-esp32 3.x and draws **horizontal stripes** on 2.0.x — a library bug
with the GC9D01 on the S3, found by bisection (bit-bang works → raw hw-SPI works → TFT_eSPI
fails). `eyes.cpp` drives the panels directly (init sequence from Waveshare's
"0.71inch-DualEye-LCD-Module-Demo"). `libraries/` is no longer used by `stella_s3`.

`.pio/` and `build/` outputs are git-ignored.
