# Firmware — ESP32-S3 Robotic Hand

Firmware source: `firmware/hand/hand.ino`. Host-side helper: `firmware/handctl.py`.
Pi-side driver: `parts_used/esp32_hand.py`.

## Target / toolchain

- **Board:** ESP32-S3-WROOM-1
- **arduino-cli**, FQBN `esp32:esp32:esp32s3`
- **Serial:** `/dev/ttyACM0` (CH343 USB-serial), 115200 baud, newline-terminated ASCII

## I2C / PWM

- I2C to PCA9685 at address **0x40**, **SDA = GPIO 8, SCL = GPIO 9**.
- PCA9685 driven by raw register writes. `setFreq()` **clears the SLEEP bit** before setting
  the prescaler, or the servos won't move:
  `write MODE1 0x10 (sleep) → set PRESCALE → 0x20 (wake) → 0xA0 (auto-increment + restart)`.
- PWM ~50 Hz; per-servo pulse calibration for open/closed positions.

## Servo → finger map (PCA9685 channel)

| Channel | Finger |
|:------:|--------|
| 0 | Pinky |
| 1 | Ring |
| 2 | Middle |
| 3 | Index |
| 4 | Thumb |

On boot / serial connect the ESP32 resets and **homes to a closed fist**.

## Serial command protocol (Pi → ESP32)

One command per line, lowercase ASCII + `\n`.

| Command | Action |
|---------|--------|
| `fist` | Close all fingers (home pose) |
| `open` | Open all fingers |
| `hello` | Wave gesture (greeting) |
| `middle` | Raise middle finger ~3 s, then lower |
| `middlehold` | Raise middle finger and **keep** it up until told to lower |
| `point` | Index finger only |
| `peace` | Index + middle (victory) |
| `thumbs` | Thumbs-up |
| `count <n>` | Show a number of fingers |
| `set <5 bits>` | Per-finger open/closed mask — used by the **hand-mirror** (e.g. `set 10100`) |
| `lower` / `rest` | Return the hand to a fist |

Bit order for `set` matches the channel map above (pinky … thumb).

## Pi-side driver notes (`parts_used/esp32_hand.py`)

- Holds **one persistent serial link** (the ESP32 keeps its pose between commands; it only
  resets on (re)connect).
- `reconnect(port)` re-opens the link — used by the hardware watchdog when `/dev/ttyACM*`
  renumbers or the link goes dead.
- Friendly gesture names map to firmware commands via the `GESTURES` dict.

## Gotchas

- If nothing moves: check the SLEEP-bit sequence and that **external servo power** is present
  with a **common ground**.
- The classic-ESP32 pin GPIO22 does not exist on the S3 — I2C was moved to GPIO 8/9 for that
  reason.
