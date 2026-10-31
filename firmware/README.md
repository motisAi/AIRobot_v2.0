# Stella firmware (everything needed to build lives in this folder)

| Sketch | Board | Drives | Talks to the Pi as |
|---|---|---|---|
| `stella_s3/stella_s3.ino` | **ESP32-S3** | hand (PCA9685) **+** eyes (dual GC9D01) | one serial link (see below) |
| `hand/hand.ino` | ESP32-S3 | hand only (reference) | `usb-1a86_*` (CH343) |
| `hand_test/`, `i2c_scan/` | test sketches | | |

The hand and the eyes are the **same ESP32-S3**. `stella_s3.ino` is the real,
combined firmware: one serial stream carries both hand gestures (lowercase words:
`fist|open|hello|middle|point|peace|thumbs|count N|set 11111|f/m/s …|speed|off|pos`)
and eye commands (`G x y`·`B`·`C`·`E n`·`P n`), routed by the first character.

Pinouts: `docs/hardware/wiring.md` (the single wiring map). Hand I²C on GPIO8/9
(PCA9685 @0x40); eyes SPI DIN4/CLK7/DC10, RST1 11 / RST2 5, CS1 2 / CS2 6, BL1 1 / BL2 3.

## ⚠ Build with arduino-esp32 **2.0.x** (not 3.x)

The vendored `libraries/TFT_eSPI` (supplier's 2.5.0, GC9D01 + second reset `TFT2_RST`,
Stella's pins pre-set) **crashes in `tft.init()` on arduino-esp32 3.x** (ESP-IDF 5:
`begin_tft_write` StoreProhibited → boot loop that looks exactly like "dim blue, no
image"). Use the 2.0.x core:
- **PlatformIO:** `platform = espressif32@6.5.0`.
- **Arduino IDE:** install ESP32 core **2.0.x** (Boards Manager), board *ESP32S3 Dev Module*.

Do **not** install the stock TFT_eSPI from the Library Manager over the vendored one
(it lacks the GC9D01 driver and the second reset). Examples/docs were stripped to keep
the repo small; the original is "0.71inch-DualEye-LCD-Module-Demo" from Waveshare.

## USB port / CDC

The S3 DevKit has two type-C ports: native-USB (`usb-Espressif_*`) and a CH343 UART
(`usb-1a86_*`). Pick one and set it in the build:
- **CH343/UART port** (reliable flashing + reset; current default): `-DARDUINO_USB_CDC_ON_BOOT=0` (Serial = UART0).
- **native-USB port:** `-DARDUINO_USB_CDC_ON_BOOT=1` (Serial = USB CDC).

## Build + flash from the laptop (PlatformIO)

```ini
# platformio.ini
[env:esp32-s3]
platform = espressif32@6.5.0        ; arduino-esp32 2.0.x — REQUIRED (see above)
board = esp32-s3-devkitc-1
framework = arduino
upload_speed = 460800
build_flags = -DARDUINO_USB_CDC_ON_BOOT=0
lib_extra_dirs = <repo>/firmware/libraries
```
```bash
pio run -t upload --upload-port COM6   # COMx = the board's port on the PC
```
Flashing over the S3's native-USB/JTAG needs **esptool 5.x** (4.5.1 fails the baud
change: "No serial data received"). PlatformIO 6.5.0's bundled esptool works; or
`python -m esptool --chip esp32s3 --port COMx --baud 115200 write-flash 0x0 merged.bin`.

Quick test (serial @115200): `pos` (hand reports servo positions), `open`/`fist`/`hello`
(hand), `G -60 0` (eyes look left), `B` blink, `C` center, `E 1` happy, `P 80` big pupils.
`build/` and `.pio/` outputs are git-ignored.
