# Stella firmware (everything needed to build lives in this folder)

| Sketch | Board | Talks to the Pi as | Libraries |
|---|---|---|---|
| `hand/hand.ino` | ESP32-S3 (CH343 UART bridge) | `/dev/serial/by-id/usb-1a86_*` | Adafruit PWM Servo Driver (Arduino Library Manager) |
| `stella_eyes/stella_eyes.ino` | **ESP32C3 Dev Module**, native USB port | `/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*` | `libraries/TFT_eSPI` (vendored, pre-configured — see below) |
| `hand_test/`, `i2c_scan/` | test sketches | | |

Pinouts: `docs/hardware/wiring.md` (the single wiring map).

## Arduino IDE: point the sketchbook here, nothing else to install for the eyes

File → Preferences → **Sketchbook location** = this `firmware` folder. The IDE then
uses `firmware/libraries/` and all sketches appear under File → Sketchbook.

`libraries/TFT_eSPI` is the **display supplier's build** (TFT_eSPI 2.5.0 with the
GC9D01 160×160 driver and a second reset line `TFT2_RST`) with Stella's pins
already set in `User_Setup.h` (`TFT_DC 10`, `TFT_RST 0`, `TFT2_RST 5`, `TFT_MOSI 4`,
`TFT_SCLK 7`, `TFT_CS -1`). Do **not** install the stock TFT_eSPI from the Library
Manager over it — the stock one has neither the driver nor the second reset and
the sketch will not compile. (Examples/docs were stripped to keep the repo small;
the original zip is "0.71inch-DualEye-LCD-Module-Demo" from the supplier.)

Eyes board options (Tools menu): board *ESP32C3 Dev Module*, **USB CDC On Boot =
Enabled** (the sketch's `Serial` must be the native USB port the Pi is wired to),
everything else default.

## Command line build + flash from the Pi (board stays plugged into Stella)

```bash
# on the laptop (arduino-cli shares the IDE's cores in %LOCALAPPDATA%\Arduino15):
arduino-cli compile --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc \
    --libraries firmware/libraries --export-binaries firmware/stella_eyes
#   -> firmware/stella_eyes/build/esp32.esp32.esp32c3/stella_eyes.ino.merged.bin
scp firmware/stella_eyes/build/esp32.esp32.esp32c3/stella_eyes.ino.merged.bin stella:/tmp/

# on the Pi (service must be stopped — it holds the port):
sudo systemctl stop airobot
venv/bin/esptool --chip esp32c3 --port /dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*-if00 \
    --baud 460800 write-flash 0x0 /tmp/stella_eyes.ino.merged.bin
sudo systemctl start airobot
```

Quick test without the service: `echo "G -60 0" > /dev/ttyACM0` makes the eyes look
left, `echo B` blinks, `echo C` centres, `echo "E 1"` happy, `echo "P 80"` big pupils.
The `build/` output folder is git-ignored.
