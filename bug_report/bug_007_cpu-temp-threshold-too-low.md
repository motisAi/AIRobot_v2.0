# Bug #007 — CPU temperature threshold too low, tripping thermal protection in normal use

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** power
- **Files touched:** `main.py`
- **Commit(s):** 505b75d, 33a3488

## Symptom
Thermal warnings / protective behaviour under ordinary load on the Pi 5 (exact user-visible effect unclear from history — commit messages only).

## Root cause
The over-temperature threshold in `main.py` was set below the Pi 5's normal operating range under AI load.

## Fix
Threshold raised to 80 C (505b75d) and then to 85 C (33a3488).

## How to verify
```bash
vcgencmd measure_temp
journalctl -u airobot | grep -i "temp"
```
No temperature alerts under idle/normal load.

## Will it come back?
No, unless the threshold is lowered again. The Pi 5 firmware throttles itself at ~85 C anyway.
