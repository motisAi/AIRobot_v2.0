# Bug #014 — Face recognition threshold 0.50 too strict for dlib; master shown as "Unknown"

- **Date found:** 2026-09-05
- **Status:** fixed
- **Area:** vision
- **Files touched:** `modules/vision/face_recognition.py`, `config/config.yaml`, `tools/reenroll.py`
- **Commit(s):** e2cc602

## Symptom
Stella did not recognise Moti (reported "Unknown"), and the displayed similarity for real matches looked absurdly low (~0.06-0.08).

## Root cause
Not enrollment. The distance threshold 0.50 was stricter than dlib's norm (~0.6), so valid matches were rejected. The displayed score `1 - dist/threshold` also made good matches look near zero. Separately, `enroll_master.py --auto` could hang indefinitely.

## Fix
- `face_recognition_threshold` raised to 0.60; score changed to `max(0, 1 - distance)`.
- Re-enrolled with new tool `tools/reenroll.py` (dlib/HOG detection, hard 40 s cap, spoken guidance). Run with the service stopped.
Live similarity went from ~0.06 to ~0.65-0.68.

## How to verify
```bash
journalctl -u airobot -f | grep "Face detected"
```
Expect `Face detected: Moti` with similarity >= 0.60 while facing the camera.

## Will it come back?
Partly — similarity dipped again to ~0.45-0.58 after switching to the USB camera (2026-09-12/13); re-enrolling under current lighting (`python reenroll.py`, service stopped) is still pending per `docs/CONTINUE.md`.
