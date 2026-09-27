# Bug #008 — Face detection spam (tiny false faces, repeated events every frame)

- **Date found:** 2026-05-31
- **Status:** fixed
- **Area:** vision
- **Files touched:** `modules/vision/face_recognition.py`, `core/robot_brain.py`
- **Commit(s):** 58f05b5

## Symptom
The log flooded with face-detected events and the robot reacted repeatedly to the same (or non-existent, tiny) faces.

## Root cause
Haar cascade `minSize` was too small (picking up noise) and every frame emitted a new face event with no throttle.

## Fix
`minSize` raised to 80 px and a 10-second per-face throttle added. The same commit extended the speech timeout to 10 s and added speech logging.

## How to verify
Stand in front of the camera for 30 s: the journal shows one face event per ~10 s and none for background clutter.

## Will it come back?
No.
