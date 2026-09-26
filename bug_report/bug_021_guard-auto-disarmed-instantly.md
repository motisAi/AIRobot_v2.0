# Bug #021 — "Guard my house" disarmed itself ~6 s later

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `main.py`
- **Commit(s):** e2cc602

## Symptom
Moti said "guard my house"; guard armed, then a few seconds later Stella said she was auto-disarming ("welcome home"). It was impossible to arm guard while standing in front of the camera.

## Root cause
The "master seen -> welcome home -> auto-disarm" logic fired on any master face recognition, including the one immediately after arming.

## Fix
Auto-disarm now only fires when the master **returns after being away** (gap since last seen > 60 s). Arming while present no longer disarms; the welcome-home message is gated on an actual disarm.

## How to verify
Say "guard my house" while facing the camera, stay for a minute: guard remains armed (ask "is guard on?"). Leave for > 60 s and come back: it disarms with "welcome home".

## Will it come back?
No. Related open issue: `guard_mode` is not persisted across service restarts, so every restart resets guard to off.
