# Bug #045 — "New person" greeting with nobody there (phantom face on a wall)

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** vision
- **Files touched:** `modules/vision/face_recognition.py`, `main.py`
- **Commit(s):** see 2026-09-19 phantom-face commit

## Symptom
Stella announced she was meeting a new person ("Hello! I don't think we've met…"), waved, and her gaze pointed at a blank wall — no one was on camera.

## Root cause
Face detection uses the OpenCV Haar frontal cascade (`face_backend: opencv`), which throws false positives on wall/painting texture. The GUARD path already required an unknown face to persist ≥3 detections before alerting (phantom filter), but the **greet/enroll path had no such gate**: a single phantom Haar box with `face_id=='unknown'` immediately waved and started a "we haven't met" conversation. The gaze feed then tracked the phantom's location — "pointing at the wall".

## Fix
1. `_detect_faces` (opencv branch): after a face box, run the eye cascade on it and drop boxes with no detectable eye — wall texture almost never shows eyes. Cascades cached on the instance (no per-frame reload). minNeighbors 7→8.
2. `main.py`: new `_note_unknown(now)` windowed streak (resets if the previous unknown was >4 s ago), shared by both paths. The greet/enroll path now requires streak ≥3 before waving/greeting, same as guard.

## How to verify
Point the camera at a blank/ textured wall: no "new person" greeting, no wave, `journalctl -u airobot | grep "Unknown face"` stays quiet. A real person still greets after ~1 s (3 detections).

## Will it come back?
Only if the eye cascade is removed or a real face pattern (e.g. a poster of a face) sits in view — that would legitimately look like a person. Switching `face_backend` to a dlib/DNN detector would remove Haar phantoms entirely (future).
