# Bug #033 — object_detected handlers crashed and events flooded once detection was revived

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** vision
- **Files touched:** `core/robot_brain.py`, `main.py`, `modules/vision/object_detection.py`
- **Commit(s):** 62fec28

## Symptom
Right after bug #032 was fixed, the journal filled with tracebacks from the object-detected event handlers and a flood of detection events.

## Root cause
- Handlers called `save_object(...)` with a `confidence` kwarg it does not accept.
- Handlers read `obj["class"]` while the detector emits `label`.
- The detector emitted an event for every frame with detections.

## Fix
Corrected the `save_object` call and the `label` field in `core/robot_brain.py` / `main.py`; `object_detection.py` throttles emission to scene changes or every 5 s.

## How to verify
```bash
journalctl -u airobot -n 500 | grep -iE "Traceback|save_object|KeyError: 'class'"   # expect nothing
journalctl -u airobot -n 500 | grep -c object_detected                                # a few per minute, not per frame
```

## Will it come back?
No.
