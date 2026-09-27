# Bug #039 — YOLO ran on every frame (3-4 CPU inferences/s) and loaded even with no camera

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** vision
- **Files touched:** `modules/vision/object_detection.py`
- **Commit(s):** 231740d

## Symptom
Stella was sluggish overall; CPU load high from object detection even when no camera was attached.

## Root cause
Since bug #032 revived detection on OpenCV-DNN (CPU), the detector ran inference on every delivered frame and was loaded unconditionally at startup.

## Fix
Detector is not loaded when there is no camera; `MIN_DETECT_INTERVAL = 1.0` s throttles inference.

## How to verify
```bash
top -bn1 | head -15        # python CPU well below saturation while idle
journalctl -u airobot -n 200 | grep -i "detector"   # "not loaded" when no camera
```

## Will it come back?
No; if on-NPU detection is enabled later the throttle may be relaxed deliberately.
