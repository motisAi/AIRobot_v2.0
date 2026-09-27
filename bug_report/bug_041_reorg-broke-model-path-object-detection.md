# Bug #041 — Reorganisation moved hailo_10h.py one level; object detection lost data/models

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** vision
- **Files touched:** `parts_used/hailo_10h.py`, `parts_used/esp32_controller.py`
- **Commit(s):** 36d7a7f

## Symptom
After the 2026-09-19 reorganisation the log said "No DNN model found… Object detection off" although `data/models/yolov8n.onnx` was present.

## Root cause
`PROJECT_ROOT = Path(__file__).parent.parent.parent` was written for `modules/vision/`. In `parts_used/` that resolves to the home directory, so `MODELS_DIR` pointed outside the repo. Same pattern in `esp32_controller.py` (sys.path only).

## Fix
`PROJECT_ROOT = Path(__file__).resolve().parent.parent` in both files. Guardian and `tests/test_imports.py` did not catch it because importing succeeds; only the runtime log shows it.

## How to verify
`journalctl -u airobot -b | grep "DNN detector ready"` shows `ONNX: yolov8n.onnx`.

## Will it come back?
Any time a file with a `parent.parent…` root hack is moved. Rule: compute the repo root once (`config.settings.PROJECT_ROOT`) instead of counting parents. Guardian should also grep the log for "Object detection off" — added to the log-health patterns as a warning.
