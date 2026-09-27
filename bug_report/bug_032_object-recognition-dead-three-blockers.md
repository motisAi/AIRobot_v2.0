# Bug #032 — Object recognition dead: no model file, Hailo disabled, wrong hailort pip (+ stale root config.yaml)

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** vision
- **Files touched:** `.gitignore`, `data/models/yolov8n.onnx` (gitignored), `modules/vision/object_detection.py` (loads via OpenCV-DNN), root `config.yaml` (deleted)
- **Commit(s):** cf767f8

## Symptom
Stella never detected objects; the object-detection module silently did nothing.

## Root cause
Three stacked blockers found by the architecture audit: (1) no model file on disk; (2) `enable_hailo: false`; (3) the pip `hailort` was 4.23.0 while the system HailoRT is 5.1.1. Also a stale root `config.yaml` that was never loaded (the real one is `config/config.yaml`) misled edits — the "Gonzo trap".

## Fix
Exported a stock `yolov8n.onnx` (Ultralytics, opset 12, 640) into `data/models/` (gitignored); `HailoDetector` loads it via OpenCV-DNN with `enable_hailo` still false (backend=opencv_dnn). Deleted the never-loaded root `config.yaml`. Verified: raw output (1,84,8400), real detections (class 57 couch) on a camera frame. Rollback: delete the .onnx.

## How to verify
```bash
journalctl -u airobot -n 300 | grep -i "backend=opencv_dnn"
ls -la ~/AIRobot_v2.0/data/models/yolov8n.onnx
```

## Will it come back?
Yes on a fresh clone — the .onnx is gitignored and must be re-exported/copied. On-NPU detection is a separate later roadmap phase (HailoRT 5.2.0).
