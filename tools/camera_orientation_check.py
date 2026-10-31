#!/usr/bin/env python3
"""Which way is the camera mounted? Run this whenever the camera is re-mounted.

Runs the service's own YuNet face detector on ~10 s of live frames at
0/90/180/270 degrees and reports which rotation finds a face. Someone must be
in front of the camera.

    cd ~/AIRobot_v2.0 && venv/bin/python tools/camera_orientation_check.py

Works while the airobot service runs (frames via the dashboard MJPEG stream on
:5000) or with the service stopped (opens the camera directly). Best frame per
rotation -> /tmp/stella_diag/rot_<deg>.jpg (green box = detected face).

Reading the result:
  0 wins            -> config is right, nothing to do.
  180 wins          -> toggle BOTH camera_hflip and camera_vflip in config/config.yaml.
  90 or 270 wins    -> the module is mounted sideways: re-mount it (no 90-degree
                       option in the pipeline — every consumer assumes landscape).
2026-10-31: after a re-mount she was upside-down and saw nobody — neither YuNet
nor dlib finds an inverted face, so this is the first thing to check.
"""
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
YUNET = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
OUT = Path("/tmp/stella_diag")
DASHBOARD = "http://127.0.0.1:5000/video_feed"
ROTS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def frames_from_dashboard(seconds, min_w=640):
    req = urllib.request.urlopen(DASHBOARD, timeout=5)
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = req.read(16384)
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a < 0 or b < 0:
                break
            img = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
            buf = buf[b + 2:]
            if img is not None and img.shape[1] >= min_w:   # skip the 320px "No camera" placeholder
                yield img


def frames_from_camera(seconds):
    from parts_used.camera_usb import CameraManager
    cm = CameraManager()
    if not cm.start():
        raise RuntimeError("CameraManager.start() failed (is the airobot service holding the camera?)")
    try:
        time.sleep(1.0)
        t0 = time.time()
        while time.time() - t0 < seconds:
            f = cm.get_latest_frame()
            if f is not None:
                yield f.image
            time.sleep(0.1)
    finally:
        cm.stop()


def live_frames(seconds=10.0):
    try:
        urllib.request.urlopen(DASHBOARD, timeout=3).close()
        print("source: dashboard stream (service running)", flush=True)
        return frames_from_dashboard(seconds)
    except Exception:
        print("source: camera directly (service stopped)", flush=True)
        return frames_from_camera(seconds)


def main():
    if not YUNET.exists():
        print(f"YuNet model missing: {YUNET}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    dets = {}
    best = {d: (0.0, None, None) for d in ROTS}
    stats = {d: [] for d in ROTS}
    n = 0
    for img in live_frames():
        n += 1
        if n % 2:
            continue  # ~5 fps is plenty
        for deg, code in ROTS.items():
            r = img if code is None else cv2.rotate(img, code)
            h, w = r.shape[:2]
            if (w, h) not in dets:
                dets[(w, h)] = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.5, 0.3, 50)
            _, d = dets[(w, h)].detect(r)
            if d is not None and len(d):
                conf = float(max(x[-1] for x in d))
                box = max(d, key=lambda x: x[-1])[:4]
                stats[deg].append(conf)
                if conf > best[deg][0]:
                    best[deg] = (conf, r.copy(), tuple(int(v) for v in box))
            else:
                stats[deg].append(0.0)
    print(f"frames sampled: {n // 2}")
    winner, wscore = None, 0.0
    for deg in ROTS:
        s = stats[deg]
        hits = sum(1 for c in s if c > 0)
        mean = (sum(c for c in s if c > 0) / hits) if hits else 0.0
        conf, img, box = best[deg]
        print(f"rot {deg:3d}: hits {hits}/{len(s)}  mean conf {mean:.2f}  best {conf:.2f}  box {box}")
        if hits * mean > wscore:
            winner, wscore = deg, hits * mean
        if img is not None:
            x, y, bw, bh = box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.imwrite(str(OUT / f"rot_{deg}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if winner is None:
        print("RESULT: no face found at any rotation — is someone in front of the camera, with light on the face?")
        return 2
    advice = {0: "config is correct.",
              180: "image is upside-down -> toggle BOTH camera_hflip and camera_vflip in config/config.yaml.",
              90: "image is sideways -> re-mount the camera module.",
              270: "image is sideways -> re-mount the camera module."}[winner]
    print(f"RESULT: faces are upright at rotation {winner} -> {advice}  (images in {OUT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
