#!/usr/bin/env python3
"""One-shot face pipeline diagnostic on a LIVE frame.

Grabs one frame (dashboard MJPEG stream on :5000 while the service runs, or the
camera directly when it is stopped), saves it, and reports everything that
decides whether Stella sees and knows you:

  * image stats: size, brightness, colour cast (NoIR pink), focus measure
  * YuNet detections (the service's detector) with the face-size distance estimate
  * dlib HOG detections (full frame, 640-wide, rotated 180) for comparison
  * the shared encoder (encode_face_from_frame) on each YuNet box and its
    distance to EVERY person in data/faces/face_db.pkl (match if < 0.60,
    confident if < 0.50)
  * face DB summary (who, how many samples, which camera they were enrolled with)

    cd ~/AIRobot_v2.0 && venv/bin/python tools/face_diag.py

Outputs: /tmp/stella_diag/frame.jpg and /tmp/stella_diag/face.jpg.
Typical reads: 'YuNet 0 faces' with a person present -> orientation / light
(run tools/camera_orientation_check.py); distances 0.5-0.6 for the master ->
re-enrol with tools/reenroll.py; '0.3 m' with a ~250 px face is normal.
"""
import sys
import time
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import face_recognition as fr  # noqa: E402
from modules.vision.face_recognition import encode_face_from_frame, estimate_distance_m  # noqa: E402

YUNET = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
DB = ROOT / "data" / "faces" / "face_db.pkl"
OUT = Path("/tmp/stella_diag")
DASHBOARD = "http://127.0.0.1:5000/video_feed"


def grab_dashboard(timeout=12, min_w=640):
    req = urllib.request.urlopen(DASHBOARD, timeout=5)
    buf = b""
    t0 = time.time()
    while time.time() - t0 < timeout:
        chunk = req.read(8192)
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
            if img is not None and img.shape[1] >= min_w:
                return img
    return None


def grab_camera():
    from parts_used.camera_usb import CameraManager
    cm = CameraManager()
    if not cm.start():
        raise RuntimeError("CameraManager.start() failed (is the airobot service holding the camera?)")
    try:
        time.sleep(1.5)
        for _ in range(30):
            f = cm.get_latest_frame()
            if f is not None:
                return f.image
            time.sleep(0.1)
    finally:
        cm.stop()
    return None


def grab():
    try:
        urllib.request.urlopen(DASHBOARD, timeout=3).close()
        print("source: dashboard stream (service running)", flush=True)
        return grab_dashboard()
    except Exception:
        print("source: camera directly (service stopped)", flush=True)
        return grab_camera()


def hog(img_bgr, up):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = time.time()
    locs = fr.face_locations(rgb, number_of_times_to_upsample=up, model="hog")
    return locs, time.time() - t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = grab()
    if frame is None:
        print("no frame")
        return 1
    cv2.imwrite(str(OUT / "frame.jpg"), frame)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"frame {w}x{h} dtype={frame.dtype} contiguous={frame.flags['C_CONTIGUOUS']}")
    print(f"brightness mean={gray.mean():.1f} std={gray.std():.1f}  (ideal mean 80-160)")
    b, g, r = (float(frame[..., i].mean()) for i in range(3))
    print(f"channel means B={b:.1f} G={g:.1f} R={r:.1f}  (R>>G,B = NoIR pink cast)")
    print(f"focus (laplacian var) full={cv2.Laplacian(gray, cv2.CV_64F).var():.1f}  (<50 = blurry)")

    boxes = []
    if YUNET.exists():
        yn = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.7, 0.3, 50)
        t = time.time()
        _, dets = yn.detect(frame)
        if dets is not None:
            for d in dets:
                x, y, bw, bh = (int(v) for v in d[:4])
                boxes.append((max(0, x), max(0, y), bw, bh, float(d[-1])))
        print(f"YuNet: {len(boxes)} face(s) in {time.time() - t:.2f}s -> "
              + ", ".join(f"x={x} y={y} w={bw} h={bh} conf={c:.2f} ~{estimate_distance_m(bw, w):.2f} m"
                          for x, y, bw, bh, c in boxes))
    else:
        print(f"YuNet model missing: {YUNET}")

    for label, img in (("full", frame),
                       ("640-wide", cv2.resize(frame, None, fx=640.0 / w, fy=640.0 / w)),
                       ("full rotated 180", cv2.rotate(frame, cv2.ROTATE_180))):
        for up in (0, 1):
            locs, dt = hog(img, up)
            print(f"dlib HOG {label} up={up}: {len(locs)} face(s) in {dt:.2f}s")

    db = {}
    if DB.exists():
        try:
            db = pickle.load(open(DB, "rb"))
        except Exception as exc:
            print(f"face DB unreadable: {exc}")
    for pid, rec in db.items():
        if not isinstance(rec, dict):
            continue
        md = rec.get("metadata", {}) or {}
        print(f"DB: {pid} name={rec.get('name')} samples={len(rec.get('embeddings', []))} "
              f"master={rec.get('is_master')} enrolled_via={md.get('enrolled_via')} "
              f"camera={md.get('camera')} face_px={md.get('face_px') or md.get('face_px_min_max')}")
    for i, (x, y, bw, bh, c) in enumerate(boxes):
        if i == 0:
            pad = int(bw * 0.6)
            cv2.imwrite(str(OUT / "face.jpg"), frame[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad])
        t = time.time()
        enc = encode_face_from_frame(frame, (x, y, bw, bh), 200)
        print(f"box{i}: encoder -> {'OK' if enc is not None else 'None'} in {time.time() - t:.2f}s")
        if enc is None:
            continue
        enc = np.asarray(enc)
        for pid, rec in db.items():
            embs = [np.asarray(e) for e in (rec.get("embeddings", []) if isinstance(rec, dict) else [])]
            if not embs:
                continue
            ds = sorted(float(np.linalg.norm(enc - e)) for e in embs)
            print(f"box{i}: distance to {rec.get('name')} ({pid}): min={ds[0]:.3f} "
                  f"median={ds[len(ds) // 2]:.3f}  (match < 0.60, confident < 0.50)")
    print(f"images: {OUT / 'frame.jpg'}, {OUT / 'face.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
