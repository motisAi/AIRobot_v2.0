#!/usr/bin/env python3
"""Audio-guided face (re-)enrollment — master by default, anyone with --name/--id.

Speaks instructions through Stella's speaker so you can face the camera without
reading the screen ("look at me", "come closer", "all done"). Run with the
airobot service STOPPED (the camera + speaker must be free):

    sudo systemctl stop airobot
    cd ~/AIRobot_v2.0 && venv/bin/python tools/reenroll.py            # Moti / master_001
    venv/bin/python tools/reenroll.py --name Orr --id person_1786305575 --no-master
    sudo systemctl start airobot

Frames come from the SAME CameraManager the service uses (CSI Camera Module 3
via picamera2, USB webcam as fallback), and encodings go through the SAME
pipeline the service recognises with (encode_face_from_frame: padded crop,
small-face upscaling for the 120° lens, dlib detect/align/encode). That parity
is what keeps runtime match distances low. Keeps all other faces; replaces only
the given id. Ends by SPEAKING whether recognition is strong.
"""
import sys
import time
import pickle
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import face_recognition as fr

ROOT = Path(__file__).resolve().parent.parent  # repo root (tool lives in tools/)
sys.path.insert(0, str(ROOT))
from modules.vision.face_recognition import encode_face_from_frame  # noqa: E402

NAME = "Moti"
MASTER_ID = "master_001"
TARGET = 14
MIN_OK = 5
MAX_SECONDS = 50
THRESH = 0.60
TARGET_FACE_PX = 200
DB_PATH = ROOT / "data" / "faces" / "face_db.pkl"
IMAGES_DIR = ROOT / "data" / "faces" / "images"
# I2S speaker (voicehat) first, then pulse/default, then HDMI.
AUDIO_DEVS = ["plughw:CARD=sndrpigooglevoi", "pulse", "default", "sysdefault:CARD=vc4hdmi0"]


def say(text: str):
    """Speak a line out loud (blocks). Tries espeak-ng/espeak on each audio device."""
    print(">>", text, flush=True)
    for tool in ("espeak-ng", "espeak"):
        for dev in AUDIO_DEVS:
            try:
                r = subprocess.run(
                    tool + " -s 150 -a 200 " + shlex.quote(text) +
                    " --stdout | aplay -q -D " + shlex.quote(dev),
                    shell=True, timeout=12)
                if r.returncode == 0:
                    return
            except Exception:
                continue


class FrameSource:
    """Grab BGR frames from the project's CameraManager (CSI picamera2 or USB),
    falling back to a bare cv2.VideoCapture if the manager cannot start."""

    def __init__(self):
        self.cm = None
        self.cap = None
        try:
            from parts_used.camera_usb import CameraManager
            cm = CameraManager()
            if cm.start():
                self.cm = cm
                time.sleep(1.0)  # let the first frames arrive
                print(f"camera: CameraManager mode={cm._mode} res={cm.resolution}", flush=True)
                return
        except Exception as exc:
            print(f"CameraManager unavailable ({exc}); trying cv2", flush=True)
        for i in (0, 1, 2, 3):
            c = cv2.VideoCapture(i)
            ok = c.isOpened() and c.read()[0]
            if ok:
                self.cap = c
                print(f"camera: cv2 index {i}", flush=True)
                return
            c.release()

    @property
    def ok(self):
        return self.cm is not None or self.cap is not None

    def read(self):
        if self.cm is not None:
            f = self.cm.get_latest_frame()
            return None if f is None else f.image
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def close(self):
        try:
            if self.cm is not None:
                self.cm.stop()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


YUNET_MODEL = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
_yunet = None  # (detector, (w, h)) — created lazily, re-created if the frame size changes


def biggest_face_box(frame_bgr):
    """Biggest face as (x, y, w, h) using the SAME detector the service runs
    (YuNet DNN, score >= 0.7). dlib's HOG detector missed faces on backlit /
    steep-angle frames where YuNet did not, so enrolment and runtime now agree
    on *where* the face is as well as how it is encoded. Falls back to dlib HOG
    on a 640-wide copy only if the model file is missing."""
    global _yunet
    h, w = frame_bgr.shape[:2]
    if YUNET_MODEL.exists():
        if _yunet is None or _yunet[1] != (w, h):
            _yunet = (cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), 0.7, 0.3, 50), (w, h))
        _, dets = _yunet[0].detect(frame_bgr)
        if dets is None or len(dets) == 0:
            return None
        d = max(dets, key=lambda r: float(r[2]) * float(r[3]))
        x, y, bw, bh = (int(v) for v in d[:4])
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        return (x, y, bw, bh) if bw > 0 and bh > 0 else None
    # ---- fallback: dlib HOG on a downscaled copy, mapped back to full-res ----
    scale = 640.0 / max(w, 1)
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1.0 else frame_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = fr.face_locations(rgb, model="hog")
    if not locs:
        return None
    top, right, bot, left = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
    s = 1.0 / scale if scale < 1.0 else 1.0
    return (int(left * s), int(top * s), int((right - left) * s), int((bot - top) * s))


def biggest_encoding(frame_bgr):
    box = biggest_face_box(frame_bgr)
    if box is None:
        return None, None
    return encode_face_from_frame(frame_bgr, box, TARGET_FACE_PX), box


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Enrol/re-enrol one face into the shared DB (keeps everyone else).")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--id", dest="pid", default=MASTER_ID)
    ap.add_argument("--no-master", dest="master", action="store_false")
    ap.set_defaults(master=True)
    a = ap.parse_args()
    name, target_id, is_master = a.name, a.pid, a.master
    print(f"Enrolling: name={name} id={target_id} master={is_master}", flush=True)
    src = FrameSource()
    if not src.ok:
        say("I cannot open my camera.")
        return 1

    say("Let's set up face recognition. Please look straight at my camera.")
    # Mix of distances on purpose: the wide lens makes faces small from afar,
    # so we want samples both close up and from where you usually stand.
    prompts = [(0, "Look straight at me."),
               (3, "Great. Please come a little closer, keep looking at me."),
               (6, "Now step back to about one and a half meters, still facing me."),
               (9, "Now look at me from where you usually stand."),
               (12, "Almost done. Look right at me and hold still.")]
    spoken = set()
    embs, frames, widths = [], [], []
    t0 = time.time()
    last = 0.0
    n_frames = n_faces = 0
    next_report = 5.0
    while len(embs) < TARGET and (time.time() - t0) < MAX_SECONDS:
        for n, txt in prompts:
            if len(embs) >= n and n not in spoken:
                spoken.add(n)
                say(txt)
        frame = src.read()
        if frame is None:
            time.sleep(0.05)
            continue
        n_frames += 1
        now = time.time()
        if now - t0 >= next_report:   # progress so a silent failure is diagnosable
            print(f"t={now - t0:4.1f}s frames={n_frames} faces_seen={n_faces} "
                  f"captured={len(embs)} frame={frame.shape[1]}x{frame.shape[0]}", flush=True)
            next_report += 5.0
        if now - last < 0.6:
            time.sleep(0.05)
            continue
        enc, box = biggest_encoding(frame)
        if box is not None:
            n_faces += 1
        if enc is not None:
            embs.append(np.asarray(enc))
            frames.append(frame.copy())
            widths.append(box[2])
            last = now
            print(f"captured {len(embs)}/{TARGET}  (face {box[2]}px wide)", flush=True)

    if len(embs) < MIN_OK:
        src.close()
        print(f"FAILED: frames={n_frames} faces_seen={n_faces} captured={len(embs)} (need {MIN_OK})", flush=True)
        if n_faces == 0:
            say("Sorry, I could not find a face at all. Please check that my camera is upright and pointing at you.")
        else:
            say("Sorry, I could not see your face clearly. Please try again with more light on your face.")
        return 2

    # Drop only ISOLATED embeddings (no neighbour within 0.45): a side/tilt shot
    # that encodes poorly and could sit near another person. Keep every real
    # cluster: close-up and far samples of the same face legitimately sit >0.45
    # apart in dlib space (far faces are upscaled), and the old "medoid core"
    # rule dropped the whole close-up cluster, which made her weak at 0.4 m.
    if len(embs) > MIN_OK:
        P = np.stack(embs)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        keep = [i for i in range(len(embs)) if nn[i] <= 0.45]
        if MIN_OK <= len(keep) < len(embs):
            dropped = [(widths[i], round(float(nn[i]), 2)) for i in range(len(embs)) if i not in keep]
            print(f"kept {len(keep)}/{len(embs)} embeddings; dropped isolated (px, nearest): {dropped}", flush=True)
            embs = [embs[i] for i in keep]
            frames = [frames[i] for i in keep]
            widths = [widths[i] for i in keep]
    print(f"samples kept: {len(embs)}  face px: {sorted(widths)}", flush=True)

    now_iso = datetime.now().isoformat()
    entry = {target_id: {
        "id": target_id, "name": name,
        "embeddings": [e.tolist() for e in embs],
        "first_seen": now_iso, "last_seen": now_iso,
        "interaction_count": 0, "is_master": is_master,
        "permissions": ["all"] if is_master else ["basic"],
        "metadata": {"enrolled_via": "reenroll.py", "samples": len(embs),
                     "enrolled_at": now_iso, "needs_encoding": False,
                     "camera": "picamera2" if src.cm is not None else "cv2",
                     "face_px_min_max": [min(widths), max(widths)] if widths else None,
                     "face_px": sorted(widths)},
    }}
    db = {}
    if DB_PATH.exists():
        try:
            db = pickle.load(open(DB_PATH, "rb"))
            db = {k: v for k, v in db.items() if k != target_id}   # replace only this person; keep everyone else
        except Exception:
            db = {}
    db.update(entry)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(db, open(DB_PATH, "wb"))
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames[:5]):
            cv2.imwrite(str(IMAGES_DIR / f"{target_id}_{i}.jpg"), img)
    except Exception:
        pass

    # ---- speak a self-verification (same pipeline as runtime) ----
    say("I saved your face. Let me test it. Please keep looking at me.")
    known = embs
    best = 9.0
    t1 = time.time()
    while time.time() - t1 < 8:
        frame = src.read()
        if frame is None:
            time.sleep(0.05)
            continue
        enc, _ = biggest_encoding(frame)
        if enc is not None:
            d = min(float(np.linalg.norm(np.asarray(enc) - k)) for k in known)
            best = min(best, d)
    src.close()
    print(f"self-verify best distance: {best:.3f} (lower=better; match if < {THRESH})", flush=True)
    if best < 0.45:
        say("Excellent. I recognize you clearly now. All done.")
    elif best < THRESH:
        say("Good. I can recognize you now. All done.")
    else:
        say("Recognition is still a bit weak, but I saved it. We can try again in better light.")
    return 0


sys.exit(main())
