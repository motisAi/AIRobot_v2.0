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


def biggest_face_box(frame_bgr):
    """dlib HOG on a downscaled copy (fast), mapped back to full-res (x, y, w, h)."""
    h, w = frame_bgr.shape[:2]
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
    while len(embs) < TARGET and (time.time() - t0) < MAX_SECONDS:
        for n, txt in prompts:
            if len(embs) >= n and n not in spoken:
                spoken.add(n)
                say(txt)
        frame = src.read()
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        if now - last < 0.6:
            time.sleep(0.05)
            continue
        enc, box = biggest_encoding(frame)
        if enc is not None:
            embs.append(np.asarray(enc))
            frames.append(frame.copy())
            widths.append(box[2])
            last = now
            print(f"captured {len(embs)}/{TARGET}  (face {box[2]}px wide)", flush=True)

    if len(embs) < MIN_OK:
        src.close()
        say("Sorry, I could not see your face clearly. Please try again with more light on your face.")
        return 2

    # Drop outlier embeddings: keep the coherent core near the medoid so one bad
    # frame (a side/tilt shot that encodes poorly) can't sit near another person
    # and cause mix-ups. Never prune below MIN_OK.
    if len(embs) >= 5:
        P = embs
        best, mi = 1e9, 0
        for i, x in enumerate(P):
            tot = sum(np.linalg.norm(x - y) for y in P)
            if tot < best:
                best, mi = tot, i
        m = P[mi]
        keep = [i for i, x in enumerate(P) if np.linalg.norm(x - m) <= 0.45]
        if len(keep) >= MIN_OK and len(keep) < len(embs):
            print(f"kept {len(keep)}/{len(embs)} coherent embeddings (dropped outliers)", flush=True)
            embs = [embs[i] for i in keep]
            frames = [frames[i] for i in keep]
            widths = [widths[i] for i in keep]

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
                     "face_px_min_max": [min(widths), max(widths)] if widths else None},
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
