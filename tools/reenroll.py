#!/usr/bin/env python3
"""Audio-guided master re-enrollment.

Speaks instructions through Stella's speaker so you can face the camera without
reading the screen ("look at me", "turn left/right", "all done"). Uses dlib/HOG
detection + a hard time cap (never hangs). Run with the airobot service STOPPED
(camera + speaker are free). Keeps all non-master faces; replaces the master.
Ends by SPEAKING whether recognition is strong.
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

NAME = "Moti"
MASTER_ID = "master_001"
TARGET = 12
MIN_OK = 4
MAX_SECONDS = 40
THRESH = 0.60
ROOT = Path(__file__).resolve().parent.parent  # repo root (tool lives in tools/)
DB_PATH = ROOT / "data" / "faces" / "face_db.pkl"
IMAGES_DIR = ROOT / "data" / "faces" / "images"
# "pulse" first: pulseaudio owns the HDMI device, so route through it (direct
# hardware access returns "Device or resource busy").
AUDIO_DEVS = ["plughw:CARD=sndrpigooglevoi", "pulse", "default", "sysdefault:CARD=vc4hdmi0"]  # I2S speaker first


def say(text: str):
    """Speak a line out loud (blocks). Tries espeak-ng/espeak to the HDMI device."""
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


def find_cam():
    for i in (0, 1, 2, 3):
        c = cv2.VideoCapture(i)
        ok = False
        if c.isOpened():
            ok, _ = c.read()
        c.release()
        if ok:
            return i
    return 0


def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def biggest_encoding(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = fr.face_locations(rgb, model="hog")
    if not locs:
        return None
    loc = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
    enc = fr.face_encodings(rgb, [loc])
    return enc[0] if enc else None


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
    idx = find_cam()
    cap = open_cam(idx)
    if not cap.isOpened():
        say("I cannot open my camera.")
        return 1

    say("Let's set up face recognition. Please look straight at my camera.")
    prompts = [(0, "Look straight at me."),
               (3, "Great. Please come a little closer, keep looking at me."),
               (6, "Now lean back a bit, still facing me."),
               (9, "Now look at me from where you usually stand."),
               (11, "Almost done. Look right at me and hold still.")]
    spoken = set()
    embs, frames = [], []
    t0 = time.time()
    last = 0.0
    while len(embs) < TARGET and (time.time() - t0) < MAX_SECONDS:
        for n, txt in prompts:
            if len(embs) >= n and n not in spoken:
                spoken.add(n)
                say(txt)
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        now = time.time()
        if now - last < 0.6:
            continue
        enc = biggest_encoding(frame)
        if enc is not None:
            embs.append(enc)
            frames.append(frame.copy())
            last = now
            print(f"captured {len(embs)}/{TARGET}", flush=True)
    cap.release()

    if len(embs) < MIN_OK:
        say("Sorry, I could not see your face clearly. Please try again with more light on your face.")
        return 2

    # Drop outlier embeddings: keep the coherent core near the medoid so one bad
    # frame (a side/tilt shot that encodes poorly) can't sit near another person
    # and cause mix-ups. Never prune below MIN_OK.
    if len(embs) >= 5:
        P = [np.array(e) for e in embs]
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

    now_iso = datetime.now().isoformat()
    entry = {target_id: {
        "id": target_id, "name": name,
        "embeddings": [e.tolist() for e in embs],
        "first_seen": now_iso, "last_seen": now_iso,
        "interaction_count": 0, "is_master": is_master,
        "permissions": ["all"] if is_master else ["basic"],
        "metadata": {"enrolled_via": "reenroll.py", "samples": len(embs),
                     "enrolled_at": now_iso, "needs_encoding": False},
    }}
    db = {}
    if DB_PATH.exists():
        try:
            db = pickle.load(open(DB_PATH, "rb"))
            db = {k: v for k, v in db.items() if k != target_id}   # replace only this person; keep everyone else
        except Exception:
            db = {}
    db.update(entry)
    pickle.dump(db, open(DB_PATH, "wb"))
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames[:5]):
            cv2.imwrite(str(IMAGES_DIR / f"{target_id}_{i}.jpg"), img)
    except Exception:
        pass

    # ---- speak a self-verification ----
    say("I saved your face. Let me test it. Please keep looking at me.")
    known = [np.array(e) for e in embs]
    best = 9.0
    cap = open_cam(idx)
    t1 = time.time()
    while time.time() - t1 < 8:
        ok, frame = cap.read()
        if not ok:
            continue
        enc = biggest_encoding(frame)
        if enc is not None:
            d = min(float(np.linalg.norm(enc - k)) for k in known)
            best = min(best, d)
    cap.release()
    print(f"self-verify best distance: {best:.3f} (lower=better; match if < {THRESH})", flush=True)
    if best < 0.45:
        say("Excellent. I recognize you clearly now. All done.")
    elif best < THRESH:
        say("Good. I can recognize you now. All done.")
    else:
        say("Recognition is still a bit weak, but I saved it. We can try again in better light.")
    return 0


sys.exit(main())
