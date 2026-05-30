"""
Master Face Enrollment Script
=============================
Opens your webcam, captures multiple angles of your face, and saves you
as the master / owner of the robot.  Only the master can issue physical
commands (lights, motors, servos, etc.).

Uses OpenCV only — no dlib required.  Face embeddings are computed on the
Raspberry Pi when the robot starts (where dlib installs easily via apt).

Usage:
    python enroll_master.py              # interactive — press SPACE to capture
    python enroll_master.py --auto       # auto-capture 10 frames over 5 seconds
    python enroll_master.py --name Moti  # set the master's display name
"""

import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import face_recognition for embeddings (optional on Windows)
try:
    import face_recognition as fr
    HAS_FR = True
except ImportError:
    HAS_FR = False

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "faces"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "face_db.pkl"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MASTER_ID = os.getenv("MASTER_USER_ID", "master_001")
MIN_SAMPLES = 5
DEFAULT_SAMPLES = 10

# OpenCV's built-in Haar cascade — ships with opencv-python, no extras needed
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_camera() -> int:
    """Find the first working camera index."""
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return idx
    return 0


def enroll(name: str, samples: int, auto: bool, camera: int = None) -> bool:
    cam_idx = camera if camera is not None else detect_camera()
    print(f"  Using camera index: {cam_idx}")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    if face_cascade.empty():
        print("ERROR: Could not load Haar cascade.")
        return False

    collected_frames = []
    collected_embeddings = []

    print(f"\n{'='*50}")
    print(f"  MASTER FACE ENROLLMENT — {name}")
    print(f"{'='*50}")
    if HAS_FR:
        print("  face_recognition available — embeddings will be computed now.")
    else:
        print("  face_recognition not available (dlib not built).")
        print("  Images will be saved; embeddings computed on the Pi.\n")

    if auto:
        print(f"  Auto-capturing {samples} frames in 3 seconds...")
        print(f"  Look straight at the camera.\n")
        time.sleep(3)
    else:
        print(f"  Press SPACE to capture (need {samples} shots)")
        print(f"  Press Q to finish early")
        print(f"  Try different angles: front, slight left, slight right\n")

    last_auto_time = 0

    while len(collected_frames) < samples:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Status text
        status = f"Captured: {len(collected_frames)}/{samples}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if len(faces) == 0:
            cv2.putText(display, "No face detected - adjust position", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Master Enrollment", display)

        capture_now = False

        if auto:
            if len(faces) > 0 and time.time() - last_auto_time > 0.5:
                capture_now = True
                last_auto_time = time.time()
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and len(faces) > 0:
                capture_now = True
            elif key == ord('q'):
                break

        if capture_now and len(faces) > 0:
            collected_frames.append(frame.copy())

            # Compute embedding if library is available
            if HAS_FR:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = fr.face_encodings(rgb)
                if encodings:
                    collected_embeddings.append(encodings[0])

            print(f"  [+] Captured sample {len(collected_frames)}/{samples}")

        if auto:
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if len(collected_frames) < MIN_SAMPLES:
        print(f"\nERROR: Only got {len(collected_frames)} samples (need {MIN_SAMPLES}).")
        return False

    # Build embeddings list (may be empty if dlib not available)
    emb_list = [emb.tolist() for emb in collected_embeddings] if collected_embeddings else []

    # Build face entry compatible with FaceRecognitionModule
    face_data = {
        MASTER_ID: {
            'id': MASTER_ID,
            'name': name,
            'embeddings': emb_list,
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'interaction_count': 0,
            'is_master': True,
            'permissions': ['all'],
            'metadata': {
                'enrolled_via': 'enroll_master.py',
                'samples': len(collected_frames),
                'enrolled_at': datetime.now().isoformat(),
                'needs_encoding': len(emb_list) == 0,
            }
        }
    }

    # Merge with existing DB if present
    if DB_PATH.exists():
        try:
            with open(DB_PATH, 'rb') as f:
                existing = pickle.load(f)
            # Remove any old master entries
            existing = {k: v for k, v in existing.items()
                        if not v.get('is_master', False)}
            existing.update(face_data)
            face_data = existing
        except Exception as e:
            print(f"  Warning: could not read existing DB ({e}), overwriting.")

    # Save DB
    with open(DB_PATH, 'wb') as f:
        pickle.dump(face_data, f)

    # Save sample images
    for i, img in enumerate(collected_frames[:5]):
        img_path = IMAGES_DIR / f"{MASTER_ID}_{i}.jpg"
        cv2.imwrite(str(img_path), img)

    print(f"\n{'='*50}")
    print(f"  SUCCESS! Master face enrolled.")
    print(f"  ID   : {MASTER_ID}")
    print(f"  Name : {name}")
    print(f"  Shots: {len(collected_frames)}")
    if emb_list:
        print(f"  Embeddings: {len(emb_list)} computed")
    else:
        print(f"  Embeddings: will be computed on the Pi at first boot")
    print(f"  DB   : {DB_PATH}")
    print(f"{'='*50}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Enroll master face for Gonzo robot")
    parser.add_argument("--name", default="Master", help="Your display name")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Number of face samples (min {MIN_SAMPLES})")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-capture without pressing SPACE")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (0, 1, 2, …). Auto-detected if omitted.")
    args = parser.parse_args()

    enroll(args.name, max(args.samples, MIN_SAMPLES), args.auto, args.camera)


if __name__ == "__main__":
    main()
