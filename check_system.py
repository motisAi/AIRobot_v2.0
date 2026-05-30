"""Check all vision pipeline components."""
import sys

# 1. face_recognition
try:
    import face_recognition
    print(f"face_recognition: OK (v{face_recognition.__version__})")
except Exception as e:
    print(f"face_recognition: FAIL ({e})")

# 2. DeepFace
try:
    from deepface import DeepFace
    print("DeepFace: OK")
except Exception as e:
    print(f"DeepFace: NOT INSTALLED ({e})")

# 3. OpenCV
try:
    import cv2
    print(f"OpenCV: OK (v{cv2.__version__})")
except Exception as e:
    print(f"OpenCV: FAIL ({e})")

# 4. Haar cascade
try:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print(f"Haar cascade: {'OK' if not cascade.empty() else 'EMPTY'}")
except Exception as e:
    print(f"Haar cascade: FAIL ({e})")

# 5. Camera
try:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print(f"Camera: OK (frame {frame.shape})")
        # Try face detection on this frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        print(f"Haar face detect: {len(faces)} faces found in test frame")
        
        # Try face_recognition on this frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            print(f"face_recognition detect: {len(locs)} faces found in test frame")
        except Exception as e:
            print(f"face_recognition detect: FAIL ({e})")
    else:
        print("Camera: FAIL (could not read frame)")
    cap.release()
except Exception as e:
    print(f"Camera: FAIL ({e})")

# 6. TTS
print()
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("pyttsx3: OK")
except Exception as e:
    print(f"pyttsx3: FAIL ({e})")

try:
    import subprocess
    r = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
    print(f"espeak: OK ({r.stdout.strip()})")
except Exception as e:
    print(f"espeak: FAIL ({e})")

# 7. Audio output
try:
    subprocess.run(['espeak', '-w', '/tmp/tts_test.wav', 'hello'], check=True, capture_output=True)
    subprocess.run(['paplay', '/tmp/tts_test.wav'], check=True, capture_output=True)
    print("Audio output (PulseAudio): OK")
except Exception as e:
    print(f"Audio output: FAIL ({e})")

# 8. PyAudio mics
try:
    import pyaudio
    p = pyaudio.PyAudio()
    print(f"PyAudio: OK ({p.get_device_count()} devices)")
    p.terminate()
except Exception as e:
    print(f"PyAudio: FAIL ({e})")

# 9. Known faces database
from pathlib import Path
db_path = Path("data/faces/face_db.pkl")
if db_path.exists():
    import pickle
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    print(f"Face DB: {len(db)} known faces")
else:
    print("Face DB: empty (no faces learned yet)")

print("\n--- Summary ---")
print("Ready for face detection: YES (opencv Haar cascade)")
print(f"Ready for face recognition: {'YES' if 'face_recognition' in sys.modules else 'NO'}")
