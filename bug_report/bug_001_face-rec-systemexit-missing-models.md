# Bug #001 — face_recognition's quit() killed the process when dlib models were missing

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** vision
- **Files touched:** `modules/vision/face_recognition.py`
- **Commit(s):** 9985a8c

## Symptom
On the fresh RPi5 install the whole robot exited during startup as soon as the face-recognition module was initialised.

## Root cause
The `face_recognition` library calls `quit()` when its dlib model files are not found. `quit()` raises `SystemExit`, which the module init did not catch, so the exception propagated and terminated the main process.

## Fix
Catch `SystemExit` (in addition to ordinary exceptions) around the model-loading call so face recognition degrades to disabled instead of exiting. A helper `fix_face_models.py` was added in the following commit (a22333e) to install the models.

## How to verify
Start the service with the dlib model files absent — the robot must keep running with face recognition disabled and a logged warning, not exit.

## Will it come back?
No, unless the try/except in `face_recognition.py` init is removed. Missing model files on a new machine now only log a warning.
