# Bug #004 — Face recognition passed a cropped face to dlib instead of full frame + location

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** vision
- **Files touched:** `modules/vision/face_recognition.py`
- **Commit(s):** e5d42bd

## Symptom
Faces were detected but never recognised (no match for the enrolled master), or encodings failed.

## Root cause
The code cropped the face out of the frame and handed the crop to `face_recognition.face_encodings`, which then re-detected inside a tiny crop and produced poor/no encodings. dlib expects the full frame plus the known face location.

## Fix
Pass the full frame and the detected face location (`known_face_locations`) to dlib so encodings are computed on the original image.

## How to verify
Stand in front of the camera after enrolling; the journal should show `Face detected: Moti` with a similarity score rather than `Unknown`.

## Will it come back?
No. (A later, separate recognition problem — threshold too strict — is bug #014.)
