# Bug #048 — Recognition similarity collapsed after switching to YuNet detector

- **Date found:** 2026-09-19
- **Status:** fixed-needs-verify (re-enroll pending)
- **Area:** vision
- **Files touched:** `modules/vision/face_recognition.py`
- **Commit(s):** 57ad402

## Symptom
After the Haar->YuNet detector change, Moti's face read sim 0.00-0.52 (threshold 0.60) and often "unknown"; recognition became unreliable.

## Root cause
YuNet returns a TIGHT face box. The enrolled embeddings were built from looser dlib-HOG (Moti) / Haar (Orr) crops. Feeding the tight box to dlib `face_encodings` shifts the landmark chip, so the 128-D descriptor drifts and euclidean distance sits on the 0.60 edge.

## Fix
Before encoding, expand the YuNet box toward HOG geometry (top -35%h, bottom +15%h, sides +20%w, clamped) at BOTH the recognize and enroll sites, so live crops match enrolled ones. Partial recovery in code; FULL fix = re-enroll under YuNet (operator step, camera needed).

## How to verify
Live sim for Moti rises above ~0.55 on frontal frames; after re-enroll it should sit ~0.65+.

## Will it come back?
Recognition stays marginal until re-enroll aligns enrolled geometry with YuNet. Durable path: dedicated I2S mics + better camera (planned).
