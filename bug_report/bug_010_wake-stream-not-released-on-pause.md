# Bug #010 — Wake-word PyAudio stream not closed on pause, so the speech mic could not open

- **Date found:** 2026-05-31
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/wake_word.py`
- **Commit(s):** 5012188

## Symptom
Wake word fired, but the command capture immediately failed to open the microphone (device busy) — Stella woke up and then heard nothing.

## Root cause
Pausing the wake-word listener only stopped its loop; the PyAudio input stream stayed open and held the single USB mic, so `speech_recognition` could not open the same device.

## Fix
On pause, the wake-word listener closes its PyAudio stream and signals that it is closed; it reopens the stream on resume. (A later refinement, commit 231740d, sets `_stream_closed_event` in the paused branch to remove a 2 s stall.)

## How to verify
Journal sequence on wake: "Pausing wake-word listener — releasing mic" -> command captured -> "listener resumed", with no "device busy" error in between.

## Will it come back?
No, as long as pause/resume remains the only place that touches the stream — see bug #028 for what happens when another thread closes it.
