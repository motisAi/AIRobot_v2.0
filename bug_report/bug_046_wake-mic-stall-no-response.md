# Bug #046 — "No response to Hey Stella" after a while (wake mic stream went silent)

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/wake_word.py`, `config/config.yaml`
- **Commit(s):** see 2026-09-19 wake-mic commit

## Symptom
She recognised Moti and answered normally, then after some minutes "Hey Stella" got no response at all. Face detection kept working the whole time.

## Root cause
The wake-word listener was opening the microphone built into the USB "Auto Focus Camera" (via the ambiguous PortAudio "sysdefault" device). That mic shares one USB device with the camera video; under continuous camera+face-recognition load the audio stream went silent while still "open" — no exception, no kernel USB error, just zero audio. The wake loop only recovered from a failure to OPEN the stream, not from a stream that goes silent while open, so it never reopened. Vosk produced no results for minutes (log: last "wake heard" then silence while faces kept detecting).

## Fix
1. `wake_word.py`: self-heal the read loop — reopen the stream after repeated read errors, and detect a dead stream (long run of exact-zero frames; a live mic never returns perfect digital silence) and reopen it (~20 s of zeros).
2. `config.yaml`: move the wake mic off the camera's shared mic to the dedicated **USB PnP Sound Device** (the reliable command mic). Wake pauses during a conversation, so sharing one good mic is fine and removes the camera contention + the "sysdefault" ambiguity.

## How to verify
`journalctl -u airobot | grep "wake heard"` keeps producing lines over time; "Hey Stella" responds after long idle periods and after heavy camera use. If a stream ever dies, the log shows "Wake mic went silent (dead stream) — reopening".

## Will it come back?
The self-heal recovers from any silent/failed stream regardless of cause. The durable hardware fix is dedicated I2S mics (INMP441) instead of USB audio; planned. Related audio-flakiness: bug_002, bug_009, bug_010, bug_037.
