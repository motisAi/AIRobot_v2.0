# Bug #046 — "No response to Hey Stella" after a while (wake mic stream went silent)

- **Date found:** 2026-09-19
- **Status:** fixed (fix step 1 stands; fix step 2 — wake mic on the USB dongle — superseded)
- **Area:** audio
- **Files touched:** `modules/audio/wake_word.py`, `config/config.yaml`
- **Commit(s):** 743ef31 (2026-09-19 wake-mic commit)
- **Superseded by:** bug_053 (wake mic moved back to the camera mic, d3b2577)
- **Reverted:** fix step 2 (wake on "USB PnP Sound Device") reverted by d3b2577 on 2026-09-26; the current layout is wake = camera mic, command = USB PnP — see `docs/architecture/stella-architecture.md` §4

## Symptom
She recognised Moti and answered normally, then after some minutes "Hey Stella" got no response at all. Face detection kept working the whole time.

## Root cause
The wake-word listener was opening the microphone built into the USB "Auto Focus Camera" (via the ambiguous PortAudio "sysdefault" device). That mic shares one USB device with the camera video; under continuous camera+face-recognition load the audio stream went silent while still "open" — no exception, no kernel USB error, just zero audio. The wake loop only recovered from a failure to OPEN the stream, not from a stream that goes silent while open, so it never reopened. Vosk produced no results for minutes (log: last "wake heard" then silence while faces kept detecting).

## Fix
1. `wake_word.py`: self-heal the read loop — reopen the stream after repeated read errors, and detect a dead stream (long run of exact-zero frames; a live mic never returns perfect digital silence) and reopen it (~20 s of zeros). **This part stands.**
2. `config.yaml`: move the wake mic off the camera's shared mic to the dedicated **USB PnP Sound Device**. **Superseded:** the dongle has AGC and poor pickup at conversational distance, so Vosk mistranscribed and the wake never matched (bug_053). d3b2577 moved the wake mic back to the camera mic; the self-heal from step 1 covers the stall that motivated this move. Do not move the wake mic to the USB PnP device again — that device is the command mic.

## How to verify
`journalctl -u airobot | grep "wake heard"` keeps producing lines over time; "Hey Stella" responds after long idle periods and after heavy camera use. If a stream ever dies, the log shows "Wake mic went silent (dead stream) — reopening". The wake stream must open on the camera mic: `Vosk wake mic open at 48000 Hz (device <hw:3 entry>)`.

## Will it come back?
The self-heal recovers from any silent/failed stream regardless of cause. The durable hardware fix is dedicated I2S mics (INMP441) instead of USB audio; planned. Related audio-flakiness: bug_002, bug_009, bug_010, bug_037; the mic-role history continues in bug_053 and bug_055.
