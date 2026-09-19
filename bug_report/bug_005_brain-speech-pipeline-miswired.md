# Bug #005 — Brain speech pipeline mis-wired (wrong module, unregistered modules, double listen call)

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `core/robot_brain.py`, `main.py`, `modules/audio/speech_recognition.py`, `config/settings.py`
- **Commit(s):** f4c986f, 06b0749, 2c1ded4

## Symptom
After the wake word fired, no command was captured, or the robot listened twice in a row; `.env` secrets were not loaded; the master greeting used the wrong name.

## Root cause
- `robot_brain` called `start_recording` on the wrong module instead of the speech module, and modules were not registered with the brain (f4c986f).
- Brain did not call `listen_for_command`; there was no `.env` loader; greeting name not taken from config (06b0749).
- `main.py` invoked `listen_for_command` a second time after the brain already had (2c1ded4). FLAC (needed by the Google STT backend) was not installed on the Pi.

## Fix
Use the speech module for recording, register modules properly, call `listen_for_command`, add a `.env` loader in `main.py`, remove the duplicate call in `main.py`, install `flac` on the Pi.

## How to verify
Say the wake word, then a command: the journal shows exactly one "listening for command" and one transcription per wake.

## Will it come back?
No, barring a refactor of brain/module registration.
