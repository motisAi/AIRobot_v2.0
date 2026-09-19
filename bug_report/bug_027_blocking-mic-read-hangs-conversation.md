# Bug #027 — Blocking mic read hung conversations; wake mic stayed paused (deaf for 10 min)

- **Date found:** 2026-09-13
- **Status:** fixed (recovery via watchdog; root cause worked around, not removed)
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py`, `modules/conversation/manager.py`, `core/watchdog.py`
- **Commit(s):** 382fb4c, cf4fc5f

## Symptom
Intermittently Stella stopped responding to voice. Journal: wake fired, "Pausing wake-word listener — releasing mic", she asked "Can I do anything else?", then silence for 10 minutes — no farewell, no "listener resumed".

## Root cause
PortAudio `stream.read()` blocks forever when the flaky USB mic stops delivering frames mid-capture. `capture_utterance` hangs, the conversation never ends, and the wake-word listener never resumes. `get_read_available()` always returns 0 here, so it cannot be used to detect the stall.

## Fix
A first fix (reader thread + queue with timeout) caused crashes and was reverted — see bug #028. The safe replacement: the conversation stamps `_conv_activity` on every spoken line (and, from cf4fc5f, on user speech too); the hardware watchdog restarts the service if a conversation is active but silent for 120 s (later tuned 60 -> 75 s in cf4fc5f to stop false restarts). Normal silence still emits wrap-up/farewell every ~12-24 s, so only a true hang trips it.

## How to verify
Unplug the USB mic mid-conversation: within ~75 s the watchdog restarts `airobot` and the wake listener is back. `journalctl -u airobot | grep -i watchdog`.

## Will it come back?
The hang itself can still occur (flaky USB audio) — the watchdog only recovers from it. A source-level fix must reopen the whole PyAudio instance or do a process-level restart, never a cross-thread stream close.
