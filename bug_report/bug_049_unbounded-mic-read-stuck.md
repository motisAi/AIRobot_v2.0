# Bug #049 — Hard "stuck / no response" + 30s blackout (unbounded mic read)

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py`, `core/watchdog.py`
- **Commit(s):** 440223d

## Symptom
Mid-conversation she would stop responding, then the whole service restarted (~30s blackout: wake, vision, guard, Telegram all drop).

## Root cause
`capture_utterance` called `stream.read()` with no deadline. A stalled USB dongle wedged the read forever, freezing `_conv_activity`, blocking `conversation.stop()`, so the watchdog's only escape was a full `systemctl restart`.

## Fix
Read frames in a background daemon thread; the consumer times out after 8s of no audio and ends the utterance cleanly (get_read_available is unreliable on this ALSA build — returns 0 — so the thread+timeout variant is used). Watchdog now soft-stops the session first and only restarts the service if still stuck ~25s later.

## How to verify
A mic stall logs "Command mic stalled (no audio 8s) — ending capture", the session ends, wake resumes, no service restart. Verified the reader-thread path returns cleanly (no hang).

## Will it come back?
Self-bounded now regardless of the dongle. Durable fix: dedicated I2S mics off USB.
