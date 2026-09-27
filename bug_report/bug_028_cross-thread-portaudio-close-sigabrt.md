# Bug #028 — Self-inflicted crash loop: closing the PortAudio stream from another thread (SIGABRT every few minutes)

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py` (patch reverted), `core/watchdog.py`, `modules/conversation/manager.py`
- **Commit(s):** 382fb4c

## Symptom
Stella went "offline from time to time"; guard was unreliable (each crash reset `guard_mode` to off). systemd showed restarts every ~1.5-3.5 minutes with core dumps.

## Root cause
The reader-thread fix for bug #027 closed the PortAudio stream from the main thread while the reader thread was still reading it -> heap corruption (`malloc_consolidate(): unaligned fastbin chunk`) -> SIGABRT. systemd restarted the service each time.

## Fix
Reverted the reader-thread capture patch (restored the blocking read). Hang recovery moved to the watchdog (`_conv_activity` stamps + restart after a stuck conversation). Verified: 0 core dumps, stable uptime.

## How to verify
```bash
systemctl show airobot -p NRestarts
coredumpctl list | grep -c python      # should not grow
```

## Will it come back?
No while nobody touches a PortAudio stream from a thread other than the one reading it. Lesson recorded in the decisions log: never close/stop a stream cross-thread.
