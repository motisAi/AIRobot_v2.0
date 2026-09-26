# Bug #054 — SIGABRT crash-loop every ~2 min ("malloc: unaligned tcache chunk")

- **Date found:** 2026-09-26
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py`
- **Commit(s):** ed73b10

## Symptom
She worked briefly then "got stuck" repeatedly; every ~2 minutes the whole service restarted (30s blackout). Also she "still had no history" — because each crash killed the session-end fact-learning before it could store anything.

## Root cause
The bug_049 command-mic reader thread and the main thread BOTH touched the PortAudio stream: the reader was blocked in `stream.read()` while the main thread's `finally` called `stream.stop_stream(); stream.close()`. Closing a PortAudio/ALSA stream from another thread while a read is in flight corrupts the heap → `malloc(): unaligned tcache chunk detected` → SIGABRT core-dump → systemd restart. Same cross-thread-audio class as an earlier reverted crash.

## Fix
Make the reader thread the SOLE owner of the stream: it closes the stream in its own `finally`. The main thread only sets the stop event and `_rt.join(timeout=2.0)` — it never touches the stream. Stress-tested 4 open/read/close cycles: clean exit, no abort. Live: 0 restarts, 0 ABRT since deploy.

## How to verify
`systemctl show airobot -p NRestarts` stays flat during use; `journalctl -u airobot | grep -c ABRT` stays 0 across conversations.

## Will it come back?
No, as long as only one thread touches a given PortAudio stream. Rule: never open/read on one thread and close on another. Durable cure for all this USB-audio fragility remains the I2S mics.
