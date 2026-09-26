# Bug #054 — SIGABRT crash-loop every ~2 min ("malloc: unaligned tcache chunk")

- **Date found:** 2026-09-26
- **Status:** fixed (final resolution = single-thread capture, 442c8a8; the first fix ed73b10 was reverted)
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py`
- **Commit(s):** ed73b10 (first fix, reverted), 442c8a8 (final)
- **Superseded by:** bug_055
- **Reverted:** ed73b10 ("reader thread is the sole owner") was reverted 12 minutes later by 442c8a8

## Symptom
She worked briefly then "got stuck" repeatedly; every ~2 minutes the whole service restarted (30s blackout). Also she "still had no history" — because each crash killed the session-end fact-learning before it could store anything.

Journal 2026-09-26: `Wake word detected` 13:31:40 → `malloc(): unaligned tcache chunk detected` 13:31:43 →
`Main process exited, code=dumped, status=6/ABRT`; again at 13:33:07 → 13:33:59.

## Root cause
The bug_049 command-mic reader thread and the main thread BOTH touched the PortAudio stream: the reader was blocked in `stream.read()` while the main thread's `finally` called `stream.stop_stream(); stream.close()`. Closing a PortAudio/ALSA stream from another thread while a read is in flight corrupts the heap → `malloc(): unaligned tcache chunk detected` → SIGABRT core-dump → systemd restart. Same cross-thread-audio class as bug_028 (2026-09-13), which had been reverted for exactly this reason.

## Fix
**First attempt, ed73b10 (13:39) — reverted.** Made the reader thread the SOLE owner of the stream (it closed the stream
in its own `finally`; the main thread only set the stop event and `_rt.join(timeout=2.0)`). Stress-tested with 4
open/read/close cycles: clean exit, no abort. That test never exercised the stall branch. In production 8 minutes later
(13:47) a capture stalled, the main thread's join timed out and it **continued, abandoning a thread that still held
the ALSA device**. Every following open failed: `Could not open command mic for capture` ×8 between 13:47 and 13:51.
An abandoned stream is a leak that only a restart clears.

**Final resolution, 442c8a8 (13:51).** Reader thread removed. `capture_utterance()` opens, reads and closes on the one
conversation thread, with a hard wall-clock deadline between reads; a truly wedged `read()` is bounded by the two-stage
watchdog (75 s soft stop, +25 s restart). No thread is ever abandoned, so nothing can hold the device. The leak was
then misread as a faulty USB mic, which is bug_055.

## How to verify
`systemctl show airobot -p NRestarts` stays flat during use; `journalctl -u airobot | grep -cE "ABRT|malloc"` stays 0
across conversations; no `Could not open command mic for capture` after a stalled capture. Verified since a2e3229
(2026-09-26 20:26): 0 restarts, 0 matches.

## Will it come back?
No, as long as only one thread touches a given PortAudio stream **and** no thread holding a stream is ever abandoned.
Rules (architecture doc §12, 1–2): never open/read on one thread and close on another; a close/join timeout is terminal
for that device — flag it and let the watchdog restart, never reopen. Durable cure for all this USB-audio fragility
remains the I²S mics.
