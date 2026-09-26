# Bug #049 — Hard "stuck / no response" + 30s blackout (unbounded mic read)

- **Date found:** 2026-09-19
- **Status:** REVERTED — do not reintroduce a reader thread; see bug_054/bug_055 (reverted in 442c8a8)
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py`, `core/watchdog.py`
- **Commit(s):** 440223d (reader thread + two-stage watchdog), ed73b10 (reader made sole owner), 442c8a8 (reader thread reverted; single-thread read kept)
- **Superseded by:** bug_054 (the crash the reader thread caused), bug_055 (the device leak and the misdiagnosis it led to)
- **Reverted:** the reader-thread half of 440223d, by 442c8a8 on 2026-09-26 13:51. The two-stage watchdog half stands.

## Symptom
Mid-conversation she would stop responding, then the whole service restarted (~30s blackout: wake, vision, guard, Telegram all drop).

## Root cause
`capture_utterance` called `stream.read()` with no deadline. A stalled USB dongle wedged the read forever, freezing `_conv_activity`, blocking `conversation.stop()`, so the watchdog's only escape was a full `systemctl restart`.

## History (kept — this is the second time the same mistake was made)
- 2026-09-13: bug_027 fixed the same hang with a reader thread; bug_028 recorded the SIGABRT crash loop it caused
  (cross-thread `stream.close()`), and the thread was reverted. The lesson was written in `docs/decisions/log.md` and
  bug_028 — but the 09-13 decision-log entry still presented the reader thread as "the fix", with the revert only in
  the next entry.
- 2026-09-19 22:43, 440223d: this bug re-added a background reader thread (8 s consumer timeout) plus the two-stage
  watchdog. The "How to verify" below originally read "Verified the reader-thread path returns cleanly (no hang)" — an
  open/read/close loop, never a real stall.
- 2026-09-26 13:31 and 13:33: `malloc(): unaligned tcache chunk detected` → SIGABRT after each wake (bug_054): the main
  thread's `finally` closed the stream while the reader was blocked in `read()`.
- 2026-09-26 13:39, ed73b10: reader thread made the sole owner of the stream. 13:47: a stall ended the capture, the main
  thread abandoned the still-blocked reader (`join(timeout=2.0)` and continue), and the reader kept the ALSA device —
  `Could not open command mic for capture` ×8.
- 2026-09-26 13:51, 442c8a8: reader thread removed; single-thread blocking read restored with a hard wall-clock
  deadline. The USB mic was then wrongly blamed and a single-mic detour tried (bug_055), until a2e3229 restored the
  two-mic layout at 20:26. 0 restarts since.

## Fix (current mechanism, 442c8a8 + a2e3229)
No reader thread. `capture_utterance()` runs **open → read loop → close on the one conversation thread**, and bounds a
stalled read two ways:
1. A **hard wall-clock deadline** checked between reads: `hard_deadline = t0 + max(start_timeout, max_seconds) + 5.0`;
   on expiry it logs `Command mic capture exceeded its hard deadline — ending` and closes the stream (same thread).
2. The **two-stage watchdog** (`core/watchdog.py`, kept from 440223d) for a `read()` that never returns: after 75 s
   with no `_conv_activity` stamp it soft-stops the session (`conversation.stop()`); if the session is still active
   25 s later it restarts the service (`sudo -n systemctl restart airobot`). This is the only bound for a truly wedged
   `read()`, by design.

The code site carries the comment: "Single-thread capture: open -> read -> close all here. (A background reader thread
was tried and REVERTED: closing the stream cross-thread crashed the process, bug_054, and abandoning a blocked reader
leaked the device so the next capture could not open it.)" Pointer to the 09-13 lesson: bug_028. Canonical design:
`docs/architecture/stella-architecture.md` §4.2 and §4.6.

## How to verify
- A stall now ends at the hard deadline: journal shows `Command mic capture exceeded its hard deadline — ending`, the
  session ends, `Wake-word listener resumed` follows, no service restart.
- A truly wedged read: `Conversation stuck (no speech 75s) — soft-stopping the session`, then (if still stuck)
  `Conversation still stuck after soft stop — restarting service` ~25 s later.
- `journalctl -u airobot | grep -cE "malloc|status=6"` stays 0 across conversations; `systemctl show airobot -p NRestarts` flat.
- The real reproduction (usbreset/unplug the mic while a read is blocked) is a bench item in the architecture doc §13 and
  has not yet been performed.

## Will it come back?
The hang class is bounded now regardless of the dongle. The *crash/leak* class comes back only if someone adds a thread
that touches a PortAudio stream it did not open, or abandons a thread holding one — both forbidden (architecture doc
§12 rules 1–2). Durable hardware fix remains dedicated I²S mics (INMP441) off USB.
