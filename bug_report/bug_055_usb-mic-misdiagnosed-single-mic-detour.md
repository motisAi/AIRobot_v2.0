# Bug #055 — USB PnP mic misdiagnosed as "wedged"; single-mic detour failed (the real cause was our leaked reader thread)

- **Date found:** 2026-09-26
- **Status:** fixed
- **Area:** audio
- **Files touched:** `config/config.yaml` (mic roles), `modules/audio/speech_recognition.py` (capture path)
- **Commit(s):** 2051a76 (the detour), a2e3229 (the fix)
- **Superseded by:** —
- **Reverted:** 2051a76 was undone by a2e3229 the same day

## Symptom
Afternoon of 2026-09-26, three episodes in a row:
1. 13:47 — after a capture stalled (`Command mic stalled (no audio 8s) — ending capture`), every following session
   logged `Could not open command mic for capture` (8 times in 4 minutes). She woke on "Hey Stella" but heard nothing.
2. The USB PnP mic was declared "wedged / unopenable" and commit 2051a76 (13:56) routed the **command** mic to the camera
   mic as well — one mic for wake and command.
3. 14:40 — under that single-mic layout: `Timed out waiting for wake-word stream to close`, then twice
   `Could not open command mic for capture`. Deaf again, this time on the camera mic.

## Root cause
A misdiagnosis. The USB PnP mic was never faulty:
- The 13:47 failures were caused by **our own leaked reader thread**. Commit ed73b10 (13:39, bug_054) made a background
  reader thread the sole owner of the command stream; when a read stalled, the main thread did `join(timeout=2)` and
  **continued**, abandoning a thread that still held the ALSA device (`hw:0,0`). Every later `audio.open()` on that
  device failed. 442c8a8 (13:51) reverted to a single-thread read, which removed the leak — but the mic had already been
  blamed.
- The single-mic detour then failed for a different, structural reason: with both roles on one ALSA device the
  wake-stream **close** (in the wake thread) and the command-stream **open** (in the conversation thread) contend for the
  same device. When the wake thread did not close within `pause_listening()`'s 2 s wait, the code logged a WARNING and
  proceeded to open the command mic on the still-held device → `Could not open command mic`. With two devices this
  contention cannot happen.

Commit 2051a76 cited "bug_055" but no such file existed until now (rule: a commit may not cite a bug number that is not
in `bug_report/`).

## Fix
a2e3229 (20:26) restored the **two-mic layout**: wake = camera mic ("Auto Focus Camera", ALSA card `Camera`, hw:3),
command = "USB PnP Sound Device" (ALSA card `Device`, hw:0). No code change to capture was needed beyond 442c8a8's
single-thread read + hard deadline. The canonical design, invariants and handoff sequence are in
[docs/architecture/stella-architecture.md §4](../docs/architecture/stella-architecture.md#4-audio--the-canonical-two-mic-design).

## How to verify
```bash
journalctl -u airobot --since "2026-09-26 20:26" | grep -E "Resolved speech mic|Vosk wake mic open"
#  Resolved speech mic 'USB PnP Sound Device' -> device 0     (hw:0,0)
#  Vosk wake mic open at 48000 Hz (device 2)                  (hw:3,0 = camera mic)
journalctl -u airobot --since "2026-09-26 20:26" | grep -cE "Could not open|Timed out waiting|status=6|malloc"   # 0
systemctl show airobot -p NRestarts                                                                              # 0
```
Verified 2026-09-26: 0 matches, `NRestarts=0`, wake at 21:12:09 on the camera mic, `Pausing` / `resumed` paired at 21:13:02.

## Will it come back?
No — as long as the two-mic invariant is enforced: wake and command resolve to two different devices (startup refuses
otherwise, architecture doc I1), no reader thread is ever reintroduced (bug_028, bug_054), and a wake-close timeout is
treated as terminal instead of proceeding to another open. Single-mic handoff is unsupported and is not to be retried.
