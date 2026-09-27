# Bug #053 — "Hey Stella" not detected: wake mic mistranscribes on the USB dongle mic

- **Date found:** 2026-09-26
- **Status:** verified (2026-09-26 journal: wake stream on the camera mic, `Wake word detected` at 21:12:09, `NRestarts=0`)
- **Area:** audio
- **Files touched:** `config/config.yaml`
- **Commit(s):** d3b2577 (wake back to the camera mic); a2e3229 (two-mic layout restored after the bug_055 detour)
- **Superseded by:** —
- **Reverted:** — (the intermediate single-mic detour 2051a76 that followed this fix is recorded in bug_055)

## Symptom
She recognises Moti on camera but does not respond to "Hey Stella". The wake mic hears audio (log shows "wake heard: ...") but Vosk mistranscribes it ("good luck to the moon", "richard"), so it never matches "stella".

## Root cause
Bug_046 moved the wake mic onto the dedicated USB PnP dongle (card 0) to dodge a stall. That mic has Auto Gain Control and poor pickup at conversational distance, so Vosk gets garbage and the wake never matches. It's a mic-quality problem, not a stall — the mic is capturing.

## Fix
Point the wake mic back at the camera mic ("Auto Focus Camera", ALSA card 3 / id `Camera`), which historically detected "stella" reliably and is better positioned. The stall that motivated bug_046 is handled by the wake-loop self-heal (dead-stream reopen). Command STT stays on the USB PnP mic (card 0 / id `Device`). This is the **canonical two-mic layout** — roles, by-id names, rates, thread ownership and the handoff sequence are specified in
[docs/architecture/stella-architecture.md §4](../docs/architecture/stella-architecture.md#4-audio--the-canonical-two-mic-design); every other mention of mic roles in the repo defers to that section.

(The original text here said the stall was "separately handled by the command-mic reader-thread (bug_049)". That reader thread was reverted the same afternoon — bug_054/bug_055 — and is not part of the design.)

## How to verify
Say "Hey Stella": `journalctl -u airobot | grep "Wake word detected"` grows; "wake heard" lines resemble real words, not gibberish; the wake stream opens on the camera mic (`Vosk wake mic open at 48000 Hz (device 2)` where device 2 is `Auto Focus Camera: USB Audio (hw:3,0)` in the boot device table).

Verified 2026-09-26 after a2e3229 (20:26): `Vosk wake mic open at 48000 Hz (device 2)` at 20:26:45; `Wake word detected (method=vosk, confidence=0.90)` at 21:12:09 followed by `Pausing wake-word listener` and, at 21:13:02, `Wake-word listener resumed` + reopen; 0 `Could not open` / `Timed out` lines; `NRestarts=0`. The 10-cycle count from the architecture doc §13 checklist is still to be logged here.

## Will it come back?
The durable cure is a proper "Hey Stella" model (openWakeWord, needs Moti's voice clips) or the dedicated INMP441 I2S mics — both planned. If the camera mic stalls under heavy video load, the self-heal reopens it within ~20 s. Moving the wake role to the USB PnP mic again would bring this bug back: the two-mic assignment is an invariant, not a tuning knob.
