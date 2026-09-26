# Bug #053 — "Hey Stella" not detected: wake mic mistranscribes on the USB dongle mic

- **Date found:** 2026-09-26
- **Status:** fixed-needs-verify (awaiting a live voice test)
- **Area:** audio
- **Files touched:** `config/config.yaml`
- **Commit(s):** d3b2577

## Symptom
She recognises Moti on camera but does not respond to "Hey Stella". The wake mic hears audio (log shows "wake heard: ...") but Vosk mistranscribes it ("good luck to the moon", "richard"), so it never matches "stella".

## Root cause
Bug_046 moved the wake mic onto the dedicated USB PnP dongle (card 0) to dodge a stall. That mic has Auto Gain Control and poor pickup at conversational distance, so Vosk gets garbage and the wake never matches. It's a mic-quality problem, not a stall — the mic is capturing.

## Fix
Point the wake mic back at the camera mic ("Auto Focus Camera", card 3), which historically detected "stella" reliably and is better positioned. The stall that motivated bug_046 is now separately handled by the wake-loop self-heal (dead-stream reopen) + the command-mic reader-thread (bug_049), so the camera mic is safe to use for wake again. Command STT stays on the USB PnP mic.

## How to verify
Say "Hey Stella": `journalctl -u airobot | grep "Wake word detected"` grows; "wake heard" lines resemble real words, not gibberish.

## Will it come back?
The durable cure is a proper "Hey Stella" model (openWakeWord, needs Moti's voice clips) or the dedicated INMP441 I2S mics — both planned. If the camera mic stalls under heavy video load, the self-heal reopens it within ~20 s.
