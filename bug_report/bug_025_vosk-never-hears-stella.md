# Bug #025 — Voice dead: the Vosk wake model never transcribes "Stella"

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/wake_word.py` (`_matches_wake`)
- **Commit(s):** 382fb4c

## Symptom
After a reboot (and unplugging the CSI camera) Stella gave no voice response at all; guard could not be controlled because it is voice-driven.

## Root cause
The small Vosk EN model renders "stella" as settler / taylor / stellar / sella / "that last" / live — never "stella". The matcher only accepted "stella"/"hey stella" or tokens starting with "stel", so the wake word never matched. It had only ever worked by luck.

## Fix
`_matches_wake` accepts a measured alias set (settler, taylor, stellar, steller, sella, estella, sailor, ...) plus a Levenshtein <= 2 fallback against the keyword. Verified live: "Hey Stella" -> wake detected (conf 0.90) -> normal greeting and answer.

## How to verify
Say "Hey Stella" five times from ~2 m; journal shows a wake detection each time. `journalctl -u airobot -f | grep -i wake`.

## Will it come back?
Possibly — Vosk is a weak wake engine for a name like "Stella" and a new alias may appear. Long-term options noted in the log: Porcupine (needs an access key) or a trained openWakeWord model (needs Moti's voice recordings).
