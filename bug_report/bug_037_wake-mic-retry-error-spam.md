# Bug #037 — Wake-word listener retried a missing mic at 1 Hz forever with an ERROR each time

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/wake_word.py`
- **Commit(s):** 231740d

## Symptom
Journal full of wake-mic open errors (381 in 6 minutes) when no mic was attached; CPU and log churn.

## Root cause
The wake-word loop retried opening the input device every second and logged at ERROR on every failure, with no backoff.

## Fix
Log once, then exponential backoff 1 s -> 60 s while no mic is present. Also set `_stream_closed_event` in the paused branch to remove a 2 s stall on every pause.

## How to verify
```bash
journalctl -u airobot --since "-10 min" | grep -ci "wake.*mic"   # a handful, not hundreds
```

## Will it come back?
No.
