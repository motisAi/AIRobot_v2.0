# Bug #003 — Root logger stuck at WARNING; numpy RMS overflow warnings in wake-word loop

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** deploy
- **Files touched:** `main.py`, `modules/audio/wake_word.py`, `check_system.py`
- **Commit(s):** c1d614a

## Symptom
No INFO/DEBUG lines appeared in the journal, making the speech pipeline impossible to debug; the wake-word energy loop printed numpy overflow RuntimeWarnings.

## Root cause
Logging setup in `main.py` left the root logger level at WARNING, so module INFO logs were dropped. The RMS energy computation squared int16 samples without widening the dtype, overflowing.

## Fix
`main.py` sets the root logger level explicitly; RMS in `wake_word.py` computes on a wider dtype before squaring. `check_system.py` diagnostic script added.

## How to verify
```bash
journalctl -u airobot -n 50 | grep -c INFO                      # > 0
journalctl -u airobot -n 500 | grep -i "overflow encountered"   # expect nothing
```

## Will it come back?
No, unless logging setup in `main.py` is rewritten.
