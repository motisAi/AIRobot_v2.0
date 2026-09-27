# Bug #018 — Double master greeting + welcome spam made Stella look frozen (the real "stuck")

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `core/robot_brain.py`, `main.py`
- **Commit(s):** e2cc602

## Symptom
"Welcome back, Moti" spoken twice in a row, re-welcomed every 30-60 s, then Stella appeared frozen / unresponsive to voice.

## Root cause
Two subscribers greeted the master on every face event: `core/robot_brain.py` (spoken "Welcome back, {name}!") and `main._welcome_master` (wave + emotion opener + listening session). The on-sight cooldown was left at demo values (gap > 20 s, since_welcome > 60 s), so it re-triggered constantly. Overlapping TTS then hit `aplay: Device or resource busy` and speech died.

## Fix
- Brain no longer speaks the master greeting; `main._welcome_master` owns it.
- On-sight cooldown restored to gap > 90 s and since_welcome > 240 s.

## How to verify
Walk in front of the camera: exactly one greeting; stay there for 5 minutes: no second welcome; `journalctl -u airobot | grep -c "Welcome back"` grows by one per return.

## Will it come back?
No, unless a new subscriber to the face event also speaks. Rare `aplay ... busy` (error 524) can still occur from other overlapping speech — it is watchdog-handled; a proper fix (pulseaudio access for the systemd unit) is still open.
