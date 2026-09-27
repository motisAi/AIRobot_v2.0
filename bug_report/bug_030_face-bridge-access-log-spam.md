# Bug #030 — Journal flooded by aiohttp access logs from the face controller polling

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** face-ui
- **Files touched:** `face_bridge/bridge.py`
- **Commit(s):** 382fb4c

## Symptom
`journalctl -u airobot` was dominated by `aiohttp.access` lines, hiding real events.

## Root cause
The Windows face controller polls `/display_state` roughly twice every 1.5 s, and the aiohttp bridge logged every request.

## Fix
Bridge created with `web.AppRunner(app, access_log=None)`. (No disk risk regardless: 90 GB free, journald caps at ~10% and rotates. Vacuuming the system journal needs sudo — skipped.)

## How to verify
```bash
journalctl -u airobot -n 200 | grep -c "aiohttp.access"    # expect 0
```

## Will it come back?
No, unless the AppRunner call is changed.
