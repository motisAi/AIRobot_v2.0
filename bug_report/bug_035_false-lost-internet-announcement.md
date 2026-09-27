# Bug #035 — "I lost internet" announced ~16 s after every offline boot

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** network
- **Files touched:** `main.py` (network monitor)
- **Commit(s):** 231740d

## Symptom
Booted with no internet, Stella spoke unprompted shortly after start, announcing that the internet connection was lost.

## Root cause
The network monitor hard-coded `was_online=True`, so a boot into an offline state looked like a transition online -> offline and triggered the announcement.

## Fix
The monitor initialises from the real connectivity state (no announce when booted offline), rate-limits announcements to once per 30 minutes, and pushes the `online` flag to the AI engine (used by bug #034).

## How to verify
Boot with the uplink unplugged; wait 2 minutes: no spoken network announcement, journal shows the monitor starting in the offline state.

## Will it come back?
No. Real online -> offline transitions are still announced (max once per 30 min).
