# Bug #031 — Master-auth bypass: AC and hand-mirror keyword handlers skipped the master check

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `modules/conversation/manager.py`
- **Commit(s):** 875df0c

## Symptom
Not user-visible as a failure — found in the 2026-09-13 architecture audit: anyone speaking could control the AC (`_maybe_ac`) or trigger the hand mirror (`_maybe_mirror`) without being recognised as the master.

## Root cause
`_do_device_control` and the LLM tools enforce the master check, but the fast keyword handlers `_maybe_ac` and `_maybe_mirror` ran before them and had no such gate.

## Fix
Added `_is_master()` / `_deny_master()` and gated `_maybe_ac` and `_maybe_mirror` behind them, respecting `require_authentication`.

## How to verify
With `require_authentication` on and the camera covered (no master recognised), say "turn on the AC": Stella refuses; uncover, get recognised, repeat: it works.

## Will it come back?
Yes if a new `_maybe_*` keyword handler is added without the gate — check every new handler against `_is_master()`.
