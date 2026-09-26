# Bug #047 — Master identity lost repeatedly (cleared on a single Unknown frame)

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** vision
- **Files touched:** `core/robot_brain.py`
- **Commit(s):** 57ad402

## Symptom
After she identified Moti, she kept losing him: "Unknown face — clearing previous identity/master" fired ~17x in 20 min, so master-only actions were denied and she felt broken/stuck.

## Root cause
`RobotBrain._handle_face_detected` cleared master/authenticated/current_user the instant ONE frame came back `unknown` — with no debounce and ignoring the configured `identity_memory_seconds` grace. Off-angle frames (which read unknown) wiped a just-confirmed master.

## Fix
Within `identity_memory_seconds` (60s) of the last master sighting, treat an `unknown` as a mis-read and return. Otherwise require 3 consecutive unknowns before clearing (same >=3 pattern main.py uses for guard). Reset the streak whenever a real identity (master or known) appears.

## How to verify
`journalctl -u airobot | grep -c "clearing previous identity"` stays near 0 while Moti is present; master persists through brief off-angle frames.

## Will it come back?
No for single-frame flips. Pair with bug_048 (recognition quality) + a re-enroll for best results.
