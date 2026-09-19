# Bug #029 — "Guard mode off" armed the guard; status questions armed it too

- **Date found:** 2026-09-13
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `modules/conversation/manager.py` (`_maybe_guard`)
- **Commit(s):** 382fb4c

## Symptom
Saying "guard mode off" turned guard ON; asking "is guard on or off?" also armed it; there was no way to hear the current guard status.

## Root cause
Keyword matching: "guard mode off" contains "guard mode" (an arm phrase) but not the exact "guard off" phrase, so the ON keyword matched first. Status questions contain the same words and were treated as ON commands. Motion detection itself was fine.

## Fix
`_maybe_guard`: STATUS queries are handled first (report, never change state); OFF intent = explicit off phrase OR (guard/security topic + an off/disable word); ON only when no "off" word is present. Verified on 7 phrasings.

## How to verify
Say in order: "guard my house" (armed), "is guard on or off?" (reports armed, unchanged), "guard mode off" (disarmed), "is the guard on?" (reports off).

## Will it come back?
No for these phrasings; new phrasings may need adding. `guard_mode` still is not persisted across restarts (open).
