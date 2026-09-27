# Bug #006 — RobotEvent objects not comparable, crashing the PriorityQueue

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `core/robot_brain.py`
- **Commit(s):** 505b75d

## Symptom
Event dispatch in the brain raised `TypeError` (comparison not supported) when two events with equal priority were queued.

## Root cause
`PriorityQueue` compares tuple entries; when priorities tie it falls through to comparing the `RobotEvent` payloads, which defined no ordering.

## Fix
Added comparison support (ordering dunder methods) to `RobotEvent` so ties resolve without error.

## How to verify
Fire several events of the same priority quickly (e.g. multiple face detections); no `TypeError: '<' not supported` in the journal.

## Will it come back?
No.
