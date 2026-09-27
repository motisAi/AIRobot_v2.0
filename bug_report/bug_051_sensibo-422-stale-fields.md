# Bug #051 — Sensibo AC returns 422 on a mode/temperature change

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** smart-home
- **Files touched:** `modules/smart_home/sensibo.py`
- **Commit(s):** 3210df1

## Symptom
"Turn on the AC to 22/23" sometimes logged "Sensibo set failed: HTTP Error 422 Unprocessable Entity"; the change didn't apply.

## Root cause
`_apply` POSTed the whole merged acState seeded from a possibly-stale cache, echoing back fields (e.g. a `targetTemperature` from a previous mode) that are invalid for the new mode — `targetTemperature` is not accepted in fan/dry mode. Sensibo rejects the combination with 422.

## Fix
Build the POST from the AC's REAL current state (not the cache), drop `targetTemperature` when the resulting mode is fan/dry, and surface the HTTP error body so any future 4xx is diagnosable.

## How to verify
Change AC mode then temperature; no 422; the AC applies the setting. On any future error the log now shows "Sensibo HTTP <code>: <body>".

## Will it come back?
Only if Sensibo changes its acState validation; the error body will show it.
