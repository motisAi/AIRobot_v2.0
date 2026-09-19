# Bug #017 — Sensibo AC commands failed with 429 (rate limit) on rapid calls

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** smart-home
- **Files touched:** `modules/smart_home/sensibo.py`
- **Commit(s):** e2cc602

## Symptom
AC voice commands intermittently failed right after a previous one; the Sensibo cloud API answered HTTP 429.

## Root cause
Sensibo's cloud API rate-limits hard, and the module made several calls per command (read state, then set).

## Fix
`sensibo.py` caches the `acState` and issues a single combined set call per command, so each voice command costs one API request.

## How to verify
Issue two AC commands a few seconds apart ("make it cold", "set the AC to 24"); both succeed, no 429 in the journal.

## Will it come back?
Yes if voice commands are fired back-to-back within a second or two — space them a few seconds apart. Sensibo is cloud-only, so it never works offline.
