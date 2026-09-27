# Bug #043 — Groq gpt-oss-120b rejects the send_photo tool call (400), falls back each time

- **Date found:** 2026-09-19
- **Status:** fixed (2026-09-19, commit 3210df1: required:[]+additionalProperties:false)
- **Area:** ai
- **Files touched:** `modules/conversation/manager.py` (tool schema), `modules/ai/ai_engine.py`
- **Commit(s):** n/a

## Symptom
Every "send me a picture" logs `agent(groq) call failed: Error code: 400 … Tool call validation failed: parameters for tool send_photo did not match schema`, then retries on groq_fast, which works. Costs ~1 s per photo request.

## Root cause
Unclear from the truncated log line. Likely the 120b model emits a `caption` in a shape the strict validator rejects (or adds a field) against a schema with no `required` list. Needs the full error text — the engine truncates it to 140 chars.

## Fix
Not applied yet. Options: log the full error once; declare `"required": []` / `"additionalProperties": false` on `send_photo`; or route photo requests straight to the keyword handler (no LLM needed).

## How to verify
Ask for a photo; journal shows no `agent(groq) call failed`.

## Will it come back?
Yes until fixed; also whenever Groq tightens tool validation.
