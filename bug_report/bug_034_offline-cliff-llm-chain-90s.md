# Bug #034 — Offline cliff: every answer took ~90 s+ with no internet

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** ai
- **Files touched:** `modules/ai/ai_engine.py`, `main.py`
- **Commit(s):** 231740d (dated 2026-09-14 by the Pi clock)

## Symptom
At work (direct cable, no internet) Stella was extremely slow to answer anything, then dumb (offline brain also dead — bug #011).

## Root cause
Every `think()` tried groq -> groq_fast -> gemini before hailo; each had a 30 s timeout times the SDK's 2 retries, and connection errors were never cooled down. Measured offline `think()` ~90 s+.

## Fix
`ai_engine.py`: an `online` flag (pushed by the network monitor in `main.py`) makes `_skip()` bypass cloud providers when offline; `max_retries=0`; `httpx.Timeout(20, connect=3)`; agent/web-search gated on online; reasoning-model `max_tokens >= 1024` with an empty-content warning. Offline `think()` measured 0.2 s.

## How to verify
Unplug the uplink, ask a question: the journal shows cloud providers skipped and a hailo answer within a few seconds (requires bug #011 fixed for the answer itself).

## Will it come back?
No for the timeouts. Startup-complete verification after deploy is still pending (Ethernet dropped during the check). Rollback: `git checkout 62fec28 -- modules/ai/ai_engine.py`.
