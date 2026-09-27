# Bug #013 — Cloud LLM models retired (404) and a blank Gemini key silently disabled providers

- **Date found:** 2026-09-05 (earlier occurrence undated: llama-3.1-8b-instant)
- **Status:** recurring
- **Area:** ai
- **Files touched:** `modules/ai/ai_engine.py`, `config/config.yaml`, `.env` (gitignored)
- **Commit(s):** e2cc602

## Symptom
Cloud answers failed with 404 and Stella fell back down the chain (eventually to a broken offline brain); the Gemini slot never answered.

## Root cause
- Groq decommissioned `llama-3.1-8b-instant` and later `llama-3.3-70b-versatile` (404).
- `.env` had `GEMINI_API_KEY=` blank and the configured `gemini-2.0-flash` model was retired (404).

## Fix
Chain updated to `groq(openai/gpt-oss-120b) -> groq_fast(openai/gpt-oss-20b) -> gemini(gemini-3.6-flash) -> hailo(qwen2.5-instruct:1.5b)`, `max_tokens` 512, Gemini key set. (`groq/compound` web-search was tried and reverted: `request_too_large` on the free tier and it shares Groq's rate budget.)

## How to verify
```bash
journalctl -u airobot -n 300 | grep -iE "404|decommission|model_not_found"   # expect nothing
```
Ask a question online; the journal should show the answer coming from the groq provider.

## Will it come back?
Yes — whenever a provider retires a model name. Fix is a config change of the model id; check the provider's model list when 404s appear.
