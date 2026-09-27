# Bug #012 — App called hailo-ollama's /api/chat, which returns empty on this build

- **Date found:** 2026-09-05
- **Status:** fixed
- **Area:** ai
- **Files touched:** `modules/ai/ai_engine.py`, `docs/hardware/hailo.md`
- **Commit(s):** e2cc602

## Symptom
Even with the Hailo driver working, the offline brain never produced an answer — the fallback to the local model yielded empty responses, so offline Stella was silent.

## Root cause
`hailo-ollama` 5.1.1 serves **only** the OpenAI-compatible `POST /v1/chat/completions`. The ollama-native `/api/chat` and `/api/generate` return empty / errors on this build, and the app was using those.

## Fix
`ai_engine.py` now uses `/v1/chat/completions` everywhere for the hailo provider and reads `choices[0].message.content`. The model was also pinned to `qwen2.5-instruct:1.5b` (llama3.2:3b is too slow on the NPU, ~19 s+) and a background warm-up preloads it at startup (cold load ~40 s).

## How to verify
```bash
curl -s localhost:8000/api/tags
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-instruct:1.5b","messages":[{"role":"user","content":"hi"}],"stream":false}'
```
Then unplug the internet and ask Stella a question — she should answer within ~2-5 s (warm).

## Will it come back?
Only if hailo-ollama is upgraded to a build with a different API surface. Requires bug #011 to be healthy first.
