# Bug #042 — "No, thank you, Stella." not recognised as goodbye → 5 s LLM round trip

- **Date found:** 2026-09-19
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `modules/conversation/manager.py`, `modules/ai/ai_engine.py`
- **Commit(s):** see decision log 2026-09-19 (goodbye normalisation)

## Symptom
Moti said "No, thank you, Stella." to end the chat. Instead of "See you later" she thought for 5.6 s, the cloud brains produced nothing, and the Hailo model replied "Sure, I'm here whenever you need help".

## Root cause
`_is_end_phrase` compared the raw lowercase text against `conversation.end_phrases` exactly; commas and her name broke the match, so the sentence went to the LLM. The cloud providers were then skipped or returned empty content, and nothing logged why.

## Fix
Strip punctuation and the robot's name before matching end phrases. Log a provider skip (cooldown/offline) and an EMPTY reply with its duration so the next slow answer is explained in the journal.

## How to verify
Say "No, thank you, Stella." → immediate "See you later". Journal shows no "Answered via fallback provider" for goodbyes.

## Will it come back?
No for goodbyes. The underlying "empty content from gpt-oss/gemini" is still possible; the new log line will show it.
