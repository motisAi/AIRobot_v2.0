# Bug #015 — Google STT failed repeatedly, dropping to weak Vosk transcriptions

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/speech_recognition.py`
- **Commit(s):** e2cc602

## Symptom
Commands were misheard or ignored; Stella appeared "stuck"/unresponsive. The journal showed Google STT failing 57 times in one session.

## Root cause
Google STT (the primary) was failing, so every utterance fell back to the small offline Vosk model, which mishears. (Note: the 2026-09-12 "stuck" itself turned out to be bug #018, not STT — but the STT weakness was real.)

## Fix
New STT order: **Groq Whisper (`whisper-large-v3-turbo`)** first via multipart POST to `/openai/v1/audio/transcriptions` using the existing `GROQ_API_KEY` (`_transcribe_groq`), then Google, then Vosk (`_transcribe_pcm16k`).

## How to verify
Speak a sentence; the journal shows the transcript coming from the groq STT path with a full, correct sentence. Test used: `espeak "turn on the light"` -> "Turn on the light."

## Will it come back?
Only if Groq rate-limits or the key is missing — the chain then degrades to Google/Vosk. Whisper introduced its own problem: hallucinated phrases from silence (bug #026).
