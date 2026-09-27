# Bug #036 — With no microphone, Stella held whole conversations with herself

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** conversation
- **Files touched:** `modules/conversation/manager.py`, `modules/audio/speech_recognition.py`
- **Commit(s):** 231740d

## Symptom
At work with no mic plugged in, Stella greeted, asked "anything else?", and said goodbye — a monologue triggered by nothing.

## Root cause
With no mic, `capture_utterance` returned `None` instantly and `_run` treated `None` as silence, walking the normal greeting -> "anything else?" -> farewell path.

## Fix
- `speech_recognition.mic_available()` checks real capture PCMs (`/dev/snd/pcmC*D*c`) so a PortAudio name flip-flop does not mute her at home; no fall-through to PortAudio 'default' when the named mic is absent.
- `manager.py`: no spoken session is started without a mic; quiet break if the mic disappears mid-chat; `_thinking` flag and monotonic activity stamps.

## How to verify
Boot with no mic: face greeting only, no spoken Q&A loop. Plug the mic back in at home: wake word and conversation work as before.

## Will it come back?
No, but verify at home that `mic_available()` returns true with the USB mic (verification after deploy was interrupted).
