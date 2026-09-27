# Bug #026 — Phantom replies: Whisper/Vosk hallucinate phrases from silence (even enrolled as names)

- **Date found:** 2026-09-12 (noticed), fixed 2026-09-13, recurred and re-fixed 2026-09-16
- **Status:** fixed-needs-verify
- **Area:** conversation
- **Files touched:** `modules/audio/speech_recognition.py` (`capture_utterance`), `modules/conversation/manager.py` (`_enroll_name`)
- **Commit(s):** 382fb4c, 231740d

## Symptom
Stella answered nobody ("Thank you.", "."), kept conversations alive forever, and on 2026-09-16 said "Nice to meet you, Foreign" — a hallucinated word was enrolled as a person's name.

## Root cause
Whisper (and Vosk) produce short no-speech hallucinations ("Foreign", "Thank you.", "so", "oh", "Hey", ".") from ambient noise between utterances. The conversation loop treated them as real speech; the name-enrollment path accepted any token as a name.

## Fix
- 382fb4c: `capture_utterance` discards captures below RMS 220 or shorter than 0.35 s before transcribing.
- 231740d: a Whisper hallucination stoplist is treated as silence; `_enroll_name` rejects junk names.

## How to verify
Leave Stella awake in a quiet room after a real exchange for 2 minutes: she wraps up and says goodbye instead of replying to nothing; `journalctl -u airobot | grep -i "nice to meet you"` shows only real names.

## Will it come back?
Likely with new hallucination strings — extend the stoplist in `speech_recognition.py`. The 231740d part still needs a post-restart verification (the Ethernet link dropped before it was confirmed).
