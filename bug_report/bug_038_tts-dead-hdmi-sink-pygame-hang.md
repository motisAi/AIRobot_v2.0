# Bug #038 — TTS pinned to a dead HDMI sink; first fix made startup hang in pygame

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/text_to_speech.py`, `core/watchdog.py`
- **Commit(s):** 231740d

## Symptom
With no HDMI display attached, speech went nowhere and the watchdog re-probed audio every 20 s. After the first fix, startup stalled at "Using Piper TTS engine" and never completed.

## Root cause
TTS output was pinned to `sysdefault:CARD=vc4hdmi0`; with no HDMI sink, `aplay` failed and the watchdog kept re-probing. **Regression:** the first fix returned `None` for the device, so TTS took the pygame path, and `pygame.mixer.init()` hangs when there is no sound device.

## Fix
`text_to_speech.py` falls back to a non-HDMI output when HDMI has no sink, `aplay` gets a 60 s timeout, and the device resolver **always returns a device string** (a fast-failing aplay) instead of `None`. Watchdog audio re-probe backoff 20 s -> 300 s.

## How to verify
Boot without HDMI: the journal reaches "startup complete" and TTS attempts log a device name; with HDMI at home, speech still comes out of the HDMI speakers.

## Will it come back?
No for the hang; the pygame path must never be reached with no device. Verification after the regression fix is pending (link dropped).
