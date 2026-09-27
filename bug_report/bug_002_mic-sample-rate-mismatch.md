# Bug #002 — Wake-word / speech mic opened at a sample rate the USB mic rejects

- **Date found:** 2026-05-30
- **Status:** fixed
- **Area:** audio
- **Files touched:** `modules/audio/wake_word.py`, `modules/audio/speech_recognition.py`, `config/settings.py`, `fix_alsa.py`, `test_mic_rates.py`
- **Commit(s):** a22333e, 7be3d13, 06b0749

## Symptom
Wake word never triggered and speech capture failed on the Pi; the audio stream could not be opened (ALSA invalid-sample-rate errors).

## Root cause
The wake-word code hard-coded its own sample rate instead of using the configured one, and the VAD had a rate check that rejected the rate the hardware actually supports. The USB mic only accepts its native rates (44.1k/48k), not 16 kHz.

## Fix
- a22333e: wake word uses the configured rate (44100 Hz); VAD rate check patched to allow it.
- 7be3d13: mic opened at 48000 Hz with a fallback list of rates; speech device index updated for PulseAudio.
- 06b0749: speech mic rate set to 44100.
Helper scripts `test_mic_rates.py` / `fix_alsa.py` were added to probe rates.

## How to verify
```bash
python test_mic_rates.py          # lists rates the mic accepts
journalctl -u airobot -n 200 | grep -i "invalid sample rate"   # expect none at startup
```

## Will it come back?
Yes if the USB mic is swapped for one with different native rates. The 2026-09-16 change (commit 231740d) made capture try the native 44.1k rate first to kill the remaining `paInvalidSampleRate` spam.
