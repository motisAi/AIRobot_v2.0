# Bug #009 — Wrong / shifting PyAudio mic device index after reboot

- **Date found:** 2026-05-31
- **Status:** fixed
- **Area:** audio
- **Files touched:** `config/settings.py`, `modules/audio/speech_recognition.py`, `modules/audio/wake_word.py`, `tools/check_deps.py`
- **Commit(s):** 9fecc95, f90451a

## Symptom
Speech capture opened the wrong device (or none) so commands were never heard, and the wake word was too insensitive; after a reboot the fix stopped working again.

## Root cause
Mic devices were addressed by hard-coded PyAudio index (config said 1, the real speech mic was 2), and PyAudio indices are re-assigned on every reboot, so any fixed index eventually points at the wrong card. Wake-word energy threshold was also too high.

## Fix
- 9fecc95: speech mic index corrected to 2; wake-word threshold lowered to 300; energy window 50.
- f90451a: mic devices resolved **by name** at runtime instead of by index.

## How to verify
Reboot the Pi twice; each time the journal shows the mic resolved by name and the wake word still fires.

## Will it come back?
Partially — the decisions log (2026-09-13) notes PyAudio enumeration is still flaky (sometimes lists only pulse/default); a clean `sudo -n systemctl restart airobot` re-resolves it. The 2026-09-16 `mic_available()` check (commit 231740d) looks at `/dev/snd/pcmC*D*c` to avoid being fooled by the name flip-flop.
