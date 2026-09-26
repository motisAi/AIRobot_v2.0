# Bug #022 — Face kiosk never opened on the PC (controller not running, then wedged instances)

- **Date found:** 2026-09-12 (second episode 2026-09-13)
- **Status:** fixed
- **Area:** face-ui
- **Files touched:** Windows-side `face_client.ps1`, Startup shortcut `StellaFace.lnk`, `%LOCALAPPDATA%\stella-face-pi.txt` (none in the Pi repo)
- **Commit(s):** n/a (Windows files, outside the repo)

## Symptom
"Show your face" was acknowledged but no face window appeared on the Surface.

## Root cause
- 2026-09-12: the Pi bridge correctly set face -> SHOW, but the Windows controller `face_client.ps1` was not running and had no auto-start.
- 2026-09-13: multiple wedged instances of the controller were fighting over the single-instance mutex, so none of them acted.

## Fix
- Started the controller, pre-seeded the Pi IP cache (`stella-face-pi.txt`), added a Startup shortcut (`StellaFace.lnk`) so it runs at login.
- Rewrote `face_client.ps1` as a stateless bridge-tracker with a log file at `%TEMP%\stella-face.log`.

## How to verify
On the PC: `Get-Content $env:TEMP\stella-face.log -Tail 20`. Say "show your face" -> Edge kiosk window "Stella" opens; "hide your face" -> it closes. `curl http://<pi>:8080/display_state` shows `show` toggling.

## Will it come back?
Possible if the Startup shortcut is removed or the PC's login session is not active; the log file is the first thing to check.
