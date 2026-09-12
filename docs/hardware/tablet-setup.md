# Tablet Face — Setup Checklist (Teclast P30T)

Do these once the tablet is charged (~50%+ is plenty). Goal: the tablet boots straight
into Stella's face, fullscreen, and reconnects on its own.

## Facts
- **Pi (face server):** IP `192.168.11.204`, hostname `motiAi`.
- **Face URL (once the bridge is running):** `http://192.168.11.204:8080/`
- Same-origin WebSocket: `ws://192.168.11.204:8080/ws` (one port, no TLS on the LAN).

## Steps
1. **WiFi:** put the tablet on the **same network** as the Pi (2.4 or 5 GHz both fine).
   Confirm it can reach the Pi: open a browser on the tablet → `http://192.168.11.204:8080/`
   (only works after Claude starts the bridge — see M0).
2. **Kiosk browser:** install **Fully Kiosk Browser** (Play Store). It gives: autostart on
   boot, load-a-URL, keep-screen-awake, auto-reload on crash, hide nav/status bars.
   - Start URL: `http://192.168.11.204:8080/`
   - Enable: *Start on boot*, *Keep screen on*, *Fullscreen / hide system bars*, *Auto-reload on connection loss*.
   - Alternative if you prefer: Chrome → "Add to Home screen" as a fullscreen PWA.
3. **Never sleep while docked:** set display timeout to never (or rely on Fully's keep-awake).
   Keep it on a charger — the 6000 mAh battery then acts as a UPS for the face.
4. **Mount** the tablet as the robot's face. (Back camera will point outward — reserved for a
   possible phase-2 rear view.)

## What Claude does on the Pi (no action from you)
- Runs the `face_bridge` (serves the face app on :8080 + the WebSocket).
- Later: wires the bridge into Stella so emotion/gaze/speech drive the face automatically.

## Verify (M0 acceptance)
- Reboot the tablet → face reappears and reconnects within ~10 s.
- Toggle WiFi off/on → face shows "offline", then recovers on its own.
- Pi log shows the face node connect/disconnect.
