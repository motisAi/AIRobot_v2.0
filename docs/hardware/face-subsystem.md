# Screen Face Subsystem (M0 done + she's alive)

Stella shows an animated face on an external screen **on voice command**, and hides it
to give the computer back. Currently running on the **Surface Pro** (dev/preview); will
move to a dedicated tablet later. Fully additive — a failure never affects the robot.

**Working now:**
- Show/hide by voice ("show your face" / "hide your face"), tolerant of STT mishears.
- **Drawn female face** (hair, skin, brows, eyes+pupils+lashes, nose, lips) — a full face,
  not floating features. (A photo "living-portrait" was tried and reverted — too crude.)
- **Emotions** — expression matches her real mood: happy/sad/curious when she reads your
  face on greeting, `alert` on a guard event, `listening` during a chat, `neutral` idle.
- **Eye-follow (gaze)** — a Pi thread (`_start_gaze_feed`) streams your detected face centre
  to the screen so her **eyes follow you** while the face is shown. Idle drift when no one's
  seen. Flip `MIRROR_X` in `main._start_gaze_feed` if the direction feels reversed.
- **Lip-sync (v1)** — her mouth moves whenever she speaks (real visemes can replace it later).
- Auto-blink so she never looks frozen.
- `GET /test` on the bridge = a visual self-test (go happy + talk 3s + relax).

## How it works
```
 You: "Stella, show your face"
   -> manager._maybe_face()  -> robot.face.show_face()   (Pi)
   -> face_bridge flips /display_state = {"show": true}
   -> face_client.ps1 on the Surface polls that, launches Edge --kiosk at the face URL
   -> the face page connects back over WebSocket and renders (eyes/blink/gaze/mouth)
 You: "Stella, hide your face"  -> show=false -> controller closes Edge.
```

## Pieces
| Where | File | Role |
|-------|------|------|
| Pi | `face_bridge/bridge.py` | aiohttp server on **:8080** — serves the face, WebSocket, `/display_state`, `/show`, `/hide`. Runs in its own thread. |
| Pi | `face_bridge/webface/index.html` | The face (2D canvas: glowing eyes, blink, idle gaze, mouth; emotion presets ready for M1). |
| Pi | `main.py` | Creates `self.face = FaceBridge(...)`, guarded. |
| Pi | `modules/conversation/manager.py` | `_maybe_face()` — the show/hide voice command. |
| PC | `face_client/face_client.ps1` | Controller: finds the Pi, polls, launches/kills Edge kiosk. |
| PC | `face_client/start-face-controller.cmd` | Double-click to run the controller. |

## Network-independent (home / work / anywhere)
The controller finds the Pi in this order, so a changing IP/subnet needs **no edits**:
1. **mDNS** — `motiAi.local` (verified working Surface↔Pi).
2. Cached last-good IP (`%LOCALAPPDATA%\stella-face-pi.txt`).
3. Quick scan of the current /24 for the bridge, then caches it.

Face URL (any network): **`http://motiAi.local:8080/`**.

## Running the controller
- **Background (recommended):** run `face_client/start-hidden.vbs` — no visible window.
- Visible (for debugging): `face_client/start-face-controller.cmd`.
- **Auto-start at login:** put a shortcut to `start-hidden.vbs` in `shell:startup`.
- It must be running for the voice command to open the face. It's the thing that opens/closes
  the face — it is NOT the face, and it does not open/close with the face. Keep it running
  (hidden) all the time; the **face window** (Edge) is what appears/disappears on command.
- **Single-instance:** a named mutex means launching it twice is harmless (the 2nd exits).
- **Foreground:** on show, the controller raises the face window to the front.
- **Manual close = hide:** if you close the face window yourself, the controller syncs to
  hidden instead of re-opening it (no more pop-back).
- **Hide** closes *any* Stella kiosk window, so an orphan from a crash gets cleaned up.

## Requirements / caveats
- PC and Pi on the **same LAN** (mesh/extender counts).
- Edge is used in `--kiosk --edge-kiosk-type=fullscreen` with its own throwaway profile.
- This is **M0** (link-up + a placeholder parametric face). Next: M1 realism
  (living-portrait look), then gaze (M2), lip-sync (M3). Emotion/gaze/viseme messages are
  already implemented in the page and the bridge API — they just aren't wired to Stella's
  events yet.

## Next-step idea (logged): use the PC's camera + mic
The Surface's 1080p cam + mics are better than the Pi's current ones. The face page can
capture them (`getUserMedia`) and stream to the Pi to improve face recognition / STT while
the face is shown. Deferred until after M1.
