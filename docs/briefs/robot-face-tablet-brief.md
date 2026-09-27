# Robot Face Subsystem — Build Brief (for Claude Code)

## 0. Read this first

This document describes adding a **tablet-based animated face + display subsystem** to an
**already working** AI robot. The existing system is mature. **Do not rebuild, refactor, or
"improve" any working subsystem.** The face is a new, isolated module that plugs in through a
single clean boundary (a WebSocket bridge on the Pi) and adds a small number of emitter calls
at existing event points.

Before writing integration code, **inspect the existing repository** to find the real hook
points (where emotions/states change, where TTS is triggered, where Hailo face-detection
coordinates are produced, where the watchdog registers monitored components). Ask me for the
relevant files if they are not obvious. Assume nothing about internal names.

---

## 1. Existing system (context — DO NOT MODIFY)

- **Compute:** Raspberry Pi 5 + Hailo-10 AI accelerator. Ubuntu Server 24.04, headless, SSH.
- **Language:** Python. Multi-threaded design.
- **Vision (Hailo):** face recognition, object detection, color detection — all working.
- **Audio in:** 2 microphones + 1 camera (one mic mounted on the camera).
  - Mic A → wake-word, own thread.
  - Mic B → conversation/STT, own thread.
- **Actuation:** robotic arm via **ESP32**, mimics operator hand movements on command. Working.
- **Comms:** Telegram bot — sends and receives commands.
- **Reliability:** a **watchdog** that detects hardware faults, including WiFi loss, and can
  prompt the operator offline (after scanning networks) for where to reconnect.

**Non-goals (explicitly out of scope):**
- No changes to wake-word thread, conversation thread, Hailo pipeline, ESP32 arm control,
  Telegram logic, or watchdog internals — **except** two additive touches:
  1. add the tablet as a new **monitored node** in the watchdog, and
  2. call the face **emitter** API at points where emotion/gaze/speech already change.

---

## 2. New hardware: the face tablet

- **Device:** Teclast P30T (model TLC005).
- **SoC:** Allwinner A523, 8× Cortex-A55 @ 1.8 GHz. **GPU: Mali-G57 MC1 (single core).**
- **RAM: 4 GB real.** (Vendor "10/15 GB" is virtual RAM — ignore it.)
- **Display: 10.1" IPS, 1280×800**, Android 14, TDDI fully-laminated.
- **Battery:** 6000 mAh (acts as a built-in UPS for the face).
- **Radios:** WiFi 2.4/5 GHz + BT 5.4. **No cellular** (keep any SIM7600 remote channel on the robot side, not the tablet).
- **Mics:** dual mic with ANS + AEC (good). **Front camera: 2 MP (weak).**

### Hard constraints this imposes on the tablet app
- **2D Canvas only.** No WebGL/3D, no heavy engines. Mali-G57 MC1 + 4 GB RAM is modest.
- Target **30 fps sustained**; degrade gracefully if thermals throttle.
- **Vanilla JS or a tiny lib.** No React/Vue/Angular, no large bundles. Keep DOM tiny.
- Use a single `<canvas>` + `requestAnimationFrame`. Avoid per-frame allocations.
- **Face-position tracking does NOT use the tablet camera** (too weak). Gaze targets are
  computed on the Pi from Hailo and streamed to the tablet.

---

## 3. Architecture

```
                         WiFi / LAN
  ┌──────────────────────┐   WebSocket (ws://)   ┌───────────────────────────┐
  │  Raspberry Pi 5       │◄─────────────────────►│  Tablet (P30T)            │
  │  + Hailo-10           │                       │  Kiosk web app = FACE     │
  │  Python orchestrator  │   Pi = server         │  Canvas 2D: eyes/mouth/   │
  │  (existing)           │   Tablet = client     │  brows, gaze, lip-sync    │
  │                       │                       │  + optional audio (ph.2)  │
  │  + face_bridge (NEW)  │                       └───────────────────────────┘
  └──────────┬────────────┘
             │ UART/USB
             ▼
        ESP32 → robotic arm + (future) sensors
```

**Role split**
- **Pi = brain.** All AI, STT/TTS, decisions, gaze source (Hailo), arm control, watchdog,
  Telegram — unchanged. Emits high-level face commands.
- **Tablet = face + screen.** Pure renderer. Receives commands, draws expressions, tracks a
  gaze target, animates the mouth from a viseme/amplitude stream, reports lightweight telemetry.

**Why serve the web app FROM the Pi:** to avoid https/ws mixed-content headaches, the Pi
serves the static face app over **plain HTTP** on the LAN, and the tablet opens it in kiosk.
The app then talks `ws://<pi-host>:<port>` **same-origin** — no mixed-content restriction,
no TLS to manage on a LAN device. (Alternative: package the app locally on the tablet and
point it at the Pi IP; same-origin HTTP is simpler.)

---

## 4. Communication protocol (WebSocket, JSON messages)

One WS connection. Messages are line-delimited JSON objects with a `type` field.
Keep it small and forward-compatible (unknown fields ignored, unknown types logged & dropped).

### 4.1 Pi → Tablet (commands to the face)

| type          | fields                                                        | meaning |
|---------------|--------------------------------------------------------------|---------|
| `emotion`     | `value` (enum), `intensity` 0–1, `transition_ms` (opt)       | set expression preset |
| `look_at`     | `x` −1..1, `y` −1..1                                          | gaze target (normalized; from Hailo) |
| `gaze_mode`   | `value`: `track`\|`idle`\|`center`\|`scan`                    | how eyes behave when no explicit target |
| `speak_start` | `duration_ms` (opt), `text` (opt caption)                    | begin talking animation |
| `viseme`      | `level` 0–1                                                   | streamed mouth-openness frame (~30–60/s) |
| `speak_end`   | —                                                            | stop talking, close mouth |
| `blink`       | —                                                            | manual blink (tablet also auto-blinks) |
| `state`       | `listening` bool, `wake_detected` bool, `busy` bool, ...     | ambient status → subtle cues |
| `display`     | `mode`: `face`\|`camera_view`\|`info`\|`qr`, `payload` obj   | switch what the screen shows (ph.2 for non-face) |
| `ping`        | `ts`                                                         | heartbeat |

`emotion.value` enum: `neutral`, `happy`, `curious`, `thinking`, `listening`, `confused`,
`sad`, `alert`, `sleep`, `offline`.

### 4.2 Tablet → Pi (telemetry / events)

| type     | fields                                                    | meaning |
|----------|----------------------------------------------------------|---------|
| `hello`  | `device`, `app_version`, `screen`:[w,h]                   | on connect |
| `pong`   | `ts`                                                      | heartbeat reply |
| `status` | `battery` 0–100, `charging` bool, `fps`, `heap_mb` (opt)  | periodic (e.g. every 5 s) |
| `touch`  | `region` (e.g. `left_eye`,`menu`), `x`, `y`              | user tapped the face |
| `imu`    | `accel`:[x,y,z], `gyro`:[x,y,z], `ts`                    | phase 2: motion/orientation/fall |
| `error`  | `code`, `msg`                                             | app-side error |

### 4.3 Heartbeat & resilience (mirror the existing watchdog philosophy)
- Pi sends `ping` every ~3 s; tablet replies `pong`. Missing N pongs ⇒ Pi watchdog marks the
  **face node** unhealthy (new monitored component, additive).
- Tablet auto-reconnects with backoff; while disconnected it shows the `offline` emotion so the
  robot visibly "knows" it lost its face.
- Tablet `status.battery` low ⇒ Pi can raise a normal watchdog/Telegram alert.

---

## 5. Pi side — `face_bridge` module (NEW, additive)

```
face_bridge/
  __init__.py
  server.py     # asyncio WebSocket server, runs in its OWN thread
  emitter.py    # FaceBridge: thread-safe API the existing orchestrator calls
  visemes.py    # compute amplitude envelope from a TTS audio buffer
  http_static.py# tiny static file server for the tablet web app (LAN, HTTP)
```

**Threading model (respect existing design):** run the asyncio WS server in a dedicated thread.
The orchestrator (in its own threads) calls a **thread-safe** `FaceBridge` API that pushes
messages onto an `asyncio.Queue` via `call_soon_threadsafe`. No blocking of existing threads.

**Emitter API the orchestrator calls (the entire integration surface):**
```python
face = FaceBridge(host="0.0.0.0", ws_port=8765, http_port=8080)
face.start()                      # starts WS + static HTTP server threads

face.set_emotion("curious", intensity=0.8)   # at existing state-change points
face.look_at(x, y)                            # feed normalized Hailo face coords
face.gaze_mode("track")                       # or "idle"/"scan" when no face seen
face.set_state(listening=True, wake_detected=False)

# lip-sync: wrap the existing TTS playback
with face.speaking(text="Hello", audio_pcm=pcm_bytes, sample_rate=22050):
    play_audio(pcm_bytes)         # existing playback; visemes stream automatically
```

**`visemes.py`:** compute a smoothed RMS envelope from the TTS PCM buffer, emit `viseme`
frames at ~40 fps between `speak_start`/`speak_end`. (Envelope-driven mouth is the v1 approach;
true phoneme visemes are optional later.)

**Integration points to find in the existing repo (ask if unclear):**
1. where emotion/robot mood is decided → `set_emotion`
2. where Hailo yields the primary face bbox/center → normalize to −1..1 → `look_at` / `gaze_mode`
3. where TTS audio is produced and played → wrap with `face.speaking(...)`
4. where the watchdog registers monitored components → add the face node + heartbeat status

---

## 6. Tablet side — kiosk web app (NEW)

```
face-app/
  index.html
  css/face.css
  js/
    main.js          # boot, canvas + RAF loop, wiring
    ws.js            # connect, reconnect w/ backoff, heartbeat, message dispatch
    face.js          # draw eyes, brows, mouth; owns the render loop
    expressions.js   # emotion presets (eye shape, brow angle, color, mouth curve)
    gaze.js          # smooth look_at interpolation; idle micro-saccades; scan mode
    visemes.js       # mouth openness from viseme level, with attack/decay smoothing
    diagnostics.js   # fps meter, battery (navigator.getBattery), status reporter
  assets/
```

**Rendering guidance**
- Single canvas sized to devicePixelRatio-aware 1280×800; clear + redraw per frame.
- Eyes: parametric (pupil position, lid openness, shape morph per emotion). Pupils lerp toward
  the gaze target; add subtle idle drift + occasional micro-saccades so it never looks frozen.
- Auto-blink every 3–6 s (randomized), faster when `listening`.
- Mouth: openness driven by latest `viseme.level`, smoothed (fast attack, slower decay);
  fully closed on `speak_end`/silence.
- Emotion transitions: interpolate parameters over `transition_ms` (default ~250 ms).
- **Perf:** no per-frame object allocation; precompute gradients; cap DPR if fps drops;
  pause rendering when `display.mode !== "face"` if a static screen is shown.

**Connection behavior**
- On load: `hello`. Reconnect with exponential backoff (cap ~5 s). Reply to `ping` with `pong`.
- On disconnect: show `offline` emotion (e.g., dimmed, half-lidded eyes) so the loss is visible.
- Send `status` every ~5 s (battery, charging, fps).

---

## 7. Kiosk setup on the P30T (Android 14)

- Keep screen awake: prevent sleep (kiosk app setting or `WakeLock`); set display timeout to never while docked/charging.
- Fullscreen, no nav/status bar, disable gestures. Recommended: **Fully Kiosk Browser**
  (autostart on boot, load URL, keep-awake, motion/PIR options). Alternative: Chrome PWA fullscreen.
- **URL:** point kiosk at `http://<pi-host>:8080/` (app served by the Pi). App opens
  `ws://<pi-host>:8765` same-origin.
- Autostart the kiosk app on boot; auto-reload the page on crash.
- Mount the tablet as the robot's face; keep it on a charger (battery = UPS, but avoid full drain).

---

## 8. Build order (milestones + acceptance criteria)

- **M0 — Link up.** Pi serves the app; tablet loads it in kiosk; WS connects; heartbeat + auto-
  reconnect. A static face renders.
  *Accept:* reboot the tablet → face returns and reconnects within ~10 s; kill/restore WiFi →
  recovers automatically; Pi logs the face node up/down.
- **M1 — Expressions.** All emotion presets render with smooth transitions; auto-blink + idle
  micro-movements.
  *Accept:* a Pi test script cycles every emotion; 30 fps sustained; transitions look smooth.
- **M2 — Gaze.** Pi forwards Hailo face center → eyes track; `idle`/`scan` when no face.
  *Accept:* operator moves in front of the robot → eyes follow with low perceived lag; smooth,
  no jitter; graceful idle when the face leaves frame.
- **M3 — Lip-sync.** `speak_start` → `viseme` stream → `speak_end`; mouth moves with speech and
  closes on silence.
  *Accept:* spoken output looks synced; mouth doesn't move during silence.
- **M4 — Resilience + telemetry.** Battery/fps `status` to Pi; `offline` face on disconnect;
  face node integrated into the watchdog; `touch` events delivered.
  *Accept:* pull WiFi → tablet shows offline face, Pi watchdog flags the node, both recover;
  low battery raises a normal alert.

### Phase 2 (later, optional)
- Move **TTS audio playback to the tablet** (sound comes from the face → better AV-sync). Pi
  sends audio + timing; tablet plays and lip-syncs locally.
- **IMU telemetry** (`imu`) as another sensor into the watchdog: detect pushed/moved/fallen.
- **Display modes:** `camera_view` (show what Hailo sees), `info` dashboard, `qr` for quick setup.
- **Telegram → face**: trigger expressions/messages on the face via existing Telegram commands
  (trivial once the emitter exists).

---

## 9. Assumptions & decisions (OVERRIDE ANY OF THESE)

1. **Transport = WebSocket**, Pi = server, tablet = client, JSON messages. *(Alt: MQTT if you
   already run a broker — say so and I'll swap the bridge.)*
2. **Gaze source = Hailo on the Pi**, not the tablet's 2 MP camera.
3. **Lip-sync = amplitude envelope** streamed from the Pi; **audio stays on the Pi in v1**.
4. **Face style = expressive robot eyes in 2D Canvas.** *(Alt: anime/large-eye style — one flag.)*
5. **App served over HTTP from the Pi**, `ws://` same-origin (no TLS on the LAN).
6. **Pi orchestrator = Python, threaded**; the bridge runs asyncio in its own thread with a
   thread-safe emitter — no changes to existing threads.
7. Tablet app = **vanilla JS + Canvas 2D**, 30 fps target, no heavy frameworks.

If any of these is wrong for your setup, change it here before implementation — everything
downstream (module layout, protocol) follows from these choices.
