# Decision Log

Newest first. Each entry: what we decided, why, and what we rejected. Keep it short.

---

### 2026-09-12 — Sensibo AC control added
**Decision:** Stella controls the Sensibo AC via the **Sensibo cloud REST API**
(`modules/hardware/sensibo.py`, key in `.env: SENSIBO_API_KEY`). Voice handler `_maybe_ac`
in the manager (runs before generic device control): power on/off, temperature (17–30),
mode cool/heat/fan/dry/auto ("make it cold/warm"), and unit selection by room. Two units
found: "סלון" (Living room, default) + "moti's device". Verified on/off works.
**Why cloud not local:** Sensibo has no local API — control is via their cloud, so it needs
internet (works remotely too) but NOT offline. Different from the Tuya plug (local/LAN).
**Gotcha:** Sensibo **rate-limits** hard (429 on rapid calls) — the module now **caches the
acState** so each command is a single API call; still, space voice commands a few seconds
apart. **Keys** stay in `.env` (gitignored).

---

### 2026-09-12 — Smart-home control: MQTT hub + first Tuya device (LSPA8 plug)
**Decision:** Stella is now a home-control hub. Two control paths, checked in order
**Tuya → MQTT → wired-MCU** by both the `control_device` tool and the keyword path
("turn on/off the X"):
- **MQTT (mosquitto on the Pi, :1883):** for local devices (Tasmota/ESPHome/Sonoff) on
  RobotNet. `modules/hardware/mqtt_devices.py`, config `mqtt.devices` (name→topic). Broker via
  `~/setup_mqtt.sh`. Sample ESP32 relay sketch is in the RobotNet notes.
- **Tuya (tinytuya, local key):** for cloud plugs like the **LSPA8** without reflashing.
  `modules/hardware/tuya_devices.py` loads `devices.json` (from `python -m tinytuya wizard`,
  which needs a free Tuya IoT project + Smart Life app link). Verified: "Smart Plug" @
  192.168.11.195, v3.5, DP 1 = switch — Stella turns it on/off. Matches spoken "plug".
**Security:** `devices.json`/`snapshot.json`/`tinytuya.json` added to `.gitignore` (they hold
local keys — never commit).
**Caveats:** (1) plug's IP is DHCP — if it changes, re-run the scan (or reserve its IP in the
router). (2) Device control requires Stella to know you're the master (face auth). (3) Tuya
plug stays on home WiFi → home control only; move it to RobotNet + it works portably too.
**Next:** flash ESP32 relays (RobotNet + MQTT) for portable/offline switches; add more devices.

---

### 2026-09-12 — RobotNet (portable AP) WORKING, on the USB dongle
**Decision:** Pi runs its own WiFi network **"RobotNet"** (10.0.0.1/24, WPA2, DHCP .10–.50,
NAT internet-sharing) so peripherals (ESP32s, drone, dog, cams) connect to the robot, not a
home router — travels with it, works offline.
**Key finding:** the Pi's **onboard Broadcom can't do AP+STA with real clients** — it broadcasts
but clients never associate (hostapd logs nothing). So AP-on-onboard-while-on-home-WiFi is out.
**Solution:** dedicate the **USB dongle (TP-Link Archer T2U Plus, RTL8821AU)** to the AP; onboard
`wlan0` stays the home-WiFi client (internet + SSH untouched, no lockout).
**Driver:** the apt `rtl8812au-dkms` (2014) is the WRONG driver — doesn't bind the RTL8821AU.
Correct one = **morrownr `8821au` DKMS** (`github.com/morrownr/8821au-20210708`, install-driver.sh).
It supports AP mode (`iw list` → `* AP`). Dongle comes up as `wlx984827df22bb`.
**Verified:** phone joined, got 10.0.0.40, -34 dBm, reached dashboard :5000 + face :8080.
**Setup:** `~/setup_robotnet.sh` (idempotent; `--remove` to undo). Channel 1 (home is ch8).
**Gotchas:** (1) morrownr driver is DKMS → **rebuild after kernel updates** (like Hailo:
`sudo dkms autoinstall` after headers). (2) When away from home (no wlan0 uplink) the AP still
serves devices; internet just isn't shared (Stella's offline Hailo brain covers AI).
**Next:** flash ESP32s with ssid="RobotNet" pass="Aa123456" → they reach Stella at 10.0.0.1.

---

### 2026-09-05 — Gemini added (4-brain chain) + face recognition fixed
**Gemini:** `.env` had `GEMINI_API_KEY=` **blank** and the model was retired. Set the key
and updated the model to **`gemini-3.6-flash`** (2.0-flash returns 404 now). Chain is now
`groq(gpt-oss-120b) → groq_fast(gpt-oss-20b) → gemini(gemini-3.6-flash) → hailo(qwen1.5b)`.
Gemini is a **separate provider with its own quota**, so it covers Groq rate-limits without
touching Groq's budget. (Gemini uses the OpenAI-compatible endpoint — no extra library.)

**Face recognition:** root cause was NOT enrollment — the **threshold 0.50 was too strict**
for dlib (its norm is ~0.6) so valid matches were rejected as "Unknown," and the displayed
`similarity = 1 - dist/threshold` made good matches look like ~0.08. Fixes: re-enrolled here,
raised `face_recognition_threshold` to **0.60**, changed the score to `max(0, 1 - distance)`.
Live result went from ~0.06 to **~0.65–0.68** (reliable). Tool: `reenroll.py` (dlib/HOG
detection + hard 40s cap so it can't hang like `enroll_master.py --auto` did; speaks guidance
so you can face the camera without reading the screen).

**Rejected — TTS through `pulse`:** tried routing her voice through pulseaudio to kill the rare
`aplay 524 busy` drops. It works when run from a *user* SSH session but the **systemd service
can't reach the user's pulseaudio** (`Connection refused`), which broke her speech. Reverted to
the original direct-HDMI (`sysdefault:CARD=vc4hdmi0`) — mostly reliable, rare 524 (watchdog-
handled). NOTE: `reenroll.py` (run as the user) *can* use `pulse` for its spoken guidance.
Proper future fix for the 524: give the airobot systemd unit pulseaudio access (XDG_RUNTIME_DIR/
PULSE_SERVER) or stop pulseaudio grabbing the HDMI card.

---

### 2026-09-05 — Maximized free AI: gpt-oss-120b cloud + revived Hailo offline brain
**Decision:** Use the strongest free tools. **Cloud primary → `openai/gpt-oss-120b`** on Groq
(120B, ~480 tok/s, reasoning kept out of `content`), fallback `openai/gpt-oss-20b`; `max_tokens`
512. **Why:** `llama-3.3-70b-versatile` was decommissioned (404); gpt-oss-120b is the biggest/
fastest model the account can use. No Gemini/OpenAI/Anthropic keys present (and Claude is
un-authorized by the boss), so those chain slots are inert.
**Hailo revived:** the kernel had been upgraded to `6.8.0-1060-raspi` and the `hailo1x_pci`
DKMS driver wasn't rebuilt → no `/dev/hailo0`. Fixed by installing headers + `dkms autoinstall`
+ `modprobe`. Then fixed the app to call hailo-ollama's **`/v1/chat/completions`** (the only
endpoint 5.1.1 serves; `/api/chat` returns empty) — so the **offline brain (qwen1.5b) now
actually works** and is warmed at startup. Full chain:
`groq(gpt-oss-120b) → groq_fast(gpt-oss-20b) → hailo(qwen1.5b, offline)`.
**Rejected:** `llama3.2:3b` on the NPU (too slow); Hailo object-detection for now (would
contend with the LLM for the single NPU context; vision stays on cloud Moondream);
**`groq/compound` web-search fallback** — tested and reverted: on the free tier it returns
`request_too_large` (very token-heavy) AND shares Groq's rate-limit budget with the main
gpt-oss-120b brain, so it could starve her primary brain. Kept the existing DuckDuckGo
`web_search` (reliable, no Groq budget cost). Note: urllib needs a browser `User-Agent`
or Groq returns 403 — relevant if compound is ever revisited on a paid tier.
See `../hardware/hailo.md` (incl. the kernel-upgrade fix that WILL recur).

**Best remaining FREE upgrade (needs a key):** a **Gemini** key (Google AI Studio, 2-min,
free) — it's a *separate* provider with its own quota, so unlike compound it wouldn't compete
with Groq. Add `GEMINI_API_KEY` to `.env` and it slots into the existing `groq → gemini →
hailo` chain automatically.

---

### 2026-08-16 — Face = drawn character (not a photo) + eye-follow
**Decision:** Reverted the photo "living-portrait" (looked crude/uncanny with canvas
overlays) back to a **drawn, animated female face** — full face now (hair, skin, brows,
eyes with moving pupils + lashes, nose, lips), emotions, blink, and lip-sync. Added a
**gaze feed**: a thread streams the primary detected face's centre from the Pi camera to the
screen face so her **eyes follow you** (mirror X; flip `MIRROR_X` in `main._start_gaze_feed`
if it feels reversed). **Why:** convincing photoreal needs a neural talking-head (not
feasible on the tablet); a polished drawn face reads better and is fully controllable.
**Rejected:** static-photo + overlay warping (uncanny). Photoreal parked for a future
GPU/AI-module upgrade. `face.png`/`face_meta.json` left on the Pi, unused.

### 2026-08-16 — Anthropic key present but NOT wired yet (pre-demo safety)
**Decision:** `.env` now has an `ANTHROPIC_API_KEY` slot, but the provider chain is still
`groq -> groq_fast -> hailo`. Did not wire Claude in before the boss demo — adding an
untested provider is a risk. Ready to wire when we can test it calmly.

---

### 2026-08-16 — On-demand face on the Surface, found via mDNS (M0 done)
**Decision:** Show/hide the face by voice ("show/hide your face"). A tiny PowerShell
controller on the PC polls the Pi's `/display_state` and launches/kills Edge kiosk.
Use the Surface Pro as the dev/preview screen for now (it's the user's main PC; on-demand
means it doesn't hog the machine). **Network:** the controller finds the Pi via
**`motiAi.local`** (mDNS, verified), with cached-IP + subnet-scan fallbacks, so a changing
IP/subnet needs no edits. **Why:** the Surface is far more capable than the P30T and already
here; mDNS kills the hardcoded-IP problem. See `../hardware/face-subsystem.md`.

---

### 2026-08-15 — Face look: "living portrait" (realistic 2D), not 3D
**Decision:** Aim for "a real person on a screen" via a 2D **living portrait** — a realistic
human photo base with canvas-animated eyes (gaze + blink), viseme mouth swaps for lip-sync,
and expression crossfades between a few photos of the same person.
**Why:** true real-time photoreal 3D is impossible on the P30T (Mali-G57 MC1, 4 GB); this reads
as human and runs at 30 fps. **Fallback:** semi-realistic stylized human if consistent assets
are hard. **Note:** the WS protocol is identical regardless of look — realism is pure tablet-side
rendering + assets, no Pi changes.

### 2026-08-15 — Bridge on a single port (8080), ws path /ws
**Decision:** Serve the face app over HTTP and the WebSocket on the **same port 8080**
(`ws://<pi>:8080/ws`) using aiohttp. **Why:** same-origin, no mixed-content, no TLS on the LAN,
one thing to run. Deviates from the brief's separate 8765/8080 for simplicity.
**Libs already present:** `websockets`, `aiohttp`, `flask` — no new installs.

### 2026-08-15 — Vision stays on the Pi camera; tablet cameras deferred
**Decision:** Gaze + all vision use Stella's **existing USB camera + Hailo**; the tablet's front
camera is unused (2 MP, faces outward). The tablet's **back camera** is a logged **phase-2** idea
(rear/"behind me" view or on-demand still, streamed to the Pi). **Why:** matches the brief; keeps
v1 scope tight.

### 2026-08-15 — Project docs live in the repo on the Pi
**Decision:** Canonical design docs go in `AIRobot_v2.0/docs/` (versioned, pushed to GitHub),
not on the PC. **Why:** survives a PC wipe, sits next to the code, understandable long-term.
**Workflow:** drop files on the PC → Claude files them into `docs/` on the Pi.

### 2026-08-15 — Emotion-aware greeting via cloud VLM (not a local model)
**Decision:** Read facial emotion with the existing Moondream VLM on master re-appearance,
rate-limited to once/10 min, and open a mood-appropriate conversation.
**Why:** reuses existing infra, no new heavy CPU model in the camera path (stability), free tier.
**Rejected:** a local ONNX/TF emotion model — more integration risk on top of MediaPipe + dlib.

### 2026-08-15 — Tablet face = additive subsystem over WebSocket (brief received)
**Decision (proposed, not built):** add a tablet face (Teclast P30T) as an isolated renderer;
Pi = WS server + static HTTP app; tablet = 2D-Canvas client. No changes to existing threads
beyond two additive touches (watchdog node + face emitter calls).
**Why:** keeps the mature system untouched; single clean boundary. See
`../briefs/robot-face-tablet-brief.md`. **Open:** transport (WS vs MQTT), face style.

### earlier — LLM chain
**Decision:** Groq `llama-3.3-70b-versatile` primary → Groq `openai/gpt-oss-20b` fallback →
offline Hailo (currently broken, disabled). Anthropic Claude key pending → to be added to the chain.
**Why:** GPT-OSS-20B replaced the decommissioned `llama-3.1-8b-instant`; offline Hailo LLM is
unstable (`HAILO_OUT_OF_PHYSICAL_DEVICES`).

### earlier — Hand controller = ESP32-S3 + PCA9685, external servo power
**Decision:** ESP32-S3-WROOM-1 over USB serial; PCA9685 for PWM; servos on a separate 5–6 V
supply with common ground; I2C on GPIO 8/9. **Why:** classic ESP32 shorted from a loose jumper;
GPIO22 doesn't exist on the S3; servos must not draw from the Pi.

---

### 2026-09-12 — STT: Groq Whisper primary; single-plug generic naming
**Decision:** Speech-to-text now tries **Groq Whisper (whisper-large-v3-turbo)** first, then
Google, then offline Vosk (`modules/audio/speech_recognition.py::_transcribe_groq` +
`_transcribe_pcm16k`). Uses the existing `GROQ_API_KEY` (free), multipart POST to
`/openai/v1/audio/transcriptions`. **Why:** Google STT failed 57x in one session and dropped to
weak Vosk, which mishears -> Stella appeared "stuck"/unresponsive. Verified: espeak "turn on the
light" -> Groq -> "Turn on the light." round-trips.

**Also:** Tuya `_match` gives a **single** device generic on/off names (light, lamp, socket,
switch, relay1, power, outlet, device). **Why:** "turn on the light" was routed to a
non-existent microcontroller relay because no device was literally named "light"; now the one
plug answers to any common on/off word. Combined with the flexible `set_ac` tool + intent-based
`control_device` description, Stella acts on meaning ("it is hot", "kill the light"), not exact
keywords.

---

### 2026-09-12 (later) — The real "stuck": double greeting + welcome spam
**Correction to the STT entry above:** the "stuck" was NOT the STT. Vosk was
transcribing full sentences fine ("it is hot can you turn on the air conditioner").
The real cause: TWO subscribers greeted the master on every face event —
`core/robot_brain.py` (spoken "Welcome back, {name}!") AND
`main._welcome_master` (wave + emotion opener + listening session) — so she
greeted twice, and the on-sight cooldown was left at demo values (gap>20,
since_welcome>60) so she re-welcomed every ~30-60s. Overlapping TTS then hit
`aplay: Device or resource busy` and she looked frozen.
**Fix:** brain no longer speaks the master greeting (main owns it); on-sight
cooldown restored to gap>90 & since_welcome>240. STT change (Groq Whisper) kept
as a genuine improvement for online accuracy.

---

### 2026-09-12 (later) — Camera "off" after reboot: CSI camera renumbered the USB webcam
**Symptom:** after a reboot, "Camera opened but test read failed", no face detection,
Stella unresponsive to sight. **Cause:** the Pi CSI camera (added to the CSI slot)
claimed /dev/video0-7 (rp1-cfe-csi2 / pispbe nodes — NOT readable by OpenCV),
pushing the working USB "Auto Focus Camera" from video0 to video8. The hard-coded
config index 0 then opened a CSI node that never yields a cv2 frame.
**Fix (`modules/hardware/camera_manager.py`):** camera open now AUTO-DETECTS the
USB webcam — probes indices ordered [configured, USB-UVC-by-name, 0..15], skipping
CSI/ISP nodes (rp1-cfe/pispbe/rpivid), opens with MJPG, and warm-reads up to ~2.5s
(USB cams fail the first reads). Survives node renumbering across reboots; `_reopen`
re-probes too. Verified: auto-detected index 8, "Face detected: Moti".
**Note:** the Pi CSI camera needs libcamera/picamera2 (not cv2) — wire that up later
if we want to switch to it. Also: onboard WiFi uplink did not reconnect on this boot
(`no internet` until `sudo netplan apply`) — boot auto-reconnect still TODO.

---

### 2026-09-12 (later) — Guard auto-disarmed instantly + face kiosk never opened
**Guard:** "guard my house" armed guard, but ~6s later the master face was
recognised -> "auto-disarming guard". You cannot arm guard while standing in
front of the camera. **Fix (main.py):** auto-disarm ("welcome home") now only
fires when the master RETURNS after being away (gap since last seen > 60s);
arming while present no longer disarms. was_armed welcome-home messaging gated on
an actual disarm.

**Face kiosk "did not open":** the Pi bridge sent face->SHOW correctly, but the
Windows face controller (face_client.ps1) was NOT running and had no auto-start.
**Fix (Windows):** started the controller, pre-seeded the Pi IP cache
(stella-face-pi.txt), and added a Startup shortcut (StellaFace.lnk) so it
auto-starts at login. Verified end-to-end: /show launched the Edge kiosk
("Stella" window), /hide closed it.

**Noise, not a bug:** repeated `paInvalidSampleRate` ALSA lines are the capture
code trying 16k first (USB mic rejects it) then falling back to 44.1/48k and
resampling — harmless. STT itself (Groq Whisper) transcribed full sentences
well; natural-language AC/light/face/vision commands all worked.

---

### 2026-09-12 (later) — Music: right song + it actually plays now
Two bugs behind "find and play music... not":
1. **Wrong result:** ytsearch1 took the literal top hit, so vague queries played
   guitar tutorials, covers, or unrelated videos ("A shocking incident for a
   nomadic family..."). Fixed: fetch 5 candidates, filter junk
   (tutorial/lesson/cover/tab/reaction/karaoke/backing), prefer official / "- Topic"
   channels + query-word matches (music.py _search_candidates/_pick_best/_pick_target).
2. **No audio at all:** YouTube now 403s the raw stream URL handed to ffmpeg
   (PO-token requirement for the default web client) — every play silently failed
   ("it's not playing"). Fixed: yt-dlp fetches with the ANDROID player client
   (no PO token) and PIPES audio into ffmpeg (yt-dlp does the HTTP with correct
   headers). Verified: continuous playback, correct titles (Eagles/Queen/Pink Floyd).
   Also installed deno 2.9.6 to ~/.deno (yt-dlp JS runtime) via python unzip since
   apt/unzip were not available; the android client works even without it, but it is
   there for future extractor needs.

**Not bugs, just funny:** "someone at the door" = the LLM called the vision tool
with 'who is at the door?' and the VLM described whoever it saw (you) — she is
looking through the USB webcam at the room, there is no door. "Replies to nothing"
= Whisper hallucinating short phrases ("Thank you.", ".") from silence/noise
between utterances. Can tighten with a min-voiced-duration gate if it annoys.

---

### 2026-09-12 (later) — Console "alerts": CSI camera kernel spam
Screen showed repeating "rp1-cfe 1f00128000.csi: csi2_chN node link is not
enabled." The Pi CSI camera is plugged in but not configured (no libcamera
pipeline), so opening its /dev/video0-7 nodes makes the kernel print that to the
console. Our probe tried the configured index (0 = a CSI node) FIRST on every
camera start, so each service start/boot emitted one line. The apparent flood was
mostly today's own repeated service restarts while debugging music (NRestarts=0 —
not a crash loop).
**Fix (camera_manager._candidate_indices):** CSI/ISP nodes are now excluded
entirely (by /sys name); candidate order is USB webcams first, CSI never opened.
Verified: candidates = [8,9,10,...], 0 new kernel messages across a restart.
Lines already on the console are static — `clear` or a reboot wipes them.
