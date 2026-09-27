# Decision Log

> **Entries before 2026-09-19 reference the pre-reorganisation paths (`modules/hardware/*`).** The current layout is in
> [../architecture/stella-architecture.md](../architecture/stella-architecture.md) §11; the canonical audio design is §4.

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

---

### 2026-09-13 — Voice dead + phantom replies: Vosk can't hear "Stella"
**Symptom:** after reboot + CSI-unplug, no voice response; guard uncontrollable
(guard is armed/disarmed by voice); "replies to nothing".
**Root cause (voice):** the small Vosk EN wake model renders "stella" as
settler / taylor / stellar / sella / "that last" / live — never "stella". The
matcher only accepted "stella"/"hey stella" or tokens starting "stel", so the
wake word NEVER matched -> she never woke. (It worked before only by luck.)
**Fix (wake_word._matches_wake):** accept a measured alias set (settler, taylor,
stellar, steller, sella, estella, sailor, ...) + a Levenshtein<=2 fallback to the
keyword. Verified live: "Hey Stella" -> wake detected (conf 0.90) -> she greeted
and answered normally.
**Root cause (phantom replies):** Whisper/Vosk hallucinate short phrases
("Thank you.", ".") from ambient noise, which kept conversations alive forever.
**Fix (speech_recognition.capture_utterance):** discard captures below rms 220 or
under 0.35s before transcribing. No more phantom loop.
**Notes:** PyAudio device enumeration is flaky (sometimes shows only pulse/default,
sometimes the hw cards) — a clean service restart re-resolves it. Vosk is a weak
wake engine for a name like "Stella"; if it stays flaky, move to Porcupine (needs
access key) or a trained openWakeWord model. Face-recognition sim is low (~0.45-0.49)
— re-enroll under the current USB-cam lighting is still pending.

---

### 2026-09-13 — "Stuck / not responding": blocking mic read hung conversations

**REVERTED — do not reintroduce a reader thread.** The fix below (mic read on a daemon thread) caused the SIGABRT
crash loop in the next entry and bug_028; it was reintroduced on 2026-09-19 (440223d, bug_049) and reverted again on
2026-09-26 (442c8a8, bug_054/bug_055). Current mechanism: single-thread capture + hard deadline + two-stage watchdog
(architecture doc §4.2).
**Symptom:** intermittently she stops responding to voice. Log shows a
conversation started (wake fired, "Pausing wake-word listener — releasing mic"),
she asked "Can I do anything else?", then SILENCE for 10 min — no farewell, no
"listener resumed". The wake mic stays paused, so she is fully deaf.
**Root cause:** PortAudio `stream.read()` blocks forever when the flaky USB mic
stops delivering frames mid-capture. `capture_utterance` hangs -> conversation
never ends -> wake-word listener never resumes.
**Fix (speech_recognition.capture_utterance):** read the mic on a daemon thread
feeding a queue; the main loop consumes with a 1s queue timeout and bails out if
no audio arrives for start_timeout+max_seconds+3s. A stalled mic now ends the
capture (None -> farewell -> wake resumes) instead of hanging. `get_read_available()`
is unreliable here (always returns 0), hence the reader-thread approach.
**Also this session:** face kiosk fixed by rewriting the Windows controller
(face_client.ps1) as a stateless bridge-tracker with a log file
(%TEMP%\stella-face.log); the old bug was multiple wedged instances fighting the
single-instance mutex. Wake word + phantom-reply fixes from earlier confirmed
working (wake fired conf 0.90, lights/AC via voice).

---

### 2026-09-13 — "Offline from time to time": self-inflicted crash, reverted
**Symptom:** Stella went offline intermittently; guard unreliable.
**Root cause:** the capture-hang fix earlier this session (reader thread + queue)
closed the PortAudio stream from the main thread while the reader thread was still
reading it -> heap corruption (`malloc_consolidate(): unaligned fastbin chunk`)
-> SIGABRT core-dump every ~1.5-3.5 min. systemd restarted each time = "offline
from time to time". Guard was unreliable mainly because every crash reset
`guard_mode` to off.
**Fix:** REVERTED the reader-thread capture patch (restored blocking read). No more
cross-thread audio ops -> no more crashes (verified: 0 core-dumps, stable uptime).
**Replacement hang recovery (safe):** conversation stamps `_conv_activity` on every
spoken line; the hardware watchdog restarts the service if a conversation is active
but silent for 120s (only happens on a true mic-read hang — normal silence still
emits wrap-up/farewell every ~12-24s). No cross-thread stream manipulation.
**Lesson:** never close/stop a PortAudio stream from a different thread than the one
reading it. If the blocking-read hang must be fixed at the source later, reopen the
whole PyAudio instance or use a process-level restart, not cross-thread close.
**Still open:** underlying USB audio is flaky (paInvalidSampleRate / xrun spam);
face-recognition sim low (~0.45-0.58) - re-enroll pending. Guard state not persisted
across restarts (re-arm needed after a restart).

---

### 2026-09-13 — Guard on/off/status parsing + log-spam
**Guard bugs (motion detection itself was fine):**
 - "guard mode off" ARMED instead of disarming — it contains "guard mode" (an
   arm phrase) but not the exact "guard off", so on-keyword matched first.
 - status questions ("is guard on or off?") armed it.
 - no way to report guard status.
**Fix (_maybe_guard):** STATUS queries handled first (report, never change); OFF
intent = explicit off phrase OR (guard/security topic + an off/disable word), so
"guard mode off" disarms; ON only when no "off" present. Verified on 7 phrases.
**Log spam:** the Windows face controller polls /display_state ~twice/1.5s, which
flooded the journal (aiohttp.access). Silenced via `web.AppRunner(app,
access_log=None)`. Note: no disk-overflow risk anyway (90G free, journald caps at
~10% and auto-rotates). Full journal vacuum needs sudo (system journal) — skipped.
**Still open:** guard_mode not persisted across restarts; USB audio flaky; face
sim low (~0.45-0.58, re-enroll pending). Blocking-mic-read hang now covered by the
safe watchdog (restart after 120s of stuck conversation) instead of the reverted
cross-thread patch.

---

### 2026-09-13 — Phone face + nicer face
**Phone face:** added a separate "phone" display target so "show your face on
phone" lights up the phone without touching the PC kiosk.
 - bridge.py: `_show_phone` state, `show_phone_face()/hide_phone_face()`,
   `/show_phone` `/hide_phone` endpoints, `/display_state` now returns
   `{show, show_phone}`, and `/phone` serves the face page.
 - The phone page self-gates: it polls `/display_state` and only wakes the face
   when `show_phone` is true; otherwise a resting screen with a hint. Add-to-Home-
   Screen makes it act like an app (PWA meta tags added).
 - manager `_maybe_face`: "on phone/mobile/cell" -> phone target + a Telegram
   tap-to-open link (`_send_phone_face_link`, best-effort LAN IP). Plain
   "show/hide your face" still = the PC kiosk.
 - Requires the phone on the same network as Stella (home WiFi or RobotNet).
**Nicer face (index.html rewrite):** warmer multi-stop skin shading with
cheekbones/temples/forehead highlight, fuller layered hair (back + fringe +
strand highlights), almond eyes with lids/lash line/outer lashes/limbal ring/
catchlights/gaze-follow pupil, softer brows, shaded nose with nostrils/tip
highlight, fuller lips with philtrum + teeth/tongue when talking, subtle
breathing bob. Same WebSocket protocol (emotion/look_at/gaze/speak/viseme), so
lip-sync, gaze and emotions still drive it. JS syntax-checked with deno; verified
it loads in the PC kiosk and phone endpoints work.
**Verify pending (needs Moti):** visual judgement of the new face; open
http://<pi>:8080/phone on the phone and say "show your face on phone".

---

### 2026-09-13 — Full architecture review (multi-agent, ultracode)
Ran a 16-agent audit + research + design pass. Canonical output: docs/architecture/stella-architecture-2026-09-13.md (north star, capability stack online+offline, phased rollback-safe roadmap, object-recognition root cause, Hailo unlock, NVIDIA NIM plan). Key finding: object recognition is dead due to 3 stacked blockers (no model file; enable_hailo:false; wrong pip hailort 4.23.0 vs system 5.1.1). Also found a master-auth bypass in keyword handlers and secret-hygiene issues. Roadmap: Phase 0 reliability+security, Phase 1 revive perception+offline voice (CPU), Phase 2 NVIDIA NIM + offline cliff, Phase 3 offline VLM, Phase 4 (optional/late) HailoRT 5.2.0 upgrade, Phase 5 on-NPU perception. NO code changed in this pass.

---

### 2026-09-13 — Build increment: Phase 0/1 quick wins (all tested + pushed)
Implemented from the architecture roadmap, each a small reversible commit (pushed to origin/jetson_new_v1):
- **Object recognition REVIVED** (`cf767f8`): exported a stock `yolov8n.onnx` (Ultralytics, opset 12, 640) and dropped it in `data/models/` (gitignored). `HailoDetector` now loads it via OpenCV-DNN (`enable_hailo` stays false). Verified live: raw output (1,84,8400), real detections (class 57 couch x5) on a camera frame; module subscribes to the camera and starts (backend=opencv_dnn). Also deleted the stale never-loaded root `config.yaml` (Gonzo trap). Rollback: delete the .onnx.
- **NVIDIA NIM VLM** (`8be3d73`): `modules/vision/vlm.py` now falls back to NIM `meta/llama-3.2-11b-vision-instruct` behind Moondream (NVIDIA_API_KEY in .env, chmod 600, gitignored). Verified: NIM vision returns correct answers from the Pi. NOTE: on this key ONLY the 11b-vision model is provisioned — all other NIM models 404 ("not found for account"), so NIM = vision only here, not an LLM/embeddings provider.
- **Watchdog reliability** (`cf4fc5f`): stamp `_conv_activity` on USER speech (not just Stella's), threshold 60->75s — cuts false mid-conversation restarts while still recovering a real mic-hang fast.
- **Security: master-auth gate** (`875df0c`): added `_is_master()`/`_deny_master()` and gated `_maybe_ac` + `_maybe_mirror` (were bypassing the master check that `_do_device_control` and the LLM tools already enforce). Respects `require_authentication`.

Deferred (need user or more care): openWakeWord "Hey Stella" (needs Moti's voice recordings + Colab training); faster-whisper offline STT; face-recognition alignment fix; web-pre-injection gate fix; the invasive HailoRT 5.1.1->5.2.0 upgrade (Phase 4). Rollback baseline for the whole session: commit `382fb4c`; full local backup at `C:\Users\Moti\AIRobot_work\_backups\AIRobot_v2.0_backup_2026-09-13.tgz`.

---

### 2026-09-16 — "Slow + talks unprompted" (work, direct cable, no mic/cam/HDMI): diagnosis + fixes
Ran a 3-agent diagnosis (slowness / unprompted speech / missing-hardware) over 3 boots of logs + code.

**Root causes found**
1. **Offline brain DEAD (biggest cause of slow + dumb offline):** kernel auto-upgraded to `6.8.0-1064-raspi`; the Hailo DKMS driver (`hailo1x_pci/5.1.1`) was only built for 1060 -> no `/dev/hailo0` -> hailo-ollama "HAILO_OUT_OF_PHYSICAL_DEVICES". `linux-headers-raspi` metapackage was never installed, so DKMS can't auto-rebuild on kernel upgrades. **Needs Moti (sudo + internet):** `sudo apt install -y linux-headers-raspi linux-headers-$(uname -r) && sudo dkms autoinstall && sudo modprobe hailo1x_pci && sudo systemctl restart hailo-ollama`. The metapackage makes future upgrades self-heal.
2. **Offline cliff in the LLM chain:** every think() tried groq -> groq_fast -> gemini (30s timeout x SDK 2 retries each) before hailo; connection errors were never cooled. Offline think() measured ~90s+.
3. **Unprompted speech:** (a) network monitor hard-coded `was_online=True`, so an offline boot "lost" internet ~16s after every start and announced it; (b) with NO mic, `capture_utterance` returned None instantly and `_run` treated it as silence -> greeting -> "anything else?" -> farewell monologue; (c) Whisper no-speech hallucinations ("Foreign", "Thank you.", "so", "oh", "Hey") were answered and even ENROLLED AS NAMES ("Nice to meet you, Foreign").
4. **Missing-hardware waste:** wake mic retried at 1 Hz forever with an ERROR each time (381/6min); TTS pinned to a dead HDMI sink + watchdog re-probed every 20s; YOLO ran on every delivered frame (~3-4 CPU inferences/s) and loaded even with no camera; conversation/watchdog timers used wall-clock (offline clock steps hours on NTP sync -> false "stuck" restart).

**Fixes deployed (8 files, all edits anchored/asserted locally first, syntax-checked on Pi)**
- ai_engine.py: `online` flag (set by network monitor) -> `_skip()` skips cloud providers offline; `max_retries=0`; `httpx.Timeout(20, connect=3)`; agent/web-search gated on online; reasoning-model `max_tokens>=1024` + empty-content warning. **Offline think() now 0.2s (was ~90s).**
- main.py: network monitor starts from real state (no announce for a booted-into offline), 30-min announce rate-limit, pushes `online` to the engine.
- manager.py: no spoken session without a mic (`mic_available()`), quiet break if mic lost mid-chat, `_thinking` flag + monotonic activity stamps, junk-name rejection in `_enroll_name`.
- speech_recognition.py: `mic_available()` (checks real capture PCMs `/dev/snd/pcmC*D*c` so PortAudio name flip-flop doesn't mute her at home), no fall-through to PortAudio 'default' for an absent named mic, native-rate-first (44.1k) to kill paInvalidSampleRate spam, Whisper hallucination stoplist treated as silence.
- wake_word.py: log-once + exponential backoff (1s->60s) when no mic; set `_stream_closed_event` in paused branch (no 2s stall).
- watchdog.py: skip while `_thinking`; monotonic clock; audio re-probe backoff (20s->300s).
- text_to_speech.py: fall back to a non-HDMI output when HDMI has no sink; aplay 60s timeout. **Regression caught & fixed:** first version returned None -> TTS took the pygame path -> `pygame.mixer.init()` HANGS with no sound device -> startup stalled at "Using Piper TTS engine". Now always returns a device string (fast-failing aplay), as the original did.
- object_detection.py: don't load the detector without a camera; `MIN_DETECT_INTERVAL=1.0s` throttle.

**Status:** all 8 deployed; TTS regression fix deployed + Pi-syntax-checked and restart issued — the Ethernet link dropped during verification, so **startup-complete verification and git commit/push are PENDING** until she's reachable again. Rollback: `git checkout -- <file>` per file (repo clean at `62fec28`), or `git reset --hard 62fec28`.


---

### 2026-09-19 — Architecture v2: layout, safety net, self-evolution, bug history
**Decision:** (1) `parts_used/` = one file per physical part (camera, audio, Hailo, hand, MCU bridge, modem, WiFi, RC toy);
`modules/` grouped by capability (`smart_home/`, `media/`, `comms/`); `core/watchdog.py`; operator scripts in `tools/`.
Pure `git mv` + import rewrite (no dynamic imports existed); stale root duplicates and `modules/vision/wake_word.py` removed.
(2) **Guardian** (`evolution/guardian.py`, deterministic, no LLM) + `deploy/deploy.sh` (restart → Guardian → `git reset --hard`
rollback) is now the only way we deploy. (3) `bug_report/` seeded with 40 historical bugs; every future fix gets a file.
(4) From the phone "self-evolution blueprint" we built the safe parts — manifest, evolution.db, free-only LLM client,
report-only Scout, morning Telegram report, 03:00 cron — and **deferred Lab/Builder (auto-merge) and the hardware inbox**:
a 1.5B model must not install packages into a running robot unsupervised; they return behind `require_operator_approval`.
(5) RC toy contract: `rc_toy.connected` flag, `RCToy.drive/halt/is_available`, dead-man stop, master-gated `drive_toy` tool.
**Findings while building:** Groq's API 403s Python's default User-Agent (browser UA needed); gpt-oss/gemini-flash need
`max_tokens ≥ 2048` + `reasoning_effort: low` or content comes back empty/truncated; the Pi hit the 80 °C soft limit (no fan);
no undervoltage recorded on the readable boots — the power-offs cut the log mid-line (hard cut), PSU still to be checked.
**Rejected:** import shims for old paths (would double the file count and defeat the point); autonomous package installs.


### 2026-09-19 — Telegram voice commands
**Decision:** a voice note sent to the bot is downloaded, decoded with ffmpeg to 16 kHz PCM, transcribed through the SAME
STT chain as her ears (SpeechRecognitionModule._transcribe_pcm16k: Groq Whisper → Google → Vosk, hallucination filter),
echoed back as "🎤 heard: …", then handled exactly like a typed message (guard, face, tools, LLM). Every reply to a voice
command is also sent as a voice note (Piper → ffmpeg libopus → Telegram sendVoice). Notes over 60 s are refused. Master chat only.
**Why:** hands-free control from anywhere; reuses existing pieces, no new dependency (ffmpeg has libopus).


### 2026-09-19 — Speak-aloud on the physical speaker (Telegram)
**Decision:** "say/speak/read ... out loud|on the speaker" and "announce/broadcast ..." now make Stella SPEAK the phrase on her own speaker instead of typing it back to the chat. Two paths: a direct handler in `telegram_bridge._process` (regex `_extract_speak_aloud`, reliable, needs no LLM) and a `speak_aloud` LLM tool (registered in the conversation manager) for natural phrasing and voice chats. Both call `tts.speak()`.
**Why:** Moti asked her to say something on the speaker and she wrote it back to the bot — she had no way to vocalise a given phrase remotely. The direct handler keeps it working when the cloud brain is rate-limited.
**Guard:** the say/speak/read verbs require an explicit out-loud/speaker cue so normal requests ("say what you see") are not hijacked; announce/broadcast trigger on their own. Verified with 10 phrasings (7 trigger, 3 must not).

### 2026-09-19 — Full audit (14-agent workflow) + reliability/perf fixes

**PARTLY REVERTED — the "background reader-thread + 8s consumer timeout" fix for bug_049 was reverted on 2026-09-26
(442c8a8) after it crashed the process (bug_054) and then leaked the command mic (bug_055). Do not reintroduce it.
The two-stage watchdog from the same commit stands.**
**Decision:** ran a 5-dimension review (stuck, face-recognition, speed/thermal, architecture/capability, correctness) with adversarial verification; applied the verified safe fixes in 4 Guardian-deployed commits (57ad402, 440223d, 3210df1, c03a86b).
**Root cause of "stuck":** unbounded PortAudio `stream.read()` on the shared USB dongle — a stall wedged the capture loop, froze `_conv_activity`, blocked `conversation.stop()`, forcing a full ~30s service restart. Fixed with a background reader-thread + 8s consumer timeout (get_read_available returns 0 on this ALSA build, so the poll variant was rejected) and a two-stage watchdog (soft conv.stop() before a restart). bug_049.
**Also fixed:** master identity cleared on a single Unknown frame (debounce + identity grace, bug_047); YuNet tight box vs dlib-enrolled geometry (box padding at both encode sites, re-enroll pending, bug_048); send_photo tool 400 on groq-120b (required:[]+additionalProperties:false, bug_043); Sensibo 422 from echoed stale fields (fresh state + drop temp in fan/dry, bug_051); TTS fallback subprocess timeouts.
**Perf/thermal:** cv2.setNumThreads(2) + object-detection MIN_DETECT_INTERVAL 1.0->4.0 (top heat source was the 1Hz all-core YOLO); dashboard encodes JPEG only while a browser is streaming; YuNet top_k 5000->50. Temp under load fell ~65C->~58C.
**Deferred (LATER, in the audit output):** YuNet frontal-landmark gate + 5-pt alignment; per-face temporal smoothing at the recognition layer; provider-chain total wall-clock budget; settings.py euclidean/0.60 default hardening. **Needs Moti:** re-enroll Moti+Orr under YuNet; a fan/heatsink; a 2nd USB sound device or the planned I2S mics.

---

### 2026-09-20 — Stella "dead" on home WiFi: onboard radio renamed wlan0 -> wlan1 (bug_052)
**Symptom:** unreachable on any home IP; only the RobotNet AP (10.0.0.1, USB dongle) answered. `iw dev` showed the
onboard radio as `wlan1`, no `wlan0`; `sudo iw dev wlan0 scan` -> "No such device".
**Root cause:** a boot race in interface naming — the onboard Broadcom radio sometimes comes up as `wlan1` and the
netplan stanza only knew `wlan0`.
**Decision:** dual-name netplan (`/etc/netplan/50-cloud-init.yaml`, system file, not in the repo) so the home-WiFi
client config applies whichever name the radio gets. Documented in bug_052 (committed 2026-09-26, 56c2dc3).
**Rejected:** a udev rename rule (another moving part at boot) and pinning by MAC only.

---

### 2026-09-26 — Wake mic moved back to the camera mic (d3b2577, bug_053)
**Symptom:** she recognised Moti on camera but never answered "Hey Stella"; `wake heard:` lines were gibberish
("good luck to the moon", "richard").
**Root cause:** bug_046 (2026-09-19, 743ef31) had moved the wake role to the USB PnP dongle to dodge a stall; that mic has
AGC and poor pickup at conversational distance, so Vosk never matched "stella". Mic quality, not a stall.
**Decision:** wake = camera built-in mic ("Auto Focus Camera", ALSA card `Camera`); command = "USB PnP Sound Device"
(card `Device`). The stall that motivated bug_046 is covered by the wake-loop self-heal (dead-stream reopen).
**Status:** verified live 2026-09-26 21:12 (wake on the camera mic, paired pause/resume, NRestarts=0).

---

### 2026-09-26 — Reader thread: re-added 2026-09-19 (440223d), crashed (bug_054), re-fixed (ed73b10), REVERTED (442c8a8)
**What happened, in order (journal + git):**
1. 2026-09-19 22:43, 440223d (bug_049): a background reader thread bounded the command-mic read (8 s consumer timeout) —
   the same design that bug_027/bug_028 had introduced and reverted on 2026-09-13. Nothing at the edit point said so.
2. 2026-09-26 13:31 and 13:33: `malloc(): unaligned tcache chunk detected` -> SIGABRT -> systemd restart after each
   wake (bug_054): the main thread's `finally` closed the stream while the reader was blocked in `read()`.
3. 13:39, ed73b10: reader made the sole owner of the stream; "stress-tested 4 open/read/close cycles" — a test that never
   exercised the stall branch.
4. 13:47: a capture stalled; the main thread `join(timeout=2)`-ed and continued, abandoning a thread that still held
   the ALSA device. `Could not open command mic for capture` x8 in 4 minutes.
5. 13:51, 442c8a8: **reader thread reverted**. `capture_utterance()` is single-thread again (open -> read -> close on the
   conversation thread) with a hard wall-clock deadline between reads; a truly wedged `read()` is bounded only by the
   two-stage watchdog (75 s soft stop, +25 s restart).
**Decision (rules, now in architecture doc §12):** a PortAudio stream is opened, read, stopped and closed by exactly one
thread, enforced by a `MicStream` owner guard in code; never abandon a thread holding a stream; a close/join timeout is
terminal for that device (flag + watchdog), never a reopen; reproduce the real stall (usbreset/unplug during a blocked
read) before any mic fix; one audio change per day with a 24 h soak. **A reverted fix is edited in place** — bug_049
now says REVERTED, the 09-13 and 09-19 entries above carry banners.
**Rejected:** sounddevice callback migration for now (bigger change; the single-thread read + watchdog is sufficient and
proven); any "stuck read" fix that adds a thread.

---

### 2026-09-26 — Single-mic detour (2051a76) and the misdiagnosis it came from (bug_055); two-mic layout restored (a2e3229)
**What happened:** the 13:47 open failures (caused by our leaked reader thread, above) were read as "USB PnP mic wedged /
unopenable". 2051a76 (13:56) routed the command mic to the camera mic as well — one device for wake and command — and
cited a bug_055 that did not exist. At 14:40 the wake thread did not close its stream within `pause_listening()`'s 2 s
(`Timed out waiting for wake-word stream to close`); the code proceeded to open the command mic on the same, still-held
device -> `Could not open command mic for capture`. Deaf again.
**Root cause of the failure:** with one ALSA device the wake-stream close and the command-stream open contend; with two
devices they cannot. The USB mic was never faulty.
**Decision:** a2e3229 (20:26) restored the canonical **two-mic layout** — wake = camera built-in mic
(by-id `usb-Signo_Camera_WB-400_Auto_Focus_Camera*`, card `Camera`, hw:3, 48 kHz), command = USB PnP Sound Device
(by-id `usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*`, card `Device`, hw:0, 44.1 kHz). Startup will refuse a config
where both roles resolve to one device; a wake-close timeout becomes terminal (no command open, watchdog escalates).
**Single-mic handoff is documented as failed and is not retried.** Wake moved three times in a week (743ef31 -> d3b2577 ->
2051a76) before being restored; from now on a mic-role change needs a bug file, a log entry and a 24 h soak.
**Status:** 0 restarts and zero `Could not open` / `Timed out` lines since 20:26.

---

### 2026-09-27 — Deep clean: one architecture doc, engineering rules, dead code and stale docs removed
**Why:** between 2026-09-19 and 09-26 roughly five audio failures were caused by fixes (three SIGABRTs, one device leak,
one failed single-mic detour) against one or two organic stalls. The hardware problem is ordinary (two cheap USB mics,
PortAudio's frozen device table, a user-session PulseAudio); the structural problem was that nothing audio had a single
owner (three mic resolvers, an unused AudioManager doing a second Pa_Initialize, hard-coded rates beside unread config
keys, a legacy brain pipeline opening mics through second paths, a silent `input_device_index=None` fall-through), and
the lessons were written in the wrong place (a decision-log entry still presenting the reader thread as THE fix,
bug_049 still "fixed", a commit citing a non-existent bug_055).
**Decisions:**
- **One architecture document**: `docs/architecture/stella-architecture.md` (sections 0–14: hardware, process/threads,
  config, the canonical two-mic audio design with invariants I1–I8, conversation, vision, smart home/guard, bridges,
  watchdog, evolution/deploy, layout, the 14 engineering rules, test checklist, debt). The two dated docs move to
  `docs/architecture/archive/` with a SUPERSEDED banner; `docs/architecture.md`, `docs/GONZO_GUIDE.md`,
  `docs/rpi5_setup.md` (apt list kept in `docs/hardware/hailo.md`), `docs/service_accounts.txt` and `claude_read.txt`
  are deleted. Docs describe the CURRENT layout; history stays in `bug_report/` and this log.
- **Engineering rules** (architecture doc §12, verbatim): PortAudio single-owner guard in code; terminal close-timeout;
  two mics / two roles / two devices; one resolver on stable identity, never a silent default; one PyAudio instance;
  reverted fixes edited in place; bug file per fix with no number gaps; reproduce before fix; one audio change per day
  + 24 h soak; deploy only via Guardian from a clean tree; config-key and no-orphan tests; identity not index; one owner
  per concern; docs updated in the same commit.
- **Removed (verified no importer / no live producer)**: `parts_used/esp32_controller.py`, `parts_used/sim7600x_modem.py`
  (its import-time serial probe blocked every `config.settings` import on `/dev/ttyAMA0`), `modules/connectivity/`,
  `tools/test_sim7600x.py`, `config/config.json` (the silent Gonzo-era fallback), `main.py --create-service` and
  `--test` code with the `face_recognition.py` legacy camera methods; `stella_manifest.yaml` untracked and gitignored
  (a tracked generated file dirtied the tree after the first nightly run and disabled deploy). Kept on purpose after
  review: brain transitions still reachable by voice (movement, learning, recovery, LED) — removal is a separate step.
- **Bug ledger**: bug_055 created (the misdiagnosis, honestly), bug_050 retired to close the gap, bug_049 -> REVERTED,
  bug_054 -> its first fix marked reverted, bug_046 -> superseded by bug_053, bug_053 -> verified; README rebuilt from
  the file headers; template gains `Superseded by` / `Reverted` fields.
- **Scheduler**: the user crontab (`0 3 * * * evolution/nightly.sh`) is the only scheduler; `deploy/stella-evolution.*`
  are an optional sudo alternative, not installed.
- **Code fixes queued** (architecture doc §4.7 / §14, each its own Guardian-deployed commit with its test): MicStream
  guard, by-id resolver, terminal wake-close timeout, startup wake≠command assertion, config mic ids and rates, gain
  boost by card id, shared device table log, config.json fallback removed with loud YAML failure, known-good dataclass
  defaults, SIM7600X probe removed, watchdog docstring + wake-listener visibility check, stale paths in
  `tools/check_deps.py` and `docs/firmware/hand-esp32.md`, `MqttConfig`, dead .gitignore lines and `gonzo.log` ->
  `stella.log`.
**Rejected:** deleting `core/robot_brain.py` dead states in this pass (live and dead code interleaved; needs its own
verification); removing the DeepFace/legacy face-recognition methods beyond the camera path; any change to the mic
assignment itself.

**2026-09-27 (later) — landed:** the deep-clean code fixes are in: dead code + config hygiene (0e01175, ed47922), docs (750d2df), audio hardening (587f965: MicStream single-owner guard, find_input_index by-id resolver, two-mic startup invariant, wedged-wake escalation, gains by card id, rate tuples from config, tests/test_audio_threading + test_mic_resolution). Journal at first start: `Mic map: wake -> device 2 (hw:3), command -> device 0 (hw:0)`, gains `Device:ok, Camera:ok`.
