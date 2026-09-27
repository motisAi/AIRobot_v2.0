# Stella — Architecture (current)

## 0. Status

> **This is the only current architecture document.**
> Superseded: `archive/stella-architecture-2026-09-13.md` and `archive/stella-architecture-2026-09-19.md`
> (both moved to `docs/architecture/archive/` with a SUPERSEDED banner) and `docs/architecture.md` (deleted).
> This file is updated **in the same commit** as any change to a device assignment, a thread ownership or a
> public interface. History lives in `bug_report/` and `docs/decisions/log.md`, not here.
>
> Last verified against commit `a2e3229` on 2026-09-26 (live journal, `systemctl show airobot -p NRestarts` = 0
> since 20:26). Written for the 2026-09-27 deep clean. Items tagged **[done 2026-09-27]** landed that day (dead code 0e01175/ed47922, docs 750d2df, audio hardening 587f965); the few still tagged **[planned]** are
> are decided but were not yet in `a2e3229` when this was verified; everything else describes the running code.

---

## 1. What Stella is

A Raspberry Pi 5 robot that listens, talks, sees, remembers people, controls the house and moves a hand.
One Python process (`main.py`) under one systemd unit (`airobot`).

| Part | Exactly what is fitted | Owner in code |
|---|---|---|
| Compute | Raspberry Pi 5, Ubuntu Server 24.04 aarch64, headless, user `moti_ai`, SSH alias `rpi5` | — |
| NPU | Hailo-10H (PCIe/M.2), driver `hailo1x_pci` (DKMS, `/dev/hailo0`). **Serves only the offline LLM** (`qwen2.5-instruct:1.5b` via the `hailo-ollama` service on `:8000`). Object detection runs on the **CPU** (OpenCV-DNN, `data/models/yolov8n.onnx`) because the pip `hailort` wheel is ABI-broken against HailoRT 5.1.1 — stated explicitly so nobody "fixes" vision by touching the NPU. | `modules/ai/ai_engine.py` (LLM), `parts_used/hailo_10h.py` (detector, CPU fallback path is the live path) |
| Camera | USB UVC webcam "Signo Camera WB-400 Auto Focus Camera" with a **built-in mic** (ALSA card id `Camera`) | `parts_used/camera_usb.py` `CameraManager` |
| Wake mic | the camera's built-in mic (see §4) | `modules/audio/wake_word.py` |
| Command mic | "USB PnP Sound Device" (C-Media dongle, ALSA card id `Device`) (see §4) | `modules/audio/speech_recognition.py` |
| Audio out | HDMI (`vc4hdmi0` / `vc4hdmi1`, auto-detected); optional BT speaker "AR-SPJ" | `modules/audio/text_to_speech.py` |
| Hand | ESP32-S3-WROOM-1 + PCA9685 + 5 servos, USB serial `/dev/ttyACM0` @115200 | `parts_used/esp32_hand.py` `Hand` |
| MCU bridge | generic real-world command sink (serial / network / null), `microcontroller.connected: false` today (log-only) | `parts_used/microcontroller_bridge.py` |
| Phone link | Telegram bot (two-way chat, alerts, voice notes, YES/NO flows) | `modules/comms/telegram_bridge.py` |
| Smart home | Tuya plug (local key), Sensibo AC (cloud), MQTT hub (mosquitto on the Pi, `:1883`, 0 devices configured) | `modules/smart_home/` |
| Screen face | `face_bridge/` — HTTP + WebSocket on **`:8080`** (ws path `/ws`), shown on a PC/tablet on voice command | `face_bridge/bridge.py` |
| Dashboard | `modules/web/dashboard.py` on `:5000` (camera, conversation, guard) | — |
| WiFi | onboard radio = home WiFi client; TP-Link Archer T2U Plus (RTL8821AU, `8821au` DKMS) = "RobotNet" AP 10.0.0.1/24 | `parts_used/wifi_adapter.py`, `robotnet-*.service` (system) |
| RC toy | stub, `rc_toy.connected: false` | `parts_used/rc_toy.py` |

**Not fitted / not used (do not document as present):** SIM7600X 4G modem, IMU, wheel encoders, wheels, lidar,
ultrasonic, Picovoice/Porcupine, Coqui TTS, DeepFace, local Whisper, `llama_cpp`, Anthropic/OpenAI providers
(keys optional, not in the chain). The code paths for these are dead and are listed as debt in §14.

---

## 2. Process model and runtime

**One unit, one interpreter.** `deploy/airobot.service`: `User=moti_ai`, `WorkingDirectory=/home/moti_ai/AIRobot_v2.0`,
`ExecStart=/home/moti_ai/AIRobot_v2.0/venv/bin/python main.py`, `Restart=on-failure`, `RestartSec=5`,
`After=network-online.target hailo-ollama.service`. The interpreter is **`venv/`**. A second `.venv/` exists in the
repo directory, is not used by anything and lacks PyYAML — never run Stella from it.

**Boot order** (`main.py`): config load → `CameraManager` → face recognition → object detection → wake word →
speech recognition → TTS → MCU bridge → navigation hooks → motion guard → VLM → face bridge (`:8080`) → music →
audio arbiter → hand → hand mirror → RC toy → conversation manager → Telegram → dashboard (`:5000`) → watchdog →
`start()` of the vision/audio modules → brain event thread → `run()` loop.

**Threads and what each one owns**

| Thread (name) | Created in | Owns |
|---|---|---|
| main | `main.py` | initialisation, `run()` loop, `shutdown()` |
| brain event thread | `core/robot_brain.py:1387` | the event bus / state machine |
| wake-word listener | `modules/audio/wake_word.py:142` (`_listen_loop` → `_vosk_loop`) | **the wake stream** (open, read, stop, close) — nobody else touches it |
| `conversation` | `modules/conversation/manager.py:55` | the spoken session and **the command stream** during `capture_utterance()` |
| camera capture | `parts_used/camera_usb.py:155` | the one `cv2.VideoCapture` |
| face capture + recognition | `modules/vision/face_recognition.py:180,185` | frame subscription, YuNet + dlib |
| TTS speech loop | `modules/audio/text_to_speech.py:316` | piper/aplay playback queue |
| `hw_watchdog` | `core/watchdog.py:38` | health checks every 20 s (§9) |
| `telegram` | `modules/comms/telegram_bridge.py:46` | `getUpdates` polling |
| `face_bridge` | `face_bridge/bridge.py:47` | HTTP/WS server `:8080` |
| `web-dashboard` | `modules/web/dashboard.py:462` | HTTP `:5000` |
| `reminders` | `modules/ai/reminders.py:83` | reminder announcer |
| one-shots | `main.py` (AI warmup, network monitor, performance monitor, gaze feed, guard alert, welcome) | — |

Rule: a resource (PortAudio stream, serial port, camera) is opened, used and closed by **one** thread. §4 is the
enforced instance of this rule for audio.

**Scheduler.** The only scheduler is the **user crontab**: `0 3 * * * /home/moti_ai/AIRobot_v2.0/evolution/nightly.sh`
(`crontab -l`). `deploy/stella-evolution.service` / `.timer` are an optional sudo alternative and are **not installed**
(`systemctl list-timers` shows none).

**Deploy.** Only through `deploy/deploy.sh` from a clean tree: restart → Guardian `--wait` → auto-rollback
(`git reset --hard <previous>` + restart + Telegram line) if unhealthy. See §10.

---

## 3. Configuration

`config/config.yaml` is the **single editable file**. It is loaded only by the dataclasses in `config/settings.py`
(`RobotConfig` and its sections), which export module-level objects (`hardware_config`, `model_config`, …) that the
code imports. Secrets live in `.env` (mode 600, gitignored): `GROQ_API_KEY`, `GEMINI_API_KEY`, `MOONDREAM_API_KEY`,
`NVIDIA_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `SENSIBO_API_KEY`.

Rules:
- No other YAML reader. **[planned]** `main.py` still reads the `mqtt` section from raw YAML; it moves to a
  `MqttConfig` dataclass (§14).
- No JSON fallback. `config/config.json` is removed in the deep clean; **[done 2026-09-27]** a `config.yaml` load failure logs
  ERROR with the path and exception and exits with code 2 instead of silently booting the Gonzo-era defaults.
- Every yaml key maps to a dataclass field **and** is read by some code — **[done 2026-09-27]** `tests/test_config_keys.py`
  enforces it (14 decoy keys exist today).
- Dataclass defaults equal the known-good live values (face threshold 0.60 / euclidean / yunet, wake word `stella`,
  `robot_name` Stella, the current Groq/Gemini model ids) — **[done 2026-09-27]**, today several defaults are stale.

| Section | Read by (verified by grep on `*_config` imports) |
|---|---|
| `behavior` | `main.py`, `core/robot_brain.py`, `modules/audio/speech_recognition.py`, `modules/audio/text_to_speech.py`, `modules/conversation/manager.py`, `modules/ai/ai_engine.py` |
| `ai` | `main.py`, `modules/ai/ai_engine.py`, `tools/check_deps.py` |
| `conversation` | `modules/conversation/manager.py` |
| `music` | `main.py` |
| `web_search` | `main.py`, `modules/ai/ai_engine.py` |
| `hand` | `main.py`, `modules/conversation/manager.py` |
| `microcontroller` | `main.py`, `parts_used/microcontroller_bridge.py`, `tools/check_deps.py` |
| `rc_toy` | `main.py` |
| `navigation` | `main.py`, `modules/navigation/navigator.py` |
| `hardware` | `main.py`, `modules/vision/face_recognition.py`, `modules/audio/{speech_recognition,text_to_speech,wake_word}.py`, `parts_used/{audio_devices,camera_usb,hailo_10h}.py` |
| `model` | `modules/vision/face_recognition.py`, `modules/audio/{speech_recognition,text_to_speech,wake_word}.py`, `parts_used/hailo_10h.py` |
| `system` | `main.py`, `core/robot_brain.py`, `modules/vision/{face_recognition,object_detection}.py`, `modules/ai/{learning_db,ai_engine}.py`, `parts_used/{camera_usb,hailo_10h}.py` |
| `security` | `main.py`, `core/robot_brain.py`, `modules/vision/face_recognition.py`, `modules/conversation/manager.py` |
| `mqtt` | `main.py` (raw YAML today — debt) |

Audio keys that matter (`hardware`): `wake_word_microphone_name: "Auto Focus Camera"`,
`speech_microphone_name: "USB PnP Sound Device"`, `microphone_rate: 48000`, `speech_microphone_rate: 44100`,
`audio_output_device: hdmi`, `audio_output_card: null` (auto-detect). **[done 2026-09-27]** `wake_mic_id` / `speech_mic_id`
(by-id strings) alongside the name keys; the rate keys become the first entry of the probe tuples (§4, invariant I7).

---

## 4. Audio — the canonical two-mic design

This section supersedes every other mention of microphones in the repo (README, `docs/hardware/*`,
`docs/schematics/*`, older bug files). It is what worked before the 2026-09-19..26 changes and what is running now at
`a2e3229` with 0 restarts.

### 4.1 Devices (stable identity, verified on the Pi)

| Role | Device | `/dev/snd/by-id` | ALSA card (id) | PortAudio name | Native rate | Config key |
|---|---|---|---|---|---|---|
| **WAKE** | camera built-in mic | `usb-Signo_Camera_WB-400_Auto_Focus_Camera_200901010001-02` → `controlC3` | card 3 (`Camera`) | `Auto Focus Camera: USB Audio (hw:3,0)` | 48000 Hz (no 16 k) | `hardware.wake_word_microphone_name: "Auto Focus Camera"` (**[done 2026-09-27]** `wake_mic_id: usb-Signo_Camera_WB-400_Auto_Focus_Camera*`) |
| **COMMAND** | dedicated USB dongle | `usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00` → `controlC0` | card 0 (`Device`) | `USB PnP Sound Device: Audio (hw:0,0)` | 44100 Hz | `hardware.speech_microphone_name: "USB PnP Sound Device"` (**[done 2026-09-27]** `speech_mic_id: usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*`) |

- Cards 1 and 2 (`vc4hdmi0`, `vc4hdmi1`) are playback-only HDMI. HDMI card numbers move across reboots; USB card
  numbers can move after a replug. **PortAudio indices are not ALSA card numbers** — at `a2e3229` the camera mic is
  PortAudio device **2** = `hw:3,0`, the dongle is PortAudio device **0** = `hw:0,0`. Only the `(hw:N,` fragment in the
  PortAudio name is a stable link to the card.
- PulseAudio runs as the user; its default source is the camera mic
  (`alsa_input.usb-Signo_Camera_WB-400_Auto_Focus_Camera_200901010001-02.analog-stereo`). **The robot never records
  through Pulse**: it opens the ALSA `hw` device by PortAudio index.
- Output: TTS via piper → espeak-ng fallback, played with `aplay` on the auto-detected HDMI device string (bug_038);
  never via pygame/pyaudio (those code paths in `text_to_speech.py` are dead, §14).

### 4.2 Ownership and threads

- **One PyAudio instance** for the process: `parts_used/audio_portaudio.get_pa()`, created once, never terminated.
  **[done 2026-09-27]** it logs the input-device table once at creation (`idx name rate`); `AudioManager`'s private
  `pyaudio.PyAudio()` + `terminate()` at boot and `text_to_speech._play_with_pyaudio` are removed. No other file may
  construct `pyaudio.PyAudio()`.
- **Wake stream**: owned exclusively by the wake-word listener thread (`_vosk_loop`): `_open()`, `stream.read()`,
  `_close()` all run inside that thread. `pause_listening()` only clears `listen_event` and waits (2 s) on
  `_stream_closed_event`; `resume_listening()` only sets `listen_event`. This is the design already in the code
  (`wake_word.py:162-175, 206-249`) and is deliberate — the comment at the open site cites bug_028 and bug_054.
- **Command stream**: owned exclusively by the conversation thread that calls `capture_utterance()`
  (`speech_recognition.py:134-217`): open → read loop with VAD → close in the same thread, with a **hard wall-clock
  deadline** checked between reads (`max(start_timeout, max_seconds) + 5 s`, current `442c8a8` code). **No reader
  thread, ever.** A wedged `stream.read()` is bounded by the two-stage watchdog (§9: 75 s no speech → soft session stop
  → +25 s → service restart), not by a thread.
- **[done 2026-09-27]** Both open sites wrap the stream in `MicStream` (new, in `audio_portaudio.py`), which records the owner
  thread id, raises `RuntimeError` on `stop_stream()`/`close()` from any other thread, refuses `device_index=None`, and
  asserts the resolved PortAudio name contains the expected `(hw:N,` before opening.

### 4.3 Handoff sequence (wake → command → wake)

1. Vosk detects "hey stella" on the wake stream (camera mic). Journal: `Wake word detected (method=vosk, confidence=…)`.
2. `main.py` calls `wake.pause_listening()`: the wake thread closes **its** stream; pause waits up to 2 s for
   `_stream_closed_event`. Journal: `Pausing wake-word listener (conversation) — releasing mic`.
3. If the wait times out: **[done 2026-09-27]** mark the wake device wedged, log ERROR, do **not** open the command mic, end the
   session, let the watchdog restart. (Today it logs `Timed out waiting for wake-word stream to close` as a WARNING and
   proceeds — that was the 2026-09-26 14:40 failure path under the single-mic layout.)
4. `capture_utterance()` resolves the command mic at every call (today: by name substring against the PortAudio table;
   **[done 2026-09-27]** by-id → `controlC0` → `(hw:0,`), opens at its native rate (today the literal tuple
   `(44100, 48000, 16000)`; **[done 2026-09-27]** `[speech_microphone_rate, 48000, 44100, 16000]` de-duplicated), reads with VAD,
   closes in the same thread.
5. At session end (`manager.py:263` → `main._resume_wake_word_listener()`, the single resume owner) the wake thread
   reopens the camera mic, re-resolving on every attempt with exponential backoff 1 s → 60 s. Journal:
   `Wake-word listener resumed` then `Vosk wake mic open at 48000 Hz (device 2)`.

Because the two roles are on two physical devices, the wake-stream close and the command-stream open never contend for
the same ALSA device. **That is the property the single-mic detour (`2051a76`) broke, and why it failed** (bug_055).

### 4.4 Invariants (each enforced in code and tested)

- **I1.** wake device ≠ command device (by-id and by `hw:N`) at startup, else exit with a clear error. **[done 2026-09-27]**
- **I2.** `input_device_index` is never `None` at `audio.open()`; an unresolved name is an ERROR + backoff, never a
  fall-through to PortAudio `default` (= ALSA `default` = pulse plugin = whichever source PulseAudio favours that day).
  Today `capture_utterance` refuses when `mic_available()` is false, but `mic_available()` keeps a stale index when the
  name does not resolve, and the wake `_open` reuses `self.device_index` — **[done 2026-09-27]** both become hard refusals.
- **I3.** A stream is touched only by the thread that opened it (`MicStream` guard raises otherwise). **[done 2026-09-27]**
  guard; the behaviour is already true on the live path.
- **I4.** No thread holding a stream is ever abandoned; a join/close timeout is terminal for that device in this process.
  **[done 2026-09-27]** for `pause_listening()`.
- **I5.** Exactly one `Pa_Initialize` per process; the device table is logged once so the journal shows what the app
  resolves against. **[done 2026-09-27]** (today the only table in the journal is `AudioManager`'s discarded one).
- **I6.** Every capture-capable card (any `/dev/snd/pcmC<n>D*c`) gets its capture gain raised at boot, by card id string
  not by index. **[done 2026-09-27]** (today `_boost_input_gains` loops over cards `(0, 1)`; card 1 is HDMI, card 3 — the wake
  mic — is never boosted).
- **I7.** Rates come from config (native-first), not from literal tuples. **[done 2026-09-27]**
- **I8.** Mic role changes require a bug file, a decision-log entry and a 24 h soak before another audio change.

### 4.5 Known failure modes and the response to each

| Failure | What happens | Response |
|---|---|---|
| Mic busy/absent at boot (PortAudio's frozen snapshot lacks `hw:N`; PortAudio V19.6.0-devel has no `Pa_RefreshDeviceList`) | resolver finds no `(hw:N,` entry | **[planned]** if zero streams are open, ONE serialized re-init under `open_lock()` (terminate + new PyAudio, table re-logged) and retry; otherwise ERROR + backoff. Watchdog gets a *visibility-only* WARNING when the wake listener has had no open stream for > 5 min. Today: needs a service restart (README "known quirks"). |
| PulseAudio grabbing a mic | Pulse (user session) holds the device | The by-id/`hw:N` path opens the ALSA hw device directly, never the pulse plugin. **[planned]** on "device busy" log which PID holds it (`fuser /dev/snd/pcmC<N>D0c`). |
| Replug / renumber during transport | card numbers change | by-id symlinks track the device across renumbering; the resolver re-reads them at every open. **[done 2026-09-27]** (today: name substring against the frozen snapshot). |
| Wedged `stream.read()` | the conversation thread blocks | two-stage watchdog (§9). **This is the only bound. Do not add a reader thread.** |

### 4.6 What NOT to do

- No reader threads for mic capture (bug_027 → bug_028 on 2026-09-13; bug_049 → bug_054 → bug_055 on 2026-09-19..26).
- No `stop_stream()` / `close()` from a thread other than the one that opened the stream (SIGABRT, `malloc(): unaligned
  tcache chunk detected`).
- Never abandon a thread that holds a stream (no join-timeout-and-continue): the abandoned reader keeps the ALSA device
  and every later open fails with `Could not open command mic for capture` (ed73b10, 2026-09-26 13:47).
- Never pass `input_device_index=None`.
- No single-mic handoff (wake and command on one device) — tried in `2051a76`, failed at 14:40 the same day, unsupported.
- No second `pyaudio.PyAudio()` anywhere.
- No index-based device selection in config or code; indices change every reboot and replug.
- Never proceed past `Timed out waiting for wake-word stream to close` into another open.

### 4.7 Changes landed on 2026-09-27 (commit 587f965) — kept as the verification checklist

(1) `MicStream` owner guard used in `wake_word._open/_close` and `speech_recognition.capture_utterance`;
(2) replace both `_resolve_microphone_index` copies and `mic_available()`'s fallback-to-saved-index with
`audio_devices.find_input_index()` built on `/dev/snd/by-id` and the shared `get_pa()`;
(3) remove the `input_device_index=self.device_index` None path — return None/error before open when unresolved;
(4) make `pause_listening()`'s timeout terminal (wedged flag, no command open, watchdog escalation);
(5) startup assertion wake ≠ command device;
(6) config keys `wake_mic_id` / `speech_mic_id` alongside the name keys;
(7) rate tuples from config;
(8) `_boost_input_gains` by capture-node card id;
(9) log the shared PyAudio device table once and delete `AudioManager`'s private init;
(10) comments at both open sites citing bug_028 + bug_054 and the two rules;
(11) `tests/test_audio_threading.py` and `tests/test_mic_resolution.py`.

---

## 5. Conversation pipeline

```
camera mic ──Vosk──► "hey stella" ──► main.py: pause wake, start ConversationManager session (thread "conversation")
   │                                            │
   │                       capture_utterance() on the USB PnP mic: VAD (webrtcvad), end_silence 1.2 s,
   │                       max_utterance 12 s, silence gate (rms < 220 or < 0.35 s = ignored)
   │                                            │
   │                       STT: Groq Whisper ──► Google (speech_recognition lib) ──► offline Vosk, hallucination filter
   │                                            │
   │                       keyword handlers (guard, AC, devices, music, mirror, face show/hide; master-gated)
   │                       else LLM: groq openai/gpt-oss-120b ──429──► groq_fast openai/gpt-oss-20b ──► gemini-3.6-flash
   │                                 ──► hailo qwen2.5-instruct:1.5b (hailo-ollama :8000, /v1/chat/completions)
   │                                 with tools: get_time/get_weather, web_search, look, set_reminder, control_device,
   │                                 set_guard_mode, send_telegram/send_photo, play_music/stop_music, do_gesture,
   │                                 speak_aloud (drive_toy only when rc_toy.connected)
   │                                            │
   │                       TTS: piper (data/models/piper/en_US-amy-medium.onnx) ──► espeak fallback ──► aplay HDMI
   │                                            │
   └──◄── idle_timeout 12 s: "Can I do anything else?" ── end phrase / silence ──► session end:
          learn_facts_from_session (per-person durable facts, background thread) → clear history →
          main._resume_wake_word_listener() (single resume owner; refused while a session is active)
```

- The brain's **legacy** `wake_word_heard` / `listen_for_command` pipeline in `core/robot_brain.py` is **not** on the
  live path (the live path is `main.py` → `ConversationManager`); it opens the camera and command mic through second
  code paths and is scheduled for removal (§14). Do not fix audio bugs there.
- Offline: the network monitor sets `engine.online = False`; cloud providers are skipped so an offline answer starts in
  ~0.2 s instead of ~90 s (bug_034). A one-time spoken offline notice is given at boot with no internet (`c47e21f`).
- Memory: `data/robot_memory.db` (`modules/ai/learning_db.py`) — people, objects, conversations, and the per-person facts
  distilled at session end (`6cf5416`).
- Telegram voice notes go through the same STT chain (`_transcribe_pcm16k`) and are answered in text and in the Piper voice.

---

## 6. Vision

- **Camera owner**: `parts_used/camera_usb.py` `CameraManager` is the only `cv2.VideoCapture` on the live path (one
  capture thread, subscribers get frames). USB webcam found **by name**, CSI nodes skipped (bug_019/020). The
  `cv2.VideoCapture(self.camera_index)` in `modules/vision/face_recognition.py:143` belongs to the legacy `--test`
  camera path removed in the deep clean; `tools/reenroll.py` / `enroll_master.py` open the camera only while `airobot`
  is stopped.
- **Face detection**: YuNet (`data/models/face_detection_yunet_2023mar.onnx`, `model.face_backend: yunet`, Haar kept
  as fallback, `top_k` 50). **Recognition**: dlib 128-d embeddings, **euclidean distance, threshold 0.60**
  (`model.face_distance_metric`, `model.face_recognition_threshold`). Lessons: cosine made a stranger match the master
  (bug_014); YuNet's tight box vs dlib-enrolled geometry needs box padding at both encode sites and a re-enrol
  (bug_048, re-enrol still pending). Identity is debounced — one Unknown frame no longer clears the master (bug_047);
  `security.identity_memory_seconds: 60`.
- **Object detection**: `parts_used/hailo_10h.py` `HailoDetector` → OpenCV-DNN on `data/models/yolov8n.onnx`
  (gitignored binary; a fresh clone must copy it — bug_032). Cadence `MIN_DETECT_INTERVAL = 4.0 s` idle
  (thermal, bug_039). `system.enable_hailo: false` on purpose. bug_041 lesson: model paths are resolved from the repo
  root (`Path(__file__).parent.parent`), never from the file's own folder — the 09-19 move broke this once.
- **Low light**: `hardware.camera_low_light_boost: true` (software brightening before detection); camera gain /
  brightness knobs in `hardware.camera_*`.
- **Enrolment**: `sudo systemctl stop airobot && venv/bin/python tools/reenroll.py && sudo systemctl start airobot`;
  `tools/manage_faces.py list|remove|rename`. Face DB: `data/faces/` (`security.store_faces: true`).
- **Scene understanding**: cloud VLM (Moondream → NVIDIA NIM), on demand only (`look` tool), never per frame.
- **Guard**: `modules/vision/motion_guard.py` (movement / body while armed → photo + YES/NO on Telegram → alarm).
- **Hand mirror**: MediaPipe landmarks → `set <5 bits>` on the hand.

---

## 7. Smart home, guard and identity

- **Device control — one owner (rule)**: `control_device` tries **Tuya → MQTT → wired MCU** in that order, and the
  master rule (`security.require_authentication: true`) is checked once. Today four entry points each carry their own
  copy of the check: the voice keyword handler (`_do_device_control` in `modules/conversation/manager.py`), the LLM
  tool (`control_device`), the Telegram bridge and the web dashboard — **debt to collapse into one function** (§14).
- **Guard — one owner (rule)**: one set/unset function and one phrase list ("guard on", "guard off", "I'm home",
  status questions must not arm — bug_029). Who may arm: anyone present, Telegram, dashboard. Who may disarm: the
  master by voice (`guard_require_master_to_disarm: true`, optional `guard_disarm_phrase`), the master's face after
  > 60 s away (bug_021), Telegram (the remote master), the dashboard. Same four writers as above — same debt.
- **Identity**: `master_mode` is set by face recognition (`user_authenticated` event) and by the Telegram master chat
  (`_remote_master`); every privileged keyword handler and tool checks it (bug_031 closed the keyword bypass).
- Sensibo: cloud API, cached `acState`, one call per command (bug_017), fresh state before a change (bug_051).
  Tuya: local key from `devices.json` (gitignored). MQTT hub: `mqtt.devices` is empty today; **[planned]** the hub does
  not start with zero devices.

---

## 8. Hardware bridges

- **ESP32-S3 hand**: firmware `firmware/hand/hand.ino` (I²C SDA GPIO 8 / SCL GPIO 9 → PCA9685 0x40, ~50 Hz; channels
  0..4 = pinky..thumb; homes to a fist on connect). Protocol and gotchas: `docs/firmware/hand-esp32.md`. Pi driver
  `parts_used/esp32_hand.py` holds one persistent link on `/dev/ttyACM0` (stop `airobot` before reflashing); the
  watchdog reconnects it. Bench tool `tools/handctl.py` (the doc's `firmware/handctl.py` path is stale — fixed in the
  deep clean). Bench sketches `firmware/hand_test/` and `firmware/i2c_scan/` are kept for hardware checks.
- **Microcontroller bridge**: `parts_used/microcontroller_bridge.py` is the **only** MCU command path
  (`parts_used/esp32_controller.py` removed in the deep clean; it had no importer and its only event, `battery_low`,
  had no live producer). `microcontroller.connected: false` → commands are logged only.
- **Audio arbiter**: `parts_used/audio_arbiter.py` decides who owns the one speaker (speech ducks music to
  `music.duck_volume`).
- **Not fitted**: `parts_used/sim7600x_modem.py` and `modules/connectivity/server_connection.py` removed (dead, and the
  modem probe at import time blocked every `config.settings` import on `/dev/ttyAMA0` for up to 2 s).

---

## 9. Health and recovery

`core/watchdog.py` (`hw_watchdog`, every **20 s**, first check after 20 s):

| Check | Threshold | Action |
|---|---|---|
| stuck conversation | `_conv_active` and no `_conv_activity` stamp for **> 75 s** (monotonic clock, bug_040), not while `_thinking` | **Stage 1**: `Conversation stuck (no speech NNs) — soft-stopping the session` → `conversation.stop()`. **Stage 2**: still active **25 s** later → `Conversation still stuck after soft stop — restarting service` → `sudo -n systemctl restart airobot`. |
| camera | frames stale > 15 s | WARNING only (`CameraManager` reopens itself) |
| hand | serial port vanished / link dead | re-detect `/dev/ttyACM*` / `/dev/ttyUSB*`, `hand.reconnect()` |
| audio out | TTS device fails to open 2× in a row, never while speaking | re-resolve the HDMI device; backoff 20 s → 300 s when no output exists |
| wedged audio **[done 2026-09-27]** | `_audio_wedged_at` set for > 10 s (wake-stream close timed out) | Stage 2 restart |
| wake listener **[planned]** | wake stream `None` for > 300 s while `running` | WARNING once per 10 min, no restart |

- The watchdog docstring still says 120 s; the code says 75 s — **[done 2026-09-27]** docstring fix. The header comment that mics
  "self-heal by re-resolving by name" is only true for devices present in PortAudio's frozen snapshot.
- systemd: `Restart=on-failure`, `RestartSec=5`. A crash (`code=dumped, status=6/ABRT`) costs ~30 s of blackout
  (wake, vision, guard, Telegram all drop) and resets in-memory guard state — which is why in-process recovery is
  preferred and **only the watchdog restarts the process**.
- Journal lines and their meaning: `Vosk wake mic open at 48000 Hz (device 2)` = wake stream healthy on the camera mic;
  `Resolved speech mic 'USB PnP Sound Device' -> device 0` = command mic resolved; `Could not open command mic for capture`
  = the dongle is held (by us — a leaked thread — or by Pulse) or absent; `Timed out waiting for wake-word stream to
  close` = the wake thread is stuck in `read()`, terminal (§4.3 step 3); `malloc(): unaligned tcache chunk detected` /
  `malloc_consolidate` = cross-thread PortAudio access, a code bug, never a hardware fault.

---

## 10. Evolution and deploy

**Guardian** (`evolution/guardian.py`, deterministic, no LLM, writes `evolution/reports/guardian-latest.json`, exit 0/1):
`syntax` (compileall) · `imports` (every module `main.py` imports, derived by parsing `main.py`) · `config`
(`RobotConfig` builds from `config.yaml`) · `service` (`--wait` up to 90 s, stays up `--stable` 45 s) · `log` (last
2 min of the journal: error count and fatal patterns — `Traceback`, `malloc_consolidate`,
`HAILO_OUT_OF_PHYSICAL_DEVICES`) · `hardware` (soft) · `brain` (hailo-ollama answers; `--no-brain`, `--require-hailo`) ·
`resources` (disk ≥ 2 GB, RAM, `vcgencmd get_throttled`, undervoltage reported loudly).
**[done 2026-09-27]** a `tests` check runs every `tests/test_*.py` with the venv python so the new tests (§13) gate every deploy; today Guardian and
`tests/test_imports.py` both derive their module list from `main.py`, so an orphan module is never checked.

**deploy.sh**: refuses a dirty tree (`git status --porcelain --untracked-files=no`); records HEAD; restarts; Guardian
`--wait`; on failure `git reset --hard <HEAD~1 or given sha>` + restart + Guardian again; Telegram one-liner either way.
Rollback of a bad commit = `git revert` (or `deploy/deploy.sh <good-sha>`) + deploy — never a hand-edit on the robot.

**nightly.sh** (crontab 03:00): skip if an SSH connection is established, the tree is dirty, disk < 2 GB or load ≥ 3;
then `manifest.py` → `guardian.py` → `scout.py` (report-only, 900 s cap) → `report.py`, all under `nice -n 15`.
Results: `evolution/reports/nightly.log`, `guardian-latest.json`, `reports/YYYY-MM-DD.md` + a 5-line Telegram summary.

**Generated files are gitignored**: `stella_manifest.yaml` (written by `manifest.py`; was tracked, which meant the
first successful nightly run dirtied the tree and disabled both nightly and deploy), `evolution/reports/*.json`,
`evolution/evolution.db`.

---

## 11. Repository layout (after the 2026-09-27 removals)

```
AIRobot_v2.0/
├── main.py                  entry point (systemd ExecStart); wires parts and modules; --test/--create-service removed
├── README.md                the map: what Stella is, how to run, where everything lives
├── requirements.txt
├── .gitignore               ignores .env, data/, devices.json, tinytuya.json, stella_manifest.yaml, evolution/reports/*.json
├── config/                  config.yaml (the ONE file you edit), settings.py (dataclasses, the only loader), platforms/raspberry_pi5.py
├── core/                    robot_brain.py (event bus + state), watchdog.py (§9)
├── parts_used/              ONE FILE PER PHYSICAL PART (README inside)
│   ├── camera_usb.py            CameraManager — the only camera owner
│   ├── audio_portaudio.py       get_pa() / open_lock()  [planned: MicStream, open_input()]
│   ├── audio_devices.py         card listing  [planned: find_input_index() by-id resolver; AudioManager lease API removed]
│   ├── audio_arbiter.py         AudioArbiter — who owns the speaker
│   ├── hailo_10h.py             HailoDetector — CPU OpenCV-DNN path is live
│   ├── esp32_hand.py            Hand — /dev/ttyACM0
│   ├── microcontroller_bridge.py  the only MCU command path
│   ├── wifi_adapter.py          is_online()/scan()/connect()
│   └── rc_toy.py                RCToy stub
├── modules/                 capabilities by what they do (README inside)
│   ├── ai/                  ai_engine.py, learning_db.py, reminders.py, web_search.py
│   ├── audio/               wake_word.py (owns the wake stream), speech_recognition.py (owns the command stream), text_to_speech.py, spell_parser.py
│   ├── vision/              face_recognition.py, object_detection.py, motion_guard.py, hand_mirror.py, vlm.py
│   ├── conversation/        manager.py
│   ├── comms/               telegram_bridge.py, notify.py
│   ├── smart_home/          sensibo.py, tuya_devices.py, mqtt_devices.py
│   ├── media/               music.py
│   ├── navigation/          navigator.py (all hardware flags false)
│   └── web/                 dashboard.py (:5000)
├── face_bridge/             bridge.py (:8080, ws /ws) + webface/
├── firmware/                hand/hand.ino, hand_test/, i2c_scan/ (bench sketches)
├── evolution/               guardian.py, manifest.py, scout.py, report.py, db.py, llm_client.py, nightly.sh, reports/
├── deploy/                  airobot.service, deploy.sh, install_service.sh, allow-service-control.sh, wifi helpers, stella-evolution.* (not installed)
├── bug_report/              bug_001..bug_055 + README.md index + _TEMPLATE.md
├── tests/                   test_imports.py  [planned: test_no_orphans, test_config_keys, test_audio_threading, test_mic_resolution]
├── tools/                   reenroll.py, enroll_master.py, manage_faces.py, check_deps.py, handctl.py
├── docs/                    README.md, CONTINUE.md, architecture/ (this file + archive/), decisions/log.md, hardware/, firmware/, schematics/, briefs/
└── data/                    runtime state, gitignored: models/, faces/, logs/, robot_memory.db, tts_cache/, reminders.json
```

Removed on 2026-09-27: `parts_used/esp32_controller.py`, `parts_used/sim7600x_modem.py`, `modules/connectivity/`,
`tools/test_sim7600x.py`, `config/config.json`, `claude_read.txt`, `docs/architecture.md`, `docs/GONZO_GUIDE.md`,
`docs/rpi5_setup.md` (apt list moved to `docs/hardware/hailo.md`), `docs/service_accounts.txt`, `main.py`
`--create-service` / `--test` code and the `face_recognition.py` legacy camera methods; `stella_manifest.yaml` untracked.

Rule for `parts_used/`: one file per physical part, one main class, `is_available()/start()/stop()`, talks to hardware
and nothing else; modules receive these objects from `main.py` and never open hardware themselves.

---

## 12. Engineering rules

1. PortAudio single-owner rule, enforced in code not prose: a stream is opened, read, stopped and closed by exactly one thread; the MicStream wrapper in parts_used/audio_portaudio.py records the owner thread id and raises RuntimeError on stop/close from any other thread. Never abandon a thread that holds a stream (no join-timeout-and-continue).
2. A close/join timeout on a mic is terminal for that ALSA device in this process: do not reopen the same device, set a wedged flag and let the watchdog's two-stage recovery (soft stop, then service restart) handle it. Never proceed past 'Timed out waiting for wake-word stream to close' into another open.
3. Two mics, two roles, two different devices: wake word on the camera mic (by-id usb-Signo_Camera_WB-400_Auto_Focus_Camera*), commands on USB PnP Sound Device (by-id usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device*). Startup refuses a config where both roles resolve to the same device. Single-mic handoff is documented as failed (bug_055) and is not retried.
4. One resolver, called at every open, never a silent default: parts_used/audio_devices.find_input_index() resolves /dev/snd/by-id -> controlC<N> -> PortAudio entry whose name contains '(hw:N,'. input_device_index=None is never passed; an unresolved mic logs ERROR and backs off exponentially.
5. One process-wide PyAudio instance (get_pa()), initialised once, never terminated, its device table logged once at init. No other file may construct pyaudio.PyAudio().
6. A reverted fix is edited IN PLACE: the bug file gets 'Status: REVERTED - do not reintroduce, see bug_0XX', the decision-log entry gets the same banner at its top, and the docstring at the code site cites both bug numbers. A later entry alone is not enough.
7. Bug file in the same commit as the fix, numbered sequentially with no gaps; a commit message may not cite a bug number that does not exist in bug_report/.
8. Reproduce the real failure before fixing it: for mic stalls that means killing/unplugging/usbreset-ing the device while a read is blocked, not an open/read/close loop. A fix that was not reproduced is not deployed to the live service.
9. One audio change per day, then a 24h soak with NRestarts=0 and zero 'Could not open' lines in journalctl before the next audio change. No stacking of fixes minutes apart on the live robot.
10. Deploy only through deploy/deploy.sh (Guardian: syntax, imports, tests, config check) from a clean tree; generated files (stella_manifest.yaml, evolution/reports/) are gitignored so the tree cannot be dirtied by the robot itself.
11. Every key in config/config.yaml must map to a dataclass field AND be read by some code (tests/test_config_keys.py enforces it); every .py under modules/ and parts_used/ must be imported by something (tests/test_no_orphans.py enforces it). Decoy keys and orphan modules fail the deploy.
12. Resolve hardware by stable identity (by-id symlink, ALSA card id string, USB VID:PID), never by index; indices change every reboot and replug.
13. One owner per concern: one device-control function, one guard set/unset function, one Telegram HTTP path, one config loader. New features call the owner; they do not add a parallel implementation.
14. Docs describe the CURRENT layout; history lives in bug_report/ and docs/decisions/log.md with a dated 'paths are pre-2026-09-19' banner. The single architecture doc is updated in the same commit as any change to a public interface, device assignment or thread ownership.

---

## 13. Test plan and verification checklist

Run top to bottom after any audio, vision or config change. "Journal" = `journalctl -u airobot`. Automated results
land in `evolution/reports/guardian-latest.json` (Guardian), `evolution/reports/nightly.log` and
`evolution/reports/YYYY-MM-DD.md` (nightly); manual results are recorded in the relevant bug file's "How to verify".

- [ ] **Imports and syntax of every live module** — `venv/bin/python -m pytest tests/test_imports.py -q`; Guardian
      `check_syntax` + `check_imports` via `deploy/deploy.sh`. Expect: all modules `main.py` imports load, no
      `ImportError`, deploy proceeds past Guardian. *Status: passing (the only test present today).*
- [x] **No orphan modules** — `tests/test_no_orphans.py` **[done 2026-09-27]**: every `.py` under `modules/` and `parts_used/`
      (except `__init__`/README) is imported by another repo file. Expect: passes once `esp32_controller`,
      `server_connection`, `sim7600x_modem` are gone. *Status: not implemented; 3 orphans before the deep clean.*
- [x] **Config keys all mapped and read** — `tests/test_config_keys.py` **[done 2026-09-27]**: walk `config.yaml`, assert each
      key is a dataclass field and its name appears in code outside `settings.py`. *Status: not implemented; 14 decoy keys.*
- [x] **PortAudio owner-thread guard** — `tests/test_audio_threading.py` **[done 2026-09-27]** with a fake PyAudio: open a
      `MicStream` in thread A, `close()` from thread B → `RuntimeError`; close from A succeeds; `open_input(None)` raises.
- [x] **Mic resolution by stable identity** — `tests/test_mic_resolution.py` **[done 2026-09-27]**: mock `/dev/snd/by-id` and a
      PortAudio table; cases normal / renumbered card / device missing / table without `hw:N`. Expect: the `hw:N`-matching
      index; `None` (never a default) when missing; the re-init path only when zero streams are open.
- [ ] **Two-mic startup invariant** — after deploy:
      `journalctl -u airobot -n 200 | grep -E 'device table|Resolved speech mic|Vosk wake mic open|same device'`.
      Expect: device table logged once; `Resolved speech mic 'USB PnP Sound Device' -> device <hw:0 entry>`;
      `Vosk wake mic open at 48000 Hz (device <hw:3 entry>)`; no `same device` error; `jack server is not running`
      triplet once. *Status: live journal shows speech → device 0, wake → device 2 @ 48 k; names not yet asserted.*
      **[planned]** `tools/check_audio.py` runs these greps and exits non-zero; run by `nightly.sh`.
- [ ] **Full wake → command → reply → resume cycle (two mics)** — say "Hey Stella", ask "what time is it", wait for the
      reply and the idle timeout; repeat 10×. Expect: 10/10 `Wake word detected` on the camera mic, 10/10 captures on
      the USB PnP mic, `Pausing wake-word listener` / `Wake-word listener resumed` paired each time, no `Timed out
      waiting`, no `Could not open command mic`. *Status: working since `a2e3229` 20:26 (NRestarts=0); count not yet
      measured — log the result in bug_053.*
- [ ] **Real stall reproduction** (the test the reader-thread fixes never ran) — **bench only, operator present**: start a
      conversation and, while `capture_utterance` is blocked in `stream.read()`, `sudo usbreset` (or unplug) the USB PnP
      mic; separately do the same to the camera mic while the wake stream is open. Expect: no SIGABRT; the capture ends
      at its hard deadline, or the watchdog soft-stops at 75 s and restarts at +25 s if still stuck; after replug the
      resolver finds the mic by by-id at the next open (no restart when it was not in use; after the watchdog restart
      when it was). *Status: never performed; bug_054 records only an open/read/close loop.*
- [ ] **Wake-close timeout is terminal** — unit test with a fake wake thread that never sets `_stream_closed_event`;
      live: a debug env var that delays `_close` by 5 s. Expect: `pause_listening()` returns False, the session ends
      without opening the command mic, the watchdog restarts after 10 s. *Status: today it warns and proceeds.*
- [ ] **Boot with a mic unplugged, plug later** — bench: stop `airobot`, unplug the USB PnP mic, start, wait 60 s, plug
      it in, say "Hey Stella" + a command. Expect: boot logs ERROR for the unresolved command mic with backoff (no
      `default` open); after plugging, the next capture resolves `hw:0` (via the zero-streams re-init if the snapshot
      lacked it) without a restart, or the log says why not. *Status: known to need a restart today.*
- [ ] **PulseAudio contention** — bench: `pactl set-default-source` to the USB PnP mic and start a Pulse recorder on it,
      then trigger a capture. Expect: the direct hw open succeeds or logs `device busy, held by PID <pulse pid>`; never a
      silent capture from a different device. *Status: unknown; fall-through to default is possible today.*
- [ ] **Capture gain on all capture cards** — `amixer -D hw:CARD=Camera sget Mic` and `amixer -D hw:CARD=Device sget Mic`.
      Expect: both at 100 % capture; one `gain set` journal line per card. *Status: only cards 0–1 attempted; the camera
      is never boosted.*
- [ ] **Watchdog two-stage recovery** — bench: set `_conv_active` and freeze `_conv_activity` via a debug hook. Expect:
      `Conversation stuck (no speech 75s) — soft-stopping` then, if still stuck, restart at +25 s; docstring matches.
      *Status: implemented; docstring says 120 s.* Unit test with a mocked monotonic clock **[planned]**.
- [ ] **Face recognition (YuNet, euclidean 0.60)** — master in front of the camera at normal and low light. Expect:
      recognised within 3 s, no false positive for a second person, the wave happens once. *Status: working after the
      bug_048 revert; re-check after any vision edit.*
- [ ] **Object detection** — hold a known object, ask "what do you see". Expect: correct object, no `model not found`
      (bug_041 class). **[planned]** `tests/test_paths.py` asserting every model path in config resolves to a file.
- [ ] **Hand, Telegram, face bridge, smart home** — "wave"; send a Telegram message and get the reply; open
      `http://<pi>:8080`; toggle one Tuya device by voice. Expect: all four respond, no exceptions in the journal.
- [ ] **Nightly self-check and deploy gate** — run `evolution/nightly.sh` by hand once after gitignoring the manifest;
      then `git status --porcelain --untracked-files=no`. Expect: nightly completes, tree stays clean, `deploy.sh` does
      not refuse.
- [ ] **Boot noise and probes gone** — `journalctl -u airobot -b | grep -cE 'SIM7600X|Coqui TTS not installed|Picovoice'`.
      Expect: 0 after the removals; boot no longer blocks on `/dev/ttyAMA0`. *Status: 212 occurrences each before.*
- [ ] **24 h soak after each audio change** — `systemctl show airobot -p NRestarts`;
      `journalctl -u airobot --since '24 hours ago' | grep -cE 'Could not open|Timed out waiting|status=6|malloc_consolidate'`.
      Expect: `NRestarts=0` and zero matches before the next audio change is allowed. *Status: 0 restarts since
      2026-09-26 20:26.*

---

## 14. Known debt and next steps (ordered)

1. **Legacy brain dialogue pipeline** (`core/robot_brain.py` `wake_word_heard` / `listen_for_command`, the
   `speech_recognition.py` `listen()`/`active_listener` path, `_record_audio`): not on the live path, opens mics through
   second code paths. Remove after verifying no transition is reachable by voice intent (the review found
   `patrol/observing/alert/battery_low` truly dead but `movement/learning/recovery/set_led_color` still reachable —
   remove call sites together with transitions).
2. **Device-control and guard consolidation**: one `control_device()` and one `set_guard()` owner with the single phrase
   list and one master check; the four entry points (§7) call them.
3. **Dead optional backends**: Porcupine, Coqui, DeepFace, local Whisper, `llama_cpp`, Anthropic/OpenAI branches,
   pygame/pyaudio TTS playback; pip-uninstall what nothing imports.
4. **46 unread settings fields** in `config/settings.py` (after `test_config_keys.py` lands, delete or wire each).
5. **Hailo runtime repair**: `pip uninstall hailort`; install the matched pyhailort wheel from the h10-hailort bundle
   (not PyPI); only then evaluate NPU vision vs single-context contention with `hailo-ollama`. Object detection stays on
   the CPU until then.
6. **`MqttConfig`** dataclass; `main.py` stops reading raw YAML; hub does not start with zero devices.
7. Audio hardening in §4.7 order (MicStream, resolver, terminal timeout, startup invariant, config ids, rates, gains,
   device table, comments, tests).
8. Hardware: PSU rating / power-offs with Ethernet plugged (use the official 27 W supply), a fan (80 °C soft limit under
   load), INMP441 I²S mics as the durable cure for USB-audio fragility, re-enrol the master face under YuNet.
9. Voice: openWakeWord "Hey Stella" (needs Moti's voice clips), faster-whisper offline STT ahead of Vosk.
10. Evolution v3: Lab (sandbox) + Builder behind `evolution.require_operator_approval`.
