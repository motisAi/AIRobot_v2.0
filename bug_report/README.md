# Stella — Bug History

One file per bug that was actually hit **and fixed** (features and design decisions live in `docs/decisions/log.md` on the Pi). Sources: `docs/decisions/log.md`, `git log` (branch `jetson_new_v1`), `docs/hardware/hailo.md`, `docs/CONTINUE.md`, and the 2026-09-16 diagnosis notes. Chronological, oldest first. Compiled 2026-09-19.

Status key: **fixed** = verified; **fixed-needs-verify** = deployed but not yet checked live; **recurring** = fixed each time but will happen again. File paths are the CURRENT layout (after the 2026-09-19 reorganization), even for older bugs.

| # | Date | Title | Area | Status | Recurring trigger |
|---|------|-------|------|--------|-------------------|
| [001](bug_001_face-rec-systemexit-missing-models.md) | 2026-05-30 | face_recognition quit() killed process when dlib models missing | vision | fixed | — |
| [002](bug_002_mic-sample-rate-mismatch.md) | 2026-05-30 | Wake/speech mic opened at rate the USB mic rejects | audio | fixed | new mic with other native rates |
| [003](bug_003_logging-stuck-warning-rms-overflow.md) | 2026-05-30 | Root logger stuck at WARNING; numpy RMS overflow | deploy | fixed | — |
| [004](bug_004_face-rec-cropped-frame-to-dlib.md) | 2026-05-30 | Cropped face passed to dlib instead of full frame + location | vision | fixed | — |
| [005](bug_005_brain-speech-pipeline-miswired.md) | 2026-05-30 | Brain speech pipeline mis-wired (wrong module, double listen) | conversation | fixed | — |
| [006](bug_006_robotevent-priorityqueue-comparison.md) | 2026-05-30 | RobotEvent not comparable in PriorityQueue | conversation | fixed | — |
| [007](bug_007_cpu-temp-threshold-too-low.md) | 2026-05-30 | CPU temp threshold too low (raised 80 -> 85 C) | power | fixed | — |
| [008](bug_008_face-detection-spam.md) | 2026-05-31 | Face detection spam (tiny faces, per-frame events) | vision | fixed | — |
| [009](bug_009_mic-device-index-changes-on-reboot.md) | 2026-05-31 | Mic device index wrong / changes on reboot | audio | fixed | PyAudio enumeration flaky; restart service |
| [010](bug_010_wake-stream-not-released-on-pause.md) | 2026-05-31 | Wake-word stream not released on pause; speech mic busy | audio | fixed | — |
| [011](bug_011_hailo-driver-missing-after-kernel-upgrade.md) | 2026-09-05 | Hailo DKMS driver gone after kernel upgrade (no /dev/hailo0) | hardware | **recurring** | every kernel upgrade until `linux-headers-raspi` installed; `sudo dkms autoinstall` |
| [012](bug_012_hailo-ollama-api-chat-empty.md) | 2026-09-05 | hailo-ollama /api/chat returns empty; use /v1/chat/completions | ai | fixed | hailo-ollama upgrade |
| [013](bug_013_cloud-llm-models-retired-404.md) | 2026-09-05 | Cloud LLM models retired (404); blank Gemini key | ai | **recurring** | provider retires a model id |
| [014](bug_014_face-rec-threshold-too-strict.md) | 2026-09-05 | Face-rec threshold 0.50 too strict; master = "Unknown" | vision | fixed | lighting/camera change -> re-enroll |
| [015](bug_015_google-stt-failing-fell-to-vosk.md) | 2026-09-12 | Google STT failing 57x, dropped to weak Vosk | audio | fixed | Groq rate-limit |
| [016](bug_016_light-command-routed-to-missing-relay.md) | 2026-09-12 | "Turn on the light" routed to non-existent MCU relay | smart-home | fixed | adding a 2nd Tuya device; plug IP change |
| [017](bug_017_sensibo-rate-limit-429.md) | 2026-09-12 | Sensibo 429 rate limit on rapid AC commands | smart-home | fixed | back-to-back commands |
| [018](bug_018_double-greeting-welcome-spam.md) | 2026-09-12 | Double greeting + welcome spam -> aplay busy, "frozen" | conversation | fixed | — |
| [019](bug_019_camera-dead-after-reboot-csi-renumbered.md) | 2026-09-12 | Camera dead after reboot: CSI renumbered USB webcam | vision | fixed | — |
| [020](bug_020_csi-kernel-console-spam.md) | 2026-09-12 | Console spam "rp1-cfe csi2 node link is not enabled" | vision | fixed | — |
| [021](bug_021_guard-auto-disarmed-instantly.md) | 2026-09-12 | "Guard my house" disarmed itself after 6 s | conversation | fixed | — |
| [022](bug_022_face-kiosk-never-opened.md) | 2026-09-12 | Face kiosk never opened on PC (controller not running / wedged) | face-ui | fixed | Startup shortcut removed |
| [023](bug_023_music-wrong-song.md) | 2026-09-12 | Music played wrong video (tutorials/covers) | audio | fixed | obscure queries |
| [024](bug_024_music-no-audio-youtube-403.md) | 2026-09-12 | Music: no sound, YouTube 403 on stream URL | audio | fixed | YouTube client changes; `pip install -U yt-dlp` |
| [025](bug_025_vosk-never-hears-stella.md) | 2026-09-13 | Vosk never transcribes "Stella"; wake word dead | audio | fixed | new Vosk misspelling -> add alias |
| [026](bug_026_phantom-replies-stt-hallucination.md) | 2026-09-13 | Phantom replies from STT hallucination ("Nice to meet you, Foreign") | conversation | fixed-needs-verify | new hallucination strings -> extend stoplist |
| [027](bug_027_blocking-mic-read-hangs-conversation.md) | 2026-09-13 | Blocking mic read hung conversation; deaf for 10 min | audio | fixed (watchdog recovery) | flaky USB mic; watchdog restarts |
| [028](bug_028_cross-thread-portaudio-close-sigabrt.md) | 2026-09-13 | Cross-thread PortAudio close -> SIGABRT crash loop | audio | fixed | — |
| [029](bug_029_guard-off-command-armed-guard.md) | 2026-09-13 | "Guard mode off" armed guard; status questions armed it | conversation | fixed | — |
| [030](bug_030_face-bridge-access-log-spam.md) | 2026-09-13 | Journal flooded by aiohttp access logs (face polling) | face-ui | fixed | — |
| [031](bug_031_master-auth-bypass-keyword-handlers.md) | 2026-09-13 | Master-auth bypass in _maybe_ac / _maybe_mirror | conversation | fixed | new keyword handler without gate |
| [032](bug_032_object-recognition-dead-three-blockers.md) | 2026-09-13 | Object recognition dead (no model, Hailo off, wrong hailort) | vision | fixed | fresh clone (onnx gitignored) |
| [033](bug_033_object-detected-handlers-crash-event-flood.md) | 2026-09-13 | object_detected handlers crashed + event flood | vision | fixed | — |
| [034](bug_034_offline-cliff-llm-chain-90s.md) | 2026-09-16 | Offline cliff: ~90 s per answer with no internet | ai | fixed | — |
| [035](bug_035_false-lost-internet-announcement.md) | 2026-09-16 | False "lost internet" announcement on offline boot | network | fixed | — |
| [036](bug_036_no-mic-monologue.md) | 2026-09-16 | No mic -> Stella monologues to herself | conversation | fixed | — |
| [037](bug_037_wake-mic-retry-error-spam.md) | 2026-09-16 | Wake mic retried at 1 Hz with ERROR spam | audio | fixed | — |
| [038](bug_038_tts-dead-hdmi-sink-pygame-hang.md) | 2026-09-16 | TTS pinned to dead HDMI sink; pygame startup hang regression | audio | fixed | — |
| [039](bug_039_yolo-every-frame-no-camera.md) | 2026-09-16 | YOLO on every frame, loaded with no camera | vision | fixed | — |
| [040](bug_040_wall-clock-timers-false-stuck-restart.md) | 2026-09-16 | Wall-clock timers -> false "stuck" restart after NTP jump | hardware | fixed | — |

## Still open (no bug file — not fixed yet)
- Boot WiFi uplink sometimes does not reconnect ("no internet" until `sudo netplan apply`) — needs a boot/periodic reconnect unit (needs sudo).
- `guard_mode` not persisted across service restarts.
- Rare `aplay` error 524 (device busy) — watchdog-handled; proper fix is pulseaudio access for the systemd unit (routing TTS via `pulse` was tried and reverted 2026-09-05).
- Face-recognition similarity dipped to ~0.45-0.58 under USB-cam lighting — re-enroll pending.
- Underlying USB audio flakiness (xrun / paInvalidSampleRate) — mitigated, not solved.
- Bug #011's permanent fix (`sudo apt install linux-headers-raspi ...`) still needs Moti's sudo as of the last log entry.

## How to add a bug
Copy `_TEMPLATE.md` to `bug_NNN_<short-kebab-slug>.md` using the next free number (chronological by date found), fill every field (write "unclear from history" rather than guessing), keep it under ~60 lines, and add a row to the table above. Only bugs that were actually hit and fixed belong here; design decisions go in `docs/decisions/log.md`.

## Verification notes
- 2026-09-19: bugs 034–040 verified live after the reorganization deploy (Guardian: 0 errors, offline brain 0.7 s, no unprompted speech in a 2-hour idle run). Bug 011 recurred on kernel 1064 and was fixed the same day with the permanent `linux-headers-raspi` install, so future kernel upgrades rebuild the driver automatically.
