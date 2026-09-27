# Stella — Bug History

One file per bug that was actually hit **and fixed** (features and design decisions live in `docs/decisions/log.md`).
Chronological, oldest first, numbered with no gaps. Index generated 2026-09-27 from each file's header
(number, date found, title, area, status). File paths inside the older entries are historical by design; the
CURRENT layout and the canonical audio design are in
[docs/architecture/stella-architecture.md](../docs/architecture/stella-architecture.md).

Status key: **fixed** = deployed and believed fixed; **verified** = checked live afterwards; **fixed-needs-verify** =
deployed but not yet checked live; **recurring** = fixed each time but will happen again (trigger in the last column);
**REVERTED** = the fix was undone — the entry says why and what replaced it; do not reintroduce it;
**retired** = number never assigned.

| # | Date | Title | Area | Status | Superseded by / recurring trigger |
|---|------|-------|------|--------|-----------------------------------|
| [001](bug_001_face-rec-systemexit-missing-models.md) | 2026-05-30 | face_recognition quit() killed process when dlib models missing | vision | fixed | — |
| [002](bug_002_mic-sample-rate-mismatch.md) | 2026-05-30 | Wake/speech mic opened at rate the USB mic rejects | audio | fixed | new mic with other native rates |
| [003](bug_003_logging-stuck-warning-rms-overflow.md) | 2026-05-30 | Root logger stuck at WARNING; numpy RMS overflow | deploy | fixed | — |
| [004](bug_004_face-rec-cropped-frame-to-dlib.md) | 2026-05-30 | Cropped face passed to dlib instead of full frame + location | vision | fixed | — |
| [005](bug_005_brain-speech-pipeline-miswired.md) | 2026-05-30 | Brain speech pipeline mis-wired (wrong module, double listen) | conversation | fixed | — |
| [006](bug_006_robotevent-priorityqueue-comparison.md) | 2026-05-30 | RobotEvent not comparable in PriorityQueue | conversation | fixed | — |
| [007](bug_007_cpu-temp-threshold-too-low.md) | 2026-05-30 | CPU temp threshold too low (raised 80 -> 85 C) | power | fixed | — |
| [008](bug_008_face-detection-spam.md) | 2026-05-31 | Face detection spam (tiny faces, per-frame events) | vision | fixed | — |
| [009](bug_009_mic-device-index-changes-on-reboot.md) | 2026-05-31 | Mic device index wrong / changes on reboot | audio | fixed | PyAudio enumeration flaky; resolve by stable identity (architecture §4) |
| [010](bug_010_wake-stream-not-released-on-pause.md) | 2026-05-31 | Wake-word stream not released on pause; speech mic busy | audio | fixed | — |
| [011](bug_011_hailo-driver-missing-after-kernel-upgrade.md) | 2026-09-05 | Hailo DKMS driver gone after kernel upgrade (no /dev/hailo0) | hardware | **recurring** | kernel upgrade; `linux-headers-raspi` installed 2026-09-19 so DKMS rebuilds; else `sudo dkms autoinstall` |
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
| [026](bug_026_phantom-replies-stt-hallucination.md) | 2026-09-12 | Phantom replies from STT hallucination ("Nice to meet you, Foreign") | conversation | fixed-needs-verify | new hallucination strings -> extend stoplist |
| [027](bug_027_blocking-mic-read-hangs-conversation.md) | 2026-09-13 | Blocking mic read hung conversation; deaf for 10 min | audio | fixed (watchdog recovery) | its reader-thread fix caused 028 and was reverted |
| [028](bug_028_cross-thread-portaudio-close-sigabrt.md) | 2026-09-13 | Cross-thread PortAudio close -> SIGABRT crash loop | audio | fixed | same class recurred as 054 on 2026-09-26 |
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
| [041](bug_041_reorg-broke-model-path-object-detection.md) | 2026-09-19 | Reorg moved hailo_10h.py; object detection lost data/models | vision | fixed | moving files with parent.parent root hacks |
| [042](bug_042_polite-goodbye-not-recognised-slow-llm-roundtrip.md) | 2026-09-19 | "No, thank you, Stella." not a goodbye -> 5 s LLM round trip | conversation | fixed | — |
| [043](bug_043_groq-120b-rejects-send_photo-tool-schema.md) | 2026-09-19 | Groq 120b rejects send_photo tool schema (400), falls back | ai | fixed (3210df1) | — |
| [044](bug_044_robotnet-dongle-driver-not-loaded-after-kernel-upgrade.md) | 2026-09-19 | RobotNet dead: 8821au DKMS module missing after kernel upgrade | network | **recurring** | kernel upgrade without headers; `dkms autoinstall` + modprobe |
| [045](bug_045_phantom-new-person-on-wall.md) | 2026-09-19 | "New person" greeting with nobody there (phantom face on a wall) | vision | fixed | — |
| [046](bug_046_wake-mic-stall-no-response.md) | 2026-09-19 | "No response to Hey Stella" after a while (wake stream went silent) | audio | fixed | **superseded by 053**: the wake-mic-on-USB-dongle part was reverted (d3b2577); self-heal stands |
| [047](bug_047_master-identity-flip-single-frame.md) | 2026-09-19 | Master identity lost repeatedly (cleared on a single Unknown frame) | vision | fixed | — |
| [048](bug_048_yunet-dlib-recognition-regression.md) | 2026-09-19 | Recognition similarity collapsed after switching to YuNet | vision | fixed-needs-verify | re-enroll pending |
| [049](bug_049_unbounded-mic-read-stuck.md) | 2026-09-19 | Hard "stuck / no response" + 30 s blackout (unbounded mic read) | audio | **REVERTED** | reader thread reverted in 442c8a8; see 054/055; single-thread read + hard deadline + two-stage watchdog is the current mechanism |
| [050](bug_050_number-retired.md) | n/a | Number retired — never assigned (numbering gap closed 2026-09-27) | n/a | retired | — |
| [051](bug_051_sensibo-422-stale-fields.md) | 2026-09-19 | Sensibo AC returns 422 on a mode/temperature change | smart-home | fixed | — |
| [052](bug_052_wifi-interface-rename-race.md) | 2026-09-20 | Stella off home WiFi: onboard radio renamed wlan0 -> wlan1 (boot race) | network | fixed | system netplan (dual-name) — outside the repo |
| [053](bug_053_wake-mic-mistranscribe-usb.md) | 2026-09-26 | "Hey Stella" not detected: wake mic mistranscribes on the USB dongle mic | audio | verified | wake = camera mic is an invariant (architecture §4) |
| [054](bug_054_sigabrt-reader-thread-crashloop.md) | 2026-09-26 | SIGABRT crash-loop every ~2 min ("malloc: unaligned tcache chunk") | audio | fixed | **superseded by 055**: first fix ed73b10 reverted 12 min later (442c8a8); final = single-thread capture |
| [055](bug_055_usb-mic-misdiagnosed-single-mic-detour.md) | 2026-09-26 | USB PnP mic misdiagnosed as wedged; single-mic detour failed (real cause: our leaked reader thread) | audio | fixed | single-mic handoff is unsupported; two-mic layout restored a2e3229 |

## Still open (no bug file — not fixed yet)
- Power-offs when the Ethernet cable is plugged in: no undervoltage flag on the readable boots, the log just stops
  (hard power cut). Needs the PSU label rating; use the official 27 W supply. Guardian reports undervoltage bits loudly.
- No fan: the Pi hits the 80 °C soft limit under load (`throttled=0x80000`).
- INMP441 I²S mics not yet wired — the durable cure for USB-audio fragility (002, 009, 010, 027, 037, 046, 053).
- Face re-enrol under YuNet / current lighting (`tools/reenroll.py`) — bug_048 verification.
- `guard_mode` not persisted across service restarts.
- Boot WiFi uplink sometimes needs `sudo netplan apply`; office WiFi netplan (needs sudo).
- Rare `aplay` error 524 (device busy) — watchdog-handled.

## How to add a bug
Copy `_TEMPLATE.md` to `bug_NNN_<short-kebab-slug>.md` using the **next free number — sequential, no gaps** — in the
same commit as the fix (a commit message may not cite a bug number that does not exist here). Fill every field (write
"unclear from history" rather than guessing), keep it under ~60 lines, and add a row to the table above. Only bugs that
were actually hit and fixed belong here; design decisions go in `docs/decisions/log.md`.

**A reverted fix is edited IN PLACE** — the original file's Status becomes `REVERTED — do not reintroduce, see bug_0XX`,
its `Reverted:` field names the reverting commit, and its Fix section gains the current mechanism. Recording the revert
only in a later entry is not enough: that is how the reader-thread mistake was made twice (027/028 on 2026-09-13, then
049/054/055 on 2026-09-19..26). The decision-log entry that introduced the fix gets the same banner at its top, and the
docstring at the code site cites both bug numbers.

## Verification notes
- 2026-09-19: bugs 034–040 verified live after the reorganization deploy (Guardian: 0 errors, offline brain 0.7 s, no
  unprompted speech in a 2-hour idle run). Bug 011 recurred on kernel 1064 and was fixed the same day with the permanent
  `linux-headers-raspi` install, so future kernel upgrades rebuild the driver automatically.
- 2026-09-26: after a2e3229 (20:26, two-mic layout restored) the journal shows the wake stream on the camera mic
  (`Vosk wake mic open at 48000 Hz (device 2)`), the command mic resolved to the USB PnP device, a wake at 21:12:09 with
  paired pause/resume, 0 `Could not open` / `Timed out` / SIGABRT lines and `NRestarts=0` — bug_053 verified, bug_055 closed.
