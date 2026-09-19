# Continue Point — resume here next session

_Last updated: 2026-09-12 (evening, IDT)_

Pick up from here. Full rationale for every change is in `docs/decisions/log.md`.

## Environment quick-reference
- **Pi:** `ssh rpi5` (192.168.11.204 on home WiFi "Arik -2.4G-ext"). Repo: `~/AIRobot_v2.0`, branch `jetson_new_v1`, remote `motisAi/AIRobot_v2.0` (push as motisAi).
- **Service:** `airobot`. Only `sudo -n systemctl restart/stop airobot` is passwordless. Read logs with `journalctl -u airobot`. Rule: **read the log first** when something "doesn't work".
- **Local mirror (Windows):** `C:\Users\Moti\AIRobot_work` (kept in sync via scp; not the git repo).
- **Secrets (gitignored, never commit):** `.env`, `devices.json`, `snapshot.json`, `tinytuya.json`, `tuya-raw.json`.
- **Camera now:** USB "Auto Focus Camera" (auto-detected). Pi CSI camera is UNPLUGGED. Code skips CSI nodes.
- **Windows face kiosk:** `face_client.ps1` runs at login via Startup shortcut `StellaFace.lnk`; Pi IP cached in `%LOCALAPPDATA%\stella-face-pi.txt`.

## What works (verified 2026-09-12)
- STT = Groq Whisper (whisper-large-v3-turbo) primary → Google → Vosk offline.
- Voice + natural-language commands: lights (Tuya plug), AC (Sensibo), show/hide face, wave, vision Q&A ("what am I holding?").
- Music: correct song + actually plays (yt-dlp android client → ffmpeg pipe; smart search filters tutorials/covers).
- Single welcome greeting (no double / no spam). Guard arms and stays armed when you leave.
- Face kiosk opens/closes on voice. Camera auto-detects USB, survives reboots. No more CSI console spam.
- RobotNet (dongle AP), MQTT broker, Tuya plug, Sensibo AC.

## TODO next session (in priority order)
1. **Boot WiFi auto-reconnect** — after a reboot the Pi's onboard WiFi uplink sometimes doesn't reconnect ("no internet" until `sudo netplan apply`). Set up a tiny boot/periodic reconnect (systemd unit or networkd-dispatcher hook). NEEDS Moti's sudo password once (run on the Pi keyboard or paste when prompted).
2. **"Stop replying to silence"** — Whisper hallucinates short phrases ("Thank you.", ".") from ambient noise, so she answers nobody. Add a min-voiced-duration gate (require ~0.5s of voiced frames) and/or a hallucination-phrase blocklist in `speech_recognition.capture_utterance`.
3. **Face recognition dips** — sim fell to ~0.47–0.58 (was 0.65–0.72); she occasionally asks "who's speaking?". Consider re-enroll (`python tools/reenroll.py` with service stopped) under current USB-cam lighting, or nudge threshold.

## Nice-to-have / later
- Wire the Pi CSI camera properly via libcamera/picamera2 if we want to switch off the USB cam.
- DHCP-reserve the Tuya plug IP; move plug to RobotNet if portability wanted.
- Boss has NOT authorized Claude API — do not wire it.

## How to restart / check her
```
ssh rpi5 'sudo -n systemctl restart airobot'
ssh rpi5 'journalctl -u airobot -n 80 --no-pager'
```
