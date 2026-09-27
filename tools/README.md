# tools — operator scripts

Run from the repo root with the venv Python (`venv/bin/python`). None of these run inside the service.

| Script | Use |
|---|---|
| `reenroll.py` | Re‑enrol the master face (speaks guidance, 40 s cap, keeps other people). **Stop `airobot` first** so the camera is free. |
| `enroll_master.py` | Original enrolment tool (interactive / `--auto`). Prefer `reenroll.py`. |
| `manage_faces.py` | `list` / `remove <id or name>` / `rename <id> <new name>` on the face DB (`data/faces/`). |
| `check_deps.py` | Import every Python dependency and report what is missing. Deep‑clean fix: its data directory is the repo `data/` (was `tools/data`, so it always reported the Vosk model missing — same class as bug_041). |
| `handctl.py` | Send raw commands to the ESP32 hand over serial (stop `airobot` first; it owns the port). Protocol: `docs/firmware/hand-esp32.md`. |
| `check_audio.py` (planned) | Greps the journal for the two‑mic startup invariant (device table once, speech mic → hw:0, wake mic → hw:3 @ 48 kHz, no "same device"), checks capture gain on both cards, exits non‑zero on failure; to be run by `evolution/nightly.sh`. |

Removed 2026-09-27: `test_sim7600x.py` (4G modem not fitted; module deleted).

```bash
sudo systemctl stop airobot && venv/bin/python tools/reenroll.py && sudo systemctl start airobot
venv/bin/python tools/manage_faces.py list
venv/bin/python tools/check_deps.py
```
