# tools — operator scripts

Run from the repo root with the venv Python. None of these run inside the service.

| Script | Use |
|---|---|
| `reenroll.py` | Re‑enrol the master face (speaks guidance, 40 s cap, keeps other people). **Stop `airobot` first** so the camera is free. |
| `enroll_master.py` | Original enrolment tool (interactive / `--auto`). Prefer `reenroll.py`. |
| `manage_faces.py` | `list` / `remove <id or name>` / `rename <id> <new name>` on the face DB. |
| `check_deps.py` | Import every Python dependency and report what is missing. |
| `handctl.py` | Send raw commands to the ESP32 hand over serial (stop `airobot` first; it owns the port). |
| `test_sim7600x.py` | Probe the 4G modem (not fitted today). |

```bash
sudo systemctl stop airobot && venv/bin/python tools/reenroll.py && sudo systemctl start airobot
venv/bin/python tools/manage_faces.py list
```
