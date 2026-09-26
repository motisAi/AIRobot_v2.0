# Continue point — 2026-09-27

## Where we are
- **One architecture document**: [architecture/stella-architecture.md](architecture/stella-architecture.md) replaces the
  two dated docs (now in `architecture/archive/`) and the old `docs/architecture.md`. Read §4 before touching audio,
  §12 before touching anything; §13 is the verification checklist.
- **Audio is stable on the two-mic layout** (commit `a2e3229`, 2026-09-26 20:26): wake = camera built-in mic
  ("Auto Focus Camera", card `Camera`, hw:3, 48 kHz), command = "USB PnP Sound Device" (card `Device`, hw:0, 44.1 kHz).
  0 restarts, zero `Could not open` / `Timed out` lines since. Single-mic handoff failed (bug_055) — never retry it.
  No reader thread for mic capture — it crashed (bug_054) and leaked the device; `capture_utterance()` is
  single-thread + hard deadline, a wedged read is bounded by the two-stage watchdog (75 s soft stop, +25 s restart).
- **Deep clean 2026-09-27**: removed `parts_used/esp32_controller.py`, `parts_used/sim7600x_modem.py`,
  `modules/connectivity/`, `tools/test_sim7600x.py`, `config/config.json`, `claude_read.txt`, the `--test` /
  `--create-service` code in `main.py`, `docs/{architecture.md,GONZO_GUIDE.md,rpi5_setup.md,service_accounts.txt}`
  (apt list kept in `hardware/hailo.md`); `stella_manifest.yaml` untracked + gitignored. Guardian green.
- **bug_report/**: 55 files (001–055, no gaps; 050 retired). bug_049 is `REVERTED`, bug_053 `verified`, bug_055 records
  the misdiagnosis. Rule: no fix without a bug file in the same commit; a reverted fix is edited in place.
- **Safety net**: `evolution/guardian.py` + `deploy/deploy.sh` (clean tree → restart → Guardian → auto-rollback). Use it
  for EVERY deploy. Nightly self-check at 03:00 via the **user crontab** (`crontab -l`); the systemd timer in
  `deploy/stella-evolution.*` is an optional sudo alternative and is not installed.
- **Hailo**: driver rebuilt with `linux-headers-raspi` installed → survives kernel upgrades. The NPU serves only the
  offline LLM (qwen2.5 1.5B, ~0.7 s warm); object detection is CPU OpenCV-DNN (`yolov8n.onnx`) because the pip
  `hailort` wheel is ABI-broken.
- **Vision**: YuNet detector + dlib, euclidean threshold 0.60; identity debounced (bug_047); phantom faces filtered (bug_045).
- **Memory**: per-person durable facts distilled at session end and recalled next time (`6cf5416`).

## Queued code fixes (each its own Guardian-deployed commit, with its test; order in the architecture doc §4.7 / §14)
1. `MicStream` owner-thread guard in `parts_used/audio_portaudio.py`, used at both open sites.
2. One resolver on stable identity (`audio_devices.find_input_index()`: by-id → `controlC<N>` → PortAudio `(hw:N,`);
   never `input_device_index=None`; `wake_mic_id` / `command_mic_id` config keys.
3. Wake-close timeout terminal (no command open, watchdog escalation); startup assertion wake ≠ command device.
4. Rates from config; capture gain on every capture card by id; shared PyAudio device table logged once, `AudioManager`
   private init removed.
5. `config.json` fallback removed with a loud YAML failure; known-good dataclass defaults; SIM7600X import-time probe
   removed; `MqttConfig`; watchdog docstring (75 s) + wake-listener visibility check.
6. Tests: `test_no_orphans`, `test_config_keys`, `test_audio_threading`, `test_mic_resolution`; Guardian `tests` step.
7. Small: `tools/check_deps.py` data path, `docs/firmware/hand-esp32.md` → `tools/handctl.py`, dead `.gitignore`
   lines, `gonzo.log` → `stella.log`.

## Open (needs Moti)
1. **PSU / brownout**: power-offs when the Ethernet cable is plugged in — no undervoltage flag on the readable boots, the
   log just stops (hard cut). Need the PSU label rating; use the official 27 W supply. Guardian reports undervoltage bits.
2. **Fan**: hits the 80 °C soft limit under load (`throttled=0x80000`). A fan or a case with airflow is the fix.
3. **INMP441 I²S mics**: not yet wired — the durable cure for USB-audio fragility; a mic-role change afterwards needs a
   bug file, a decision-log entry and a 24 h soak (architecture doc §4, I8).
4. **Re-enrol the master face** under YuNet / current lighting: `sudo systemctl stop airobot && venv/bin/python
   tools/reenroll.py && sudo systemctl start airobot` (bug_048 verification).
5. Office WiFi netplan (SSID Politech-Internal-2.4GHz, fixed 192.168.0.240/24) — needs sudo. `guard_mode` not persisted
   across restarts. `git push`: sign in once via VS Code Source Control if the token is dead.

## Bench tests still to run (architecture doc §13)
- Real stall reproduction: `usbreset`/unplug the USB PnP mic while a read is blocked (operator present, bench only).
- Boot with the command mic unplugged, plug later; PulseAudio contention; the 10-cycle wake → command → resume count
  (log it in bug_053).

## Next build steps
- Scout's top ideas: faster-whisper offline STT, openWakeWord "Hey Stella" (needs Moti's voice clips), Hailo model zoo
  after the HailoRT runtime repair (matched pyhailort wheel, not PyPI).
- Remove the legacy brain dialogue pipeline and collapse the four device-control/guard writers into one owner (§14).
- Evolution v3: Lab (sandbox) + Builder behind `require_operator_approval`.
