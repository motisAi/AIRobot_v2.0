# Continue point — 2026-09-19

## Where we are
- **Layout reorganised** (commit `e6a5b2e`): `parts_used/` one file per physical part, `modules/` by capability
  (`smart_home/`, `media/`, `comms/`), `core/watchdog.py`, `tools/`. Stale root duplicates deleted. Guardian green.
- **Safety net live**: `evolution/guardian.py` + `deploy/deploy.sh` (restart → Guardian → auto-rollback). Use it for EVERY deploy.
- **bug_report/**: 40 bugs from history. Rule: no fix without a bug file.
- **evolution/**: manifest (`stella_manifest.yaml`), `evolution.db`, free-only LLM client (Groq needs a browser User-Agent),
  Scout (report-only), morning report + Telegram. Nightly at 03:00 via the user crontab (`crontab -l`);
  systemd units in `deploy/stella-evolution.*` are the sudo alternative.
- **RC toy**: `parts_used/rc_toy.py`, `rc_toy.connected: false`, tool `drive_toy` appears only when connected (master-only).
- **Hailo**: driver rebuilt 2026-09-19 with `linux-headers-raspi` installed → survives kernel upgrades. Offline brain 0.7 s.
- **Slow/unprompted fixes** (`231740d`) verified live: 0 errors, no self-talk.

## Open (needs Moti)
1. `git push` — the GitHub token on the Pi and on the PC is dead. Sign in once via VS Code Source Control
   (Remote SSH → /home/moti_ai/AIRobot_v2.0 → Sync).
2. Power-offs when the Ethernet cable is plugged in: no undervoltage flag on the boots we could read; the logs just stop
   (hard power cut). Need the PSU label rating; use the official 27 W supply. Guardian now reports undervoltage bits loudly.
3. No fan: hits the 80 °C soft limit under load (`throttled=0x80000`). A fan or a case with airflow is the fix.
4. Office WiFi netplan (SSID Politech-Internal-2.4GHz, fixed 192.168.0.240/24) — needs sudo.
5. Face re-enrol under current lighting (`tools/reenroll.py`); guard_mode not persisted across restarts.

## Next build steps (from the architecture)
- Scout's top ideas: faster-whisper offline STT, openWakeWord "Hey Stella" (needs Moti's voice clips),
  Hailo model zoo after HailoRT 5.2.
- Evolution v3: Lab (sandbox) + Builder behind `require_operator_approval`.
