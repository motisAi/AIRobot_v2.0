# evolution — Stella's self‑improvement system

> **DISABLED (operator decision, 2026-09-27):** no autonomous runs. `evolution/AUTORUN_DISABLED` exists and the cron entry was removed; `nightly.sh` exits immediately while the marker is present. Guardian is still used by `deploy/deploy.sh` on every manual deploy. Nothing here ever installed packages or edited code (Scout is report-only; Lab/Builder were never built).

Goal: Stella knows what she is, checks her own health, looks for ways to get better, and
tells Moti every morning — using **free models only** (hailo‑ollama → Groq → Gemini; never
a paid API) and **never changing herself without a safety net**. Design:
[docs/architecture/stella-architecture.md §10](../docs/architecture/stella-architecture.md#10-evolution-and-deploy).

| File | Role | Status |
|---|---|---|
| `guardian.py` | **Deterministic health check** (no LLM): syntax, imports, config, service up & stable, log errors/fatal patterns, hardware present, offline brain answers, disk/RAM/temperature/**undervoltage**. Exit 0/1, JSON report in `reports/guardian-latest.json`. A `tests` step that runs `pytest tests/ -q` is being added in the 2026-09-27 deep clean. | live |
| `manifest.py` | Builds `stella_manifest.yaml` (repo root, **generated — gitignored**, never commit it): hardware detected now, models on disk, providers with keys, services, capabilities from config, recurring bugs from `bug_report/`. | live |
| `db.py` | `evolution.db` (SQLite, gitignored): `candidates`, `runs`, `llm_calls`. | live |
| `llm_client.py` | Free‑only LLM for the agents; every call logged with purpose + tokens. | live |
| `scout.py` | Looks for upgrades (PyPI versions of AI packages, GitHub releases of Piper/Vosk/openWakeWord/faster‑whisper/Ultralytics, Hailo model zoo), scores relevance with the LLM against the manifest. **Report‑only — installs nothing.** | live |
| `report.py` | Morning report `reports/YYYY-MM-DD.md` + 5‑line Telegram summary (Guardian, power/thermal events, errors, Scout findings, open recurring bugs). | live |
| `nightly.sh` | 03:00 daily: skip if an SSH connection is established, the tree is dirty, load ≥ 3 or disk < 2 GB; then manifest → Guardian → Scout (900 s cap) → report under `nice -n 15`; log in `reports/nightly.log`. **Scheduled by the user crontab** — `crontab -l` shows `0 3 * * * /home/moti_ai/AIRobot_v2.0/evolution/nightly.sh`. That crontab line is the one and only scheduler. | live |
| `deploy/stella-evolution.service` + `.timer` | systemd equivalent of the crontab line — an **optional sudo alternative, not installed** (`systemctl list-timers` shows no stella timer). Do not install both. | optional |
| Lab (sandbox tests) / Builder (auto‑merge) / hardware inbox | **Deferred to v3** on purpose: a small model must not install packages into a running robot unsupervised. When added they sit behind `evolution.require_operator_approval: true` (approval by Telegram reply). | planned |

## Deploying safely
```bash
git commit -am "..."          # the commit IS the rollback point (deploy.sh refuses a dirty tree)
deploy/deploy.sh              # restart → Guardian --wait → if unhealthy: git reset --hard HEAD~1 + restart + Telegram
deploy/deploy.sh <good-sha>   # roll back to a specific commit on failure
```
Guardian alone: `venv/bin/python evolution/guardian.py [--wait] [--require-hailo] [--no-brain]`.
One audio change per day, then a 24 h soak (`NRestarts=0`, zero `Could not open` lines) before the next.

## Rules
1. Guardian never uses an LLM. Pass/fail is a script.
2. Nothing in `evolution/` edits code, config, `.env`, systemd or the network. Scout writes to
   the database and the report only.
3. Runs pause while the operator is working (open SSH session or dirty git tree).
4. Secrets are read from `.env` only, never logged.
5. Generated files (`stella_manifest.yaml`, `reports/*.json`, `evolution.db`) are gitignored so a nightly run can
   never dirty the tree and disable itself or `deploy.sh`.
