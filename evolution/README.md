# evolution — Stella's self‑improvement system

Goal: Stella knows what she is, checks her own health, looks for ways to get better, and
tells Moti every morning — using **free models only** (hailo‑ollama → Groq → Gemini; never
a paid API) and **never changing herself without a safety net**. Design: §4–§5 of
[docs/architecture/stella-architecture-2026-09-19.md](../docs/architecture/stella-architecture-2026-09-19.md).

| File | Role | Status |
|---|---|---|
| `guardian.py` | **Deterministic health check** (no LLM): syntax, imports, config, service up & stable, log errors/fatal patterns, hardware present, offline brain answers, disk/RAM/temperature/**undervoltage**. Exit 0/1, JSON report in `reports/guardian-latest.json`. | live |
| `manifest.py` | Builds `stella_manifest.yaml` (repo root): hardware detected now, models on disk, providers with keys, services, capabilities from config, recurring bugs from `bug_report/`. | live |
| `db.py` | `evolution.db` (SQLite): `candidates`, `runs`, `llm_calls`. | live |
| `llm_client.py` | Free‑only LLM for the agents; every call logged with purpose + tokens. | live |
| `scout.py` | Looks for upgrades (PyPI versions of AI packages, GitHub releases of Piper/Vosk/openWakeWord/faster‑whisper/Ultralytics, Hailo model zoo), scores relevance with the LLM against the manifest. **Report‑only — installs nothing.** | live |
| `report.py` | Morning report `reports/YYYY-MM-DD.md` + 5‑line Telegram summary (Guardian, power/thermal events, errors, Scout findings, open recurring bugs). | live |
| `nightly.sh` + `deploy/stella-evolution.timer` | 03:00 daily: skip if an SSH session is open, the tree is dirty, CPU > 60 % or disk < 2 GB; then manifest → Guardian → Scout → report under `nice`. | live |
| Lab (sandbox tests) / Builder (auto‑merge) / hardware inbox | **Deferred to v3** on purpose: a small model must not install packages into a running robot unsupervised. When added they sit behind `evolution.require_operator_approval: true` (approval by Telegram reply). | planned |

## Deploying safely
```bash
git commit -am "..."          # the commit IS the rollback point
deploy/deploy.sh              # restart → Guardian --wait → if unhealthy: git reset --hard HEAD~1 + restart
deploy/deploy.sh <good-sha>   # roll back to a specific commit on failure
```
Guardian alone: `venv/bin/python evolution/guardian.py [--wait] [--require-hailo] [--no-brain]`.

## Rules
1. Guardian never uses an LLM. Pass/fail is a script.
2. Nothing in `evolution/` edits code, config, `.env`, systemd or the network. Scout writes to
   the database and the report only.
3. Runs pause while the operator is working (open SSH session or dirty git tree).
4. Secrets are read from `.env` only, never logged.
