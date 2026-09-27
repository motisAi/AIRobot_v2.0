#!/usr/bin/env python3
"""Morning report — what happened to Stella and what Scout found.

Writes evolution/reports/YYYY-MM-DD.md and sends a short Telegram summary.

    venv/bin/python evolution/report.py             # write + send
    venv/bin/python evolution/report.py --no-send   # write only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python evolution/x.py` from anywhere
from evolution.db import EvolutionDB  # noqa: E402
REPORTS = ROOT / "evolution" / "reports"
GUARDIAN = REPORTS / "guardian-latest.json"


def _env(key: str) -> str:
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def _run(cmd, timeout=20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def telegram(text: str) -> bool:
    token, chat = _env("TELEGRAM_TOKEN"), _env("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=15)
        return True
    except Exception:
        return False


def log_stats(hours: int = 24) -> dict:
    out = _run(["journalctl", "-u", "airobot", "--since", f"{hours} hours ago", "--no-pager", "-o", "cat"], 60)
    lines = out.splitlines()
    errors = [l for l in lines if " ERROR " in l]
    restarts = _run(["journalctl", "-u", "airobot", "--since", f"{hours} hours ago", "--no-pager", "-o", "cat",
                     "-g", "Started airobot|Starting airobot|Started Stella"], 30).count("\n")
    conv = sum(1 for l in lines if "Wake word detected" in l)
    top = {}
    for l in errors:
        key = l.split(" - ERROR - ")[-1][:70] if " - ERROR - " in l else l[-70:]
        top[key] = top.get(key, 0) + 1
    top_errors = sorted(top.items(), key=lambda kv: -kv[1])[:5]
    return {"lines": len(lines), "errors": len(errors), "restarts": restarts,
            "conversations": conv, "top_errors": top_errors}


def open_recurring_bugs() -> list[str]:
    out = []
    for f in sorted((ROOT / "bug_report").glob("bug_*.md")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        if "**Status:** recurring" in txt:
            out.append(txt.splitlines()[0].lstrip("# ").strip())
    return out


def build(db: EvolutionDB, day: str) -> tuple[str, str]:
    g = json.loads(GUARDIAN.read_text()) if GUARDIAN.exists() else {}
    g_ok = g.get("healthy")
    g_fail = [c["name"] + ": " + c["detail"] for c in g.get("checks", []) if not c["ok"]]
    res = next((c["detail"] for c in g.get("checks", []) if c["name"] == "resources"), "")
    stats = log_stats()
    since = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    cands = [c for c in db.candidates_since(since) if (c["relevance_score"] or 0) >= 6]
    usage = db.llm_usage_since(since)
    bugs = open_recurring_bugs()

    md = [f"# Stella Evolution Report — {day}", ""]
    md += ["## Health (Guardian)",
           f"- Status: **{'HEALTHY' if g_ok else 'UNHEALTHY' if g_ok is not None else 'not run'}** ({g.get('started', '-')})"]
    md += [f"- {f}" for f in g_fail] or ["- all checks passed"]
    md += [f"- Resources: {res}", ""]
    md += ["## Last 24 h in the log",
           f"- {stats['lines']} log lines, **{stats['errors']} errors**, {stats['restarts']} service starts, "
           f"~{stats['conversations']} conversations"]
    md += [f"  - {n}× {k}" for k, n in stats["top_errors"]]
    md += ["", "## Scout — upgrade ideas (report only, nothing installed)"]
    if cands:
        for c in cands[:10]:
            md.append(f"- **{c['name']}** {c['current_version'] or ''} → {c['new_version']}  "
                      f"[{c['relevance_score']:.0f}/10] {c['relevance_reason'] or ''}  {c['url'] or ''}")
    else:
        md.append("- nothing new worth reporting since yesterday")
    md += ["", "## Open recurring bugs"] + ([f"- {b}" for b in bugs] or ["- none"])
    md += ["", "## Evolution LLM usage (free tiers)"]
    md += [f"- {u['backend']}: {u['calls']} calls, {u['prompt_chars'] or 0} prompt chars, {u['reply_chars'] or 0} reply chars"
           for u in usage] or ["- none"]
    md += ["", f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by evolution/report.py_"]

    top = cands[0] if cands else None
    tg = (f"🌅 Stella morning report {day}\n"
          f"Health: {'✅ healthy' if g_ok else '❌ UNHEALTHY' if g_ok is not None else '—'}"
          + (f" ({'; '.join(g_fail)[:120]})" if g_fail else "") + "\n"
          f"24h: {stats['errors']} errors, {stats['restarts']} starts, ~{stats['conversations']} chats\n"
          f"Power/temp: {res.split('|')[-1].strip()[:80]}\n"
          f"Scout: {len(cands)} idea(s)" + (f" — top: {top['name']} → {top['new_version']} [{top['relevance_score']:.0f}/10]" if top else "") + "\n"
          f"Recurring bugs open: {len(bugs)}")
    return "\n".join(md), tg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-send", action="store_true")
    a = ap.parse_args()
    db = EvolutionDB()
    rid = db.start_run("report")
    day = time.strftime("%Y-%m-%d")
    md, tg = build(db, day)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{day}.md"
    out.write_text(md, encoding="utf-8")
    sent = False if a.no_send else telegram(tg)
    db.finish_run(rid, True, {"report": str(out.relative_to(ROOT)), "telegram": sent})
    print(f"wrote {out.relative_to(ROOT)}; telegram={'sent' if sent else 'not sent'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
