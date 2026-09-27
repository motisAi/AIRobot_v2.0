"""evolution.db — SQLite store shared by the evolution agents.

Tables
  candidates : upgrades Scout found (one row per name+new_version, deduped)
  runs       : one row per agent run (scout/guardian/report) with a summary
  llm_calls  : every free-LLM call the agents made (cost visibility)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "evolution" / "evolution.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY,
    found_date TEXT NOT NULL,
    source TEXT NOT NULL,           -- pypi | github | hailo_zoo | manual
    name TEXT NOT NULL,
    current_version TEXT,
    new_version TEXT,
    relevance_score REAL,           -- 1..10 from the LLM (NULL = not scored)
    relevance_reason TEXT,
    category TEXT,                  -- model | library | skill | driver | config
    status TEXT NOT NULL DEFAULT 'found',  -- found | reported | approved | rejected | integrated
    url TEXT,
    notes TEXT,
    UNIQUE(name, new_version)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    ok INTEGER,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    agent TEXT,
    purpose TEXT,
    backend TEXT,
    model TEXT,
    prompt_chars INTEGER,
    reply_chars INTEGER,
    seconds REAL,
    ok INTEGER
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class EvolutionDB:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # -- candidates -----------------------------------------------------
    def upsert_candidate(self, source: str, name: str, current: str | None, new: str | None,
                         category: str, url: str = "", notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO candidates(found_date, source, name, current_version, new_version, "
            "category, url, notes) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(name, new_version) DO UPDATE SET current_version=excluded.current_version, "
            "notes=excluded.notes RETURNING id",
            (_now(), source, name, current, new, category, url, notes))
        row = cur.fetchone()
        self.conn.commit()
        return int(row[0])

    def unscored(self):
        return self.conn.execute(
            "SELECT * FROM candidates WHERE relevance_score IS NULL OR relevance_score = 0 ORDER BY id").fetchall()

    def score(self, cid: int, score: float, reason: str) -> None:
        self.conn.execute("UPDATE candidates SET relevance_score=?, relevance_reason=? WHERE id=?",
                          (score, reason[:1000], cid))
        self.conn.commit()

    def candidates_since(self, since_date: str):
        return self.conn.execute(
            "SELECT * FROM candidates WHERE found_date >= ? ORDER BY relevance_score DESC, id",
            (since_date,)).fetchall()

    def mark(self, cid: int, status: str) -> None:
        self.conn.execute("UPDATE candidates SET status=? WHERE id=?", (status, cid))
        self.conn.commit()

    # -- runs -----------------------------------------------------------
    def start_run(self, agent: str) -> int:
        cur = self.conn.execute("INSERT INTO runs(agent, started) VALUES(?, ?) RETURNING id",
                                (agent, _now()))
        rid = int(cur.fetchone()[0])
        self.conn.commit()
        return rid

    def finish_run(self, rid: int, ok: bool, summary) -> None:
        if not isinstance(summary, str):
            summary = json.dumps(summary)
        self.conn.execute("UPDATE runs SET finished=?, ok=?, summary=? WHERE id=?",
                          (_now(), 1 if ok else 0, summary[:4000], rid))
        self.conn.commit()

    def last_runs(self, n: int = 10):
        return self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (n,)).fetchall()

    # -- llm calls ------------------------------------------------------
    def log_llm(self, agent: str, purpose: str, backend: str, model: str,
                prompt_chars: int, reply_chars: int, seconds: float, ok: bool) -> None:
        self.conn.execute(
            "INSERT INTO llm_calls(ts, agent, purpose, backend, model, prompt_chars, reply_chars, seconds, ok) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (_now(), agent, purpose, backend, model, prompt_chars, reply_chars, round(seconds, 2), 1 if ok else 0))
        self.conn.commit()

    def llm_usage_since(self, since_date: str):
        return self.conn.execute(
            "SELECT backend, COUNT(*) AS calls, SUM(prompt_chars) AS prompt_chars, "
            "SUM(reply_chars) AS reply_chars FROM llm_calls WHERE ts >= ? GROUP BY backend",
            (since_date,)).fetchall()

    def close(self) -> None:
        self.conn.close()
