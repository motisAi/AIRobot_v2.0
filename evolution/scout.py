#!/usr/bin/env python3
"""Scout — looks for upgrades relevant to Stella and REPORTS them.  Installs nothing.

Sources (all free, unauthenticated, rate-limit friendly):
  * PyPI  — newer versions of the AI/ML packages Stella uses (`pip index versions`)
  * GitHub releases — Piper, Vosk, openWakeWord, faster-whisper, Ultralytics, hailo-ollama
  * Hailo model zoo — latest tag
Each new candidate is scored 1-10 for relevance by the free LLM against the manifest.

    venv/bin/python evolution/scout.py            # full run
    venv/bin/python evolution/scout.py --no-llm   # collect only, skip scoring
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python evolution/x.py` from anywhere
from evolution.db import EvolutionDB  # noqa: E402
PY = ROOT / "venv" / "bin" / "python"
MANIFEST = ROOT / "stella_manifest.yaml"

PYPI_WATCH = ["vosk", "openai", "onnxruntime", "opencv-python-headless", "mediapipe",
              "yt-dlp", "face-recognition", "tinytuya", "paho-mqtt", "aiohttp", "flask",
              "faster-whisper", "openwakeword", "ultralytics"]
GITHUB_WATCH = {                       # name: (repo, category)
    "piper": ("rhasspy/piper", "model"),
    "vosk-api": ("alphacep/vosk-api", "library"),
    "openWakeWord": ("dscripka/openWakeWord", "model"),
    "faster-whisper": ("SYSTRAN/faster-whisper", "model"),
    "ultralytics": ("ultralytics/ultralytics", "model"),
    "hailo_model_zoo": ("hailo-ai/hailo_model_zoo", "model"),
    "hailo-rpi5-examples": ("hailo-ai/hailo-rpi5-examples", "skill"),
}
UA = {"User-Agent": "stella-scout/1.0 (Raspberry Pi robot; report-only)"}


def _get(url: str, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def pypi_candidates(db: EvolutionDB, installed: dict) -> int:
    n = 0
    for pkg in PYPI_WATCH:
        try:
            data = _get(f"https://pypi.org/pypi/{pkg}/json")
        except Exception:
            continue
        latest = data.get("info", {}).get("version")
        cur = installed.get(pkg, "not installed")
        if not latest or latest == cur:
            continue
        # numpy<2 and hailort are pinned on purpose; skip noise
        if pkg in ("numpy", "hailort"):
            continue
        db.upsert_candidate("pypi", pkg, cur, latest, "library",
                            url=f"https://pypi.org/project/{pkg}/{latest}/",
                            notes=(data["info"].get("summary") or "")[:200])
        n += 1
    return n


def github_candidates(db: EvolutionDB) -> int:
    n = 0
    for name, (repo, cat) in GITHUB_WATCH.items():
        try:
            rel = _get(f"https://api.github.com/repos/{repo}/releases/latest")
        except Exception:
            try:
                tags = _get(f"https://api.github.com/repos/{repo}/tags?per_page=1")
                rel = {"tag_name": tags[0]["name"], "html_url": f"https://github.com/{repo}",
                       "body": ""} if tags else None
            except Exception:
                rel = None
        if not rel:
            continue
        tag = rel.get("tag_name")
        db.upsert_candidate("github", name, None, tag, cat, url=rel.get("html_url", ""),
                            notes=(rel.get("body") or "")[:300].replace("\r", ""))
        n += 1
        time.sleep(1)   # 60 req/h unauthenticated — stay polite
    return n


def score_candidates(db: EvolutionDB, manifest_text: str, limit: int = 12) -> int:
    from evolution.llm_client import StellaLLM
    # Groq/Gemini first for scoring: the 1.5B NPU model is slow (~40 s) and scores
    # inconsistently; hailo stays as the offline fallback.
    llm = StellaLLM(agent="scout", db=db, prefer_local=False)
    if not llm.available:
        return 0
    n = 0
    for row in db.unscored()[:limit]:
        prompt = (
            "Stella is a Raspberry Pi 5 + Hailo-10H robot. Her manifest (YAML):\n"
            f"{manifest_text[:2200]}\n\n"
            f"Candidate upgrade: {row['name']} {row['current_version'] or ''} -> {row['new_version']} "
            f"({row['source']}, {row['category']}). Notes: {row['notes'] or '-'}\n"
            "Score how relevant/valuable this is for Stella from 1 (irrelevant) to 10 (must have), "
            "considering her known weaknesses. JSON with an integer score 1-10 and a one-sentence reason, e.g. {\"score\": 7, \"reason\": \"...\"}"
        )
        try:
            data = llm.ask_json(prompt, purpose=f"score:{row['name']}", max_tokens=120)
            score = float(data.get("score", 0))
            db.score(row["id"], max(1.0, min(10.0, score)), str(data.get("reason", ""))[:300])
            n += 1
        except Exception as e:  # noqa: BLE001
            db.score(row["id"], 0.0, f"scoring failed: {e}"[:300])
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    db = EvolutionDB()
    rid = db.start_run("scout")
    try:
        manifest = yaml.safe_load(MANIFEST.read_text()) if MANIFEST.exists() else {}
        installed = manifest.get("python_packages", {}) if isinstance(manifest, dict) else {}
        n_pypi = pypi_candidates(db, installed)
        n_gh = github_candidates(db)
        n_scored = 0 if a.no_llm else score_candidates(db, MANIFEST.read_text() if MANIFEST.exists() else "")
        summary = {"pypi": n_pypi, "github": n_gh, "scored": n_scored}
        db.finish_run(rid, True, summary)
        print("scout:", summary)
        return 0
    except Exception as e:  # noqa: BLE001
        db.finish_run(rid, False, f"{type(e).__name__}: {e}")
        print("scout failed:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
