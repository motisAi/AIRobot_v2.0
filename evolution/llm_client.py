"""Free-only LLM client for the evolution agents.

Order: hailo-ollama (local NPU, free, offline) -> Groq -> Gemini.  Never OpenAI or
Anthropic (paid; and Claude is not authorized for this project).  Every call is
logged to evolution.db so usage stays visible.

Usage:
    llm = StellaLLM(agent="scout")
    text = llm.ask("...", purpose="score-candidate")
    data = llm.ask_json("... reply with JSON {\"score\": 1-10, \"reason\": \"...\"}", purpose=...)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

from evolution.db import EvolutionDB

ROOT = Path(__file__).resolve().parent.parent

HAILO_URL = "http://127.0.0.1:8000/v1/chat/completions"
HAILO_MODEL = "qwen2.5-instruct:1.5b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"          # the small/fast one: keep the big model's quota for Stella's voice
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = "gemini-3.6-flash"


def _env(key: str) -> str:
    """Read a key from the environment or the repo .env (mode 600, never logged)."""
    if os.environ.get(key):
        return os.environ[key]
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) stella-evolution/1.0",  # Groq/Cloudflare 403s the urllib default UA
                                          **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class StellaLLM:
    def __init__(self, agent: str = "evolution", db: EvolutionDB | None = None,
                 prefer_local: bool = True):
        self.agent = agent
        self.db = db or EvolutionDB()
        self.backends = []
        if prefer_local and os.path.exists("/dev/hailo0"):
            self.backends.append(("hailo", HAILO_URL, HAILO_MODEL, {}, 120))
        if _env("GROQ_API_KEY"):
            self.backends.append(("groq", GROQ_URL, GROQ_MODEL,
                                  {"Authorization": f"Bearer {_env('GROQ_API_KEY')}"}, 40))
        if _env("GEMINI_API_KEY"):
            self.backends.append(("gemini", GEMINI_URL, GEMINI_MODEL,
                                  {"Authorization": f"Bearer {_env('GEMINI_API_KEY')}"}, 40))
        if not prefer_local and os.path.exists("/dev/hailo0"):
            self.backends.append(("hailo", HAILO_URL, HAILO_MODEL, {}, 120))

    @property
    def available(self) -> bool:
        return bool(self.backends)

    def ask(self, prompt: str, purpose: str = "", max_tokens: int = 400,
            system: str = "You are Stella's engineering assistant. Be brief and factual.") -> str:
        last_err = "no backend"
        for name, url, model, headers, timeout in self.backends:
            # gpt-oss / gemini-flash spend tokens on hidden reasoning first: never cap below 1024
            body = {"model": model, "max_tokens": max(int(max_tokens), 2048), "temperature": 0.2,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}]}
            if name in ("groq", "gemini"):
                body["reasoning_effort"] = "low"   # keep hidden reasoning short; answer is what we want
            t = time.monotonic()
            try:
                data = _post_json(url, body, headers, timeout)
                text = (data["choices"][0]["message"].get("content") or "").strip()
                if not text:
                    raise ValueError("empty reply")
                self.db.log_llm(self.agent, purpose, name, model, len(prompt), len(text),
                                time.monotonic() - t, True)
                return text
            except Exception as e:  # noqa: BLE001 — try the next backend
                last_err = f"{name}: {type(e).__name__}: {e}"
                self.db.log_llm(self.agent, purpose, name, model, len(prompt), 0,
                                time.monotonic() - t, False)
        raise RuntimeError(f"all LLM backends failed ({last_err})")

    def ask_json(self, prompt: str, purpose: str = "", max_tokens: int = 400) -> dict:
        text = self.ask(prompt + "\nReply with ONLY a JSON object, no prose, no code fences.",
                        purpose, max_tokens)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # tolerant fallback: pull the fields out of slightly broken JSON
        score = re.search(r'"score"\s*:\s*"?(\d+(?:\.\d+)?)', text)
        reason = re.search(r'"reason"\s*:\s*"([^"]*)', text)
        if score:
            return {"score": float(score.group(1)), "reason": reason.group(1) if reason else ""}
        raise ValueError(f"no JSON in reply: {text[:120]!r}")
