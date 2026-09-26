"""Free web search for the robot (no API key required).

Gives Gonzo the ability to answer questions from the live web, like Claude or
Copilot. Uses DuckDuckGo — no account, no key, no cost.

Backends, in order of preference:
1. ``ddgs`` / ``duckduckgo_search`` package if installed (best quality).
2. DuckDuckGo's Instant-Answer JSON API (``api.duckduckgo.com``) via urllib.
3. DuckDuckGo Lite HTML endpoint scraped with a tiny regex parser.

All backends return a list of dicts: ``{"title", "snippet", "url"}``.
"""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("WebSearch")

# Optional high-quality backend
try:  # the maintained package is 'ddgs' (formerly 'duckduckgo_search')
    from ddgs import DDGS  # type: ignore
    _DDGS_AVAILABLE = True
except Exception:
    try:
        from duckduckgo_search import DDGS  # type: ignore
        _DDGS_AVAILABLE = True
    except Exception:
        DDGS = None
        _DDGS_AVAILABLE = False

_USER_AGENT = "Mozilla/5.0 (X11; Linux aarch64) AIRobot/2.0"


def _via_ddgs(query: str, max_results: int, region: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region=region, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", r.get("url", "")),
            })
    return results


def _via_instant_answer(query: str, timeout: float) -> List[Dict[str, str]]:
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": 1, "skip_disambig": 1,
    })
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    results: List[Dict[str, str]] = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", query),
            "snippet": data.get("AbstractText", ""),
            "url": data.get("AbstractURL", ""),
        })
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:80],
                "snippet": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })
    return results


_LITE_ROW = re.compile(
    r'<a[^>]*class="result-link"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result-snippet">(?P<snippet>.*?)</td>',
    re.DOTALL | re.IGNORECASE,
)


def _clean(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _via_lite_html(query: str, max_results: int, timeout: float) -> List[Dict[str, str]]:
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    results: List[Dict[str, str]] = []
    for m in _LITE_ROW.finditer(body):
        results.append({
            "title": _clean(m.group("title")),
            "snippet": _clean(m.group("snippet")),
            "url": _clean(m.group("url")),
        })
        if len(results) >= max_results:
            break
    return results


def search_web(query: str, max_results: int = 4, region: str = "wt-wt",
               timeout: float = 8.0) -> List[Dict[str, str]]:
    """Return up to ``max_results`` web results for ``query``. Never raises."""
    query = (query or "").strip()
    if not query:
        return []

    if _DDGS_AVAILABLE:
        try:
            res = _via_ddgs(query, max_results, region)
            if res:
                return res[:max_results]
        except Exception as exc:
            logger.info("ddgs backend failed (%s); trying fallbacks", exc)

    try:
        res = _via_instant_answer(query, timeout)
        if res:
            return res[:max_results]
    except Exception as exc:
        logger.info("instant-answer backend failed: %s", exc)

    try:
        res = _via_lite_html(query, max_results, timeout)
        if res:
            return res[:max_results]
    except Exception as exc:
        logger.warning("all web-search backends failed: %s", exc)

    return []


def get_weather(query: str, timeout: float = 6.0) -> str:
    """Live current weather for a city via wttr.in (free, no key). Returns a
    one-line summary, or '' on failure. Text-search engines return article
    links, not live numbers — this gives the actual current conditions."""
    t = (query or "").lower()
    m = re.search(r"(?:weather|forecast|temperature|how (?:hot|cold))"
                  r"[^a-z]*(?:in|at|for|of)?\s+([a-z .'-]+)", t)
    city = ""
    if m:
        city = m.group(1).strip(" .?!")
        for tail in (" today", " now", " right now", " currently", " tomorrow"):
            if city.endswith(tail):
                city = city[: -len(tail)].strip()
    fmt = urllib.parse.quote("%l: %C, %t (feels %f), humidity %h, wind %w")
    loc = urllib.parse.quote(city)
    url = f"https://wttr.in/{loc}?format={fmt}&m"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = resp.read().decode("utf-8", errors="ignore").strip()
        if out and "Unknown location" not in out and "ERROR" not in out.upper():
            return out
    except Exception as exc:
        logger.info("weather lookup failed: %s", exc)
    return ""


def format_results(results: List[Dict[str, str]], limit: int = 4) -> str:
    """Compact, LLM-friendly rendering of search results."""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results[:limit], 1):
        snippet = (r.get("snippet") or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        title = (r.get("title") or "").strip()
        lines.append(f"[{i}] {title}: {snippet}")
    return "\n".join(lines)


def available() -> str:
    """Which backend is active (for diagnostics)."""
    return "ddgs" if _DDGS_AVAILABLE else "urllib(duckduckgo)"
