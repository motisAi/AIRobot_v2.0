"""AI Engine — online and offline natural-language intelligence.

Provides two interchangeable backends:

1. **Online** — Calls OpenAI-compatible or Anthropic chat APIs.
   Requires ``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY`` in ``.env``.
2. **Offline** — Runs a quantized GGUF model locally via
   ``llama-cpp-python``.  No internet needed.  A 3-4 GB Q4 model fits
   comfortably in the 8 GB RAM of a Raspberry Pi 5.

The engine exposes a single ``think(prompt, context)`` method that the
robot brain calls for intent analysis, conversation, and reasoning.
It automatically falls back from online to offline when the network is
unavailable.

Typical usage::

    engine = AIEngine()
    engine.start()
    reply = engine.think("What do you see?", context={"objects": ["cup", "laptop"]})
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import system_config, behavior_config

# Optional online providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

# Optional offline provider
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_AVAILABLE = False

import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# System prompt shared across backends
SYSTEM_PROMPT = (
    "You are {name}, a friendly home robot. You speak OUT LOUD, so keep replies "
    "SHORT and natural — usually 1 to 2 sentences, no lists or headings. "
    "You can see through a camera, recognise faces you've met, listen, and talk. "
    "You DO remember people and past conversations: the current user's name, "
    "relevant history, and their preferences are provided to you inside "
    "[Context: ...]. Use that memory naturally and confidently — for example, "
    "recall what a person told you earlier or what they like. "
    "But do NOT invent specifics you were not given: no made-up numbers, counts, "
    "addresses, or facts about a database, GPS, or smart-home you don't have. If "
    "the provided context doesn't contain the answer, say you don't recall it. "
    "Always use the [Context] (the user, memory, visible objects, time, weather, "
    "web results) when answering. Only obey physical or privileged commands from "
    "your authenticated master. "
    "You identify people ONLY by their face through your camera. The current "
    "user's identity is given to you in [Context] when a face is recognised. If "
    "the context does NOT name a user, you do NOT currently recognise who is "
    "speaking — say so honestly and ask; never claim to recognise someone by "
    "their voice or by past chat history. "
    "TOOLS: only call a tool when the user's LATEST message clearly and fully "
    "asks for that specific action. A bare acknowledgement or filler ('yes', "
    "'ok', 'okay', 'good', 'sure', 'thanks', 'go on', 'can you') is NOT a "
    "request — just reply in words, or ask what they'd like; do not call a tool. "
    "Never repeat an action you already performed earlier in this same "
    "conversation (e.g. setting the same reminder or sending another photo) "
    "unless the user clearly asks for it again. If a request is missing a "
    "detail you need (like the reminder text or time), ask for it instead of "
    "guessing from earlier context."
)


@dataclass
class AIConfig:
    """Runtime configuration for the AI engine.

    Non-secret settings come from ``config/config.yaml`` (the ``ai`` and
    ``web_search`` sections). Secrets (API keys) always come from ``.env``.
    Build with :meth:`from_settings` to honour the YAML config.
    """

    # Engine mode understood by start(): "online" | "offline" | "auto".
    # Mapped from the friendly YAML mode (local/online/hybrid) in from_settings().
    mode: str = "offline"
    user_mode: str = "local"          # the YAML value: local | online | hybrid
    allow_cloud_escalation: bool = False

    # Online provider
    online_provider: str = "groq"     # groq | openai | anthropic
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = "claude-sonnet-4-20250514"
    # Groq — free, fast, OpenAI-compatible (uses the openai client + base_url)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"  # rate-limit cushion (same key)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Gemini — free, OpenAI-compatible endpoint (also used for vision later)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Fallback chain: try these in order; fall through on rate-limit/error to the
    # next; the local Hailo NPU is the always-available final fallback.
    fallback_order: list = field(default_factory=lambda: ["groq", "gemini", "hailo"])
    # Agent mode: let the LLM decide which tools to call (web, weather, camera,
    # devices, reminders) and chain them. Needs a tool-capable cloud provider.
    agent_enabled: bool = True
    max_tokens: int = 400
    temperature: float = 0.7
    max_history: int = 20

    # Offline model (llama.cpp GGUF) — optional CPU fallback
    offline_model_path: str = os.getenv("OFFLINE_MODEL_PATH", "")
    offline_context_length: int = 4096
    offline_threads: int = 4
    offline_gpu_layers: int = 0

    # Hailo-10H NPU via hailo-ollama REST API
    hailo_ollama_url: str = "http://localhost:8000"
    hailo_ollama_model: str = "llama3.2:3b"

    # Web search (see modules/ai/web_search.py)
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    web_search_max_results: int = 4
    web_search_region: str = "wt-wt"
    web_search_timeout: float = 8.0
    web_search_auto: bool = True

    @classmethod
    def from_settings(cls) -> "AIConfig":
        """Build an AIConfig from the YAML-backed RobotConfig sections."""
        try:
            from config.settings import ai_config as a, web_search_config as w
        except Exception:
            return cls()

        mode_map = {"local": "offline", "online": "online", "hybrid": "offline"}
        return cls(
            mode=mode_map.get(str(a.mode).lower(), "offline"),
            user_mode=str(a.mode).lower(),
            allow_cloud_escalation=bool(a.allow_cloud_escalation),
            online_provider=a.online_provider,
            openai_model=a.openai_model,
            openai_base_url=a.openai_base_url,
            anthropic_model=a.anthropic_model,
            groq_model=getattr(a, 'groq_model', 'llama-3.3-70b-versatile'),
            groq_fast_model=getattr(a, 'groq_fast_model', 'llama-3.1-8b-instant'),
            max_tokens=a.max_tokens,
            temperature=a.temperature,
            max_history=a.conversation_memory_turns,
            offline_model_path=a.offline_model_path or "",
            hailo_ollama_url=a.hailo_ollama_url,
            hailo_ollama_model=a.hailo_ollama_model,
            gemini_model=getattr(a, 'gemini_model', 'gemini-2.0-flash'),
            fallback_order=list(getattr(a, 'fallback_order',
                                        ["groq", "gemini", "hailo"])),
            agent_enabled=bool(getattr(a, 'agent_enabled', True)),
            web_search_enabled=bool(w.enabled and w.provider != "none"),
            web_search_provider=w.provider,
            web_search_max_results=w.max_results,
            web_search_region=w.region,
            web_search_timeout=w.timeout,
            web_search_auto=bool(w.allow_auto_search),
        )


class AIEngine:
    """Unified interface for online and offline LLM inference."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cfg = config or AIConfig.from_settings()

        self._backend: Optional[str] = None  # "openai", "anthropic", "llama", None
        self._llama: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None
        self._lock = threading.Lock()

        # Optional cloud client kept ready for hybrid escalation.
        self._escalation_client: Optional[Any] = None
        self._escalation_provider: Optional[str] = None

        # Agent tools (registered by the app): OpenAI-style schema + dispatch fn.
        self._agent_tools: Optional[list] = None
        self._agent_dispatch: Optional[Any] = None

        self._conversation_history: List[Dict[str, str]] = []
        self._max_history = self.cfg.max_history  # Keep last N turns
        # provider name -> epoch until which to skip it (after a 429 rate-limit)
        self._cooldown: Dict[str, float] = {}

        # Reference to learning DB — set by main.py for memory recall
        self.learning_db = None
        # Current user ID — set by brain when a user is identified
        self._current_user: Optional[str] = None

        self.system_prompt = SYSTEM_PROMPT.format(name=behavior_config.robot_name)
        # Single language switch (behavior.language): reply in Hebrew when set.
        if str(getattr(behavior_config, 'language', 'en')).lower().startswith('he'):
            self.system_prompt += (" IMPORTANT: always reply in Hebrew (עברית), "
                                   "regardless of the language of the question.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Build the provider fallback chain (e.g. Groq -> Gemini -> Hailo).

        Each provider is tried in order per question; on rate-limit/error we
        fall through to the next. The local Hailo NPU is the always-available
        final fallback, so Stella never goes dead and cost stays zero.
        """
        # name -> (kind, client_or_None, model).  kind: openai | anthropic | hailo
        self._clients: Dict[str, Any] = {}

        if OPENAI_AVAILABLE and self.cfg.groq_api_key:
            try:
                gclient = openai.OpenAI(
                    api_key=self.cfg.groq_api_key, base_url=self.cfg.groq_base_url)
                self._clients["groq"] = ("openai", gclient, self.cfg.groq_model)
                # Same key, smaller/faster model with much higher rate limits —
                # used as a cushion when the big model is throttled (HTTP 429).
                if self.cfg.groq_fast_model:
                    self._clients["groq_fast"] = ("openai", gclient,
                                                  self.cfg.groq_fast_model)
            except Exception as exc:
                self.logger.info("Groq client init failed: %s", exc)

        if OPENAI_AVAILABLE and self.cfg.gemini_api_key:
            try:
                self._clients["gemini"] = ("openai", openai.OpenAI(
                    api_key=self.cfg.gemini_api_key, base_url=self.cfg.gemini_base_url),
                    self.cfg.gemini_model)
            except Exception as exc:
                self.logger.info("Gemini client init failed: %s", exc)

        if OPENAI_AVAILABLE and self.cfg.openai_api_key:
            try:
                kwargs = {"api_key": self.cfg.openai_api_key}
                if self.cfg.openai_base_url:
                    kwargs["base_url"] = self.cfg.openai_base_url
                self._clients["openai"] = ("openai", openai.OpenAI(**kwargs),
                                           self.cfg.openai_model)
            except Exception as exc:
                self.logger.info("OpenAI client init failed: %s", exc)

        if ANTHROPIC_AVAILABLE and self.cfg.anthropic_api_key:
            try:
                self._clients["anthropic"] = ("anthropic", anthropic.Anthropic(
                    api_key=self.cfg.anthropic_api_key), self.cfg.anthropic_model)
            except Exception as exc:
                self.logger.info("Anthropic client init failed: %s", exc)

        # Local Hailo-10H NPU (always the safety net when reachable).
        if self._init_hailo_ollama():
            self._clients["hailo"] = ("hailo", None, self.cfg.hailo_ollama_model)

        # Build the ordered chain from config, keeping only available providers.
        chain = [n for n in self.cfg.fallback_order if n in self._clients]
        # In local mode, prefer the on-device NPU first.
        if self.cfg.user_mode == "local" and "hailo" in self._clients:
            chain = ["hailo"] + [n for n in chain if n != "hailo"]
        # Slot the faster same-key Groq model right after groq, so a rate-limit
        # on the big model degrades gracefully before the flaky offline path.
        if "groq" in chain and "groq_fast" in self._clients and "groq_fast" not in chain:
            chain.insert(chain.index("groq") + 1, "groq_fast")
        # Make sure hailo is the last resort if present and not already included.
        for n in self._clients:
            if n not in chain:
                chain.append(n)
        self._chain = chain
        self._backend = chain[0] if chain else "rules"

        if chain:
            self.logger.info("AI provider chain: %s",
                             " -> ".join(f"{n}({self._clients[n][2]})" for n in chain))
        else:
            self.logger.warning("No LLM backend available — rule-based replies only")
        return True

    def warmup(self) -> None:
        """Pre-load the NPU model so the first real question isn't slow.

        Switching hailo-ollama models (e.g. to llama3.2:3b) loads a new HEF into
        the accelerator, which can take many seconds. Doing a tiny request now
        absorbs that cost up front. Safe to call in a background thread.
        """
        if "hailo" not in getattr(self, "_clients", {}):
            return
        try:
            payload = {
                "model": self.cfg.hailo_ollama_model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
            req = urllib.request.Request(
                f"{self.cfg.hailo_ollama_url}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            self.logger.info("Warming up NPU model %s ...", self.cfg.hailo_ollama_model)
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read()
            self.logger.info("NPU model warm and ready")
        except Exception as exc:
            self.logger.info("NPU warmup skipped: %s", exc)

    def _setup_escalation(self) -> None:
        """In hybrid mode, keep a cloud client ready to escalate hard questions."""
        if self.cfg.user_mode != "hybrid" or not self.cfg.allow_cloud_escalation:
            return
        try:
            if (self.cfg.online_provider == "anthropic" and ANTHROPIC_AVAILABLE
                    and self.cfg.anthropic_api_key):
                self._escalation_client = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)
                self._escalation_provider = "anthropic"
            elif OPENAI_AVAILABLE and self.cfg.openai_api_key:
                kwargs = {"api_key": self.cfg.openai_api_key}
                if self.cfg.openai_base_url:
                    kwargs["base_url"] = self.cfg.openai_base_url
                self._escalation_client = openai.OpenAI(**kwargs)
                self._escalation_provider = "openai"
            if self._escalation_client:
                self.logger.info("Hybrid escalation ready via %s", self._escalation_provider)
        except Exception as exc:
            self.logger.info("Hybrid escalation unavailable: %s", exc)

    def stop(self) -> None:
        with self._lock:
            self._llama = None
            self._openai_client = None
            self._anthropic_client = None
            self._backend = None
        self.logger.info("AI engine stopped")

    @property
    def backend_name(self) -> str:
        return self._backend or "none"

    @property
    def is_online(self) -> bool:
        return self._backend in ("openai", "anthropic")

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def think(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process user input and return a natural-language response.

        Args:
            user_input: Transcribed speech or text command.
            context: Optional dict with sensor data, visible objects, etc.

        Returns:
            The robot's response text.
        """
        # Build context string
        ctx_str = ""
        if context:
            parts = []
            if context.get("user"):
                parts.append(f"You are speaking with {context['user']} (recognised by face)")
            if "objects" in context:
                parts.append(f"Visible objects: {', '.join(str(o) for o in context['objects'])}")
            if "faces" in context:
                parts.append(f"People visible: {', '.join(str(f) for f in context['faces'])}")
            if "location" in context:
                parts.append(f"Location: {context['location']}")
            if "time" in context:
                parts.append(f"Time: {context['time']}")
            if "battery" in context:
                parts.append(f"Battery: {context['battery']}%")
            ctx_str = " | ".join(parts)

        # Inject recalled memories and user profile from DB
        memory_str = self._recall_context(user_input)
        if memory_str:
            ctx_str = f"{ctx_str} | {memory_str}" if ctx_str else memory_str

        # When the agent is active it fetches web/weather itself via tools, so
        # skip the automatic pre-injection to avoid doing it twice.
        agent_active = bool(getattr(self.cfg, "agent_enabled", True)
                            and self._agent_tools
                            and self._first_openai_client()[1] is not None)
        web_str = "" if agent_active else self._maybe_web_search(user_input)
        if web_str:
            ctx_str = f"{ctx_str} | {web_str}" if ctx_str else web_str

        full_input = f"[Context: {ctx_str}] {user_input}" if ctx_str else user_input

        # Add to history
        self._conversation_history.append({"role": "user", "content": full_input})
        self._trim_history()

        # Build the message list once, then try each provider in the chain
        # until one answers. Fall through on rate-limit / error to the next.
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._conversation_history[-self._max_history:])

        reply = None
        # Agent first: let the LLM call tools (web/weather/camera/devices/reminders).
        if agent_active:
            try:
                reply = self._agent_loop(messages)
            except Exception as exc:
                self.logger.warning("agent loop error: %s", exc)
                reply = None

        chain = getattr(self, "_chain", [])
        if not reply:
            for i, name in enumerate(chain):
                if self._cooled(name):
                    continue  # recently rate-limited — skip for now
                try:
                    reply = self._query_backend(name, messages)
                    if reply and reply.strip():
                        if i > 0:
                            self.logger.info("Answered via fallback provider: %s", name)
                        break
                    reply = None
                except Exception as exc:
                    self.logger.warning("Provider '%s' failed (%s) — trying next",
                                        name, str(exc)[:140])
                    if self._is_rate_limit(exc):
                        self._cool(name)
                    reply = None

        if not reply:
            reply = self._rule_based_response(user_input)

        self._conversation_history.append({"role": "assistant", "content": reply})

        # Automatically extract and save preferences from the conversation
        self._extract_preferences(user_input, reply)

        return reply

    def _query_backend(self, name: str, messages) -> str:
        """Send the message list to one provider (by chain name)."""
        kind, client, model = self._clients[name]
        if kind == "openai":   # groq / gemini / openai are all OpenAI-compatible
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature,
                timeout=30)
            return (resp.choices[0].message.content or "").strip()
        if kind == "anthropic":
            msgs = [m for m in messages if m.get("role") != "system"]
            resp = client.messages.create(
                model=model, system=self.system_prompt, messages=msgs,
                max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature)
            return (resp.content[0].text or "").strip()
        if kind == "hailo":
            return self._query_hailo_messages(messages)
        return ""

    def _query_hailo_messages(self, messages) -> str:
        """Chat with the local Hailo-10H NPU (hailo-ollama). Raises on failure."""
        payload = {"model": self.cfg.hailo_ollama_model, "messages": messages,
                   "stream": False}
        req = urllib.request.Request(
            f"{self.cfg.hailo_ollama_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
        parts = []
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            content = json.loads(line).get("message", {}).get("content", "")
            if content:
                parts.append(content)
        return "".join(parts).strip()

    # ------------------------------------------------------------------
    # Agent (tool-calling) support
    # ------------------------------------------------------------------
    def register_tools(self, tools: list, dispatch) -> None:
        """Register OpenAI-style tool schemas + a dispatch(name, args)->str fn."""
        self._agent_tools = tools
        self._agent_dispatch = dispatch
        self.logger.info("Agent tools registered: %d", len(tools or []))

    def _first_openai_client(self):
        """First tool-capable (OpenAI-compatible) provider in the chain."""
        for name in getattr(self, "_chain", []):
            kind, client, model = self._clients.get(name, (None, None, None))
            if kind == "openai" and client is not None:
                return name, client, model
        return None, None, None

    @staticmethod
    def _is_rate_limit(exc) -> bool:
        s = str(exc).lower()
        return "429" in s or "rate limit" in s or "rate_limit" in s

    def _cool(self, name: str, seconds: float = 60.0) -> None:
        self._cooldown[name] = time.time() + seconds
        self.logger.info("provider '%s' rate-limited — skipping it for %ds", name, int(seconds))

    def _cooled(self, name: str) -> bool:
        return self._cooldown.get(name, 0) > time.time()

    def _openai_clients(self):
        """Tool-capable (OpenAI-compatible) providers in chain order, skipping
        any currently on a rate-limit cooldown."""
        out = []
        for name in getattr(self, "_chain", []):
            if self._cooled(name):
                continue
            kind, client, model = self._clients.get(name, (None, None, None))
            if kind == "openai" and client is not None:
                out.append((name, client, model))
        return out

    def _agent_loop(self, messages, max_iters: int = 4) -> Optional[str]:
        """Let the LLM call registered tools, feeding results back until it
        produces a final answer. Returns text, or None to fall back to plain chat.

        If a provider is rate-limited/errors mid-loop, we retry the SAME turn on
        the next tool-capable provider (e.g. Groq 70B -> Groq 8B) so tools keep
        working instead of collapsing to a tool-less plain reply.
        """
        clients = self._openai_clients()
        if not clients or not self._agent_tools:
            return None
        ci = 0
        name, client, model = clients[ci]
        msgs = list(messages)
        for _ in range(max_iters):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=msgs, tools=self._agent_tools,
                    tool_choice="auto", max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature, timeout=30)
            except Exception as exc:
                self.logger.warning("agent(%s) call failed: %s", name, str(exc)[:140])
                if self._is_rate_limit(exc):
                    self._cool(name)
                ci += 1
                if ci >= len(clients):
                    return None
                name, client, model = clients[ci]
                self.logger.info("agent retrying with tool-capable fallback: %s", name)
                continue
            m = resp.choices[0].message
            calls = getattr(m, "tool_calls", None)
            if not calls:
                return (m.content or "").strip() or None
            msgs.append({
                "role": "assistant", "content": m.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in calls],
            })
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                self.logger.info("agent tool: %s(%s)", c.function.name, args)
                try:
                    result = self._agent_dispatch(c.function.name, args)
                except Exception as exc:
                    result = f"error: {exc}"
                msgs.append({"role": "tool", "tool_call_id": c.id,
                             "content": str(result)[:1500]})
        return None  # ran out of iterations -> caller falls back

    def analyze_intent(self, text: str) -> Dict[str, Any]:
        """Classify a command into an intent dict {type, entities, confidence}.

        Uses the fast local keyword classifier. This is deliberate: it is
        reliable, instantaneous, and avoids spending a whole LLM turn (and a
        possible web search) just to label the utterance. Conversational
        intents fall through to think() which does the heavy lifting.
        """
        return self._keyword_intent(text)

    # ------------------------------------------------------------------
    # Web search (retrieval-augmented answers)
    # ------------------------------------------------------------------
    def _needs_web(self, text: str) -> bool:
        """Heuristic: does this question benefit from a live web lookup?"""
        t = text.lower().strip()
        if len(t) < 8:
            return False
        # Explicit requests always search.
        if any(k in t for k in ("search", "look up", "look it up", "google",
                                "on the web", "latest", "news", "weather",
                                "current", "today", "right now", "price of",
                                "stock", "score", "who won", "release date")):
            return True
        # Personal / self-referential / control / math utterances should NOT search.
        if any(k in t for k in ("my name", "call me", "remember that", "turn on",
                                "turn off", "your name", "how are you", "thank",
                                "who are you", "are you", "what can you do",
                                "what do you see", "what did i", "plus", "minus",
                                "times", "divided", "calculate", " + ", " - ",
                                "the time", "what time", "the date", "what day",
                                "today's date")):
            return False
        # Factual question shapes — but only if it mentions a real topic (a noun
        # beyond the question word), to avoid searching trivial questions.
        starts = ("what is", "what are", "who is", "who was", "who are", "when",
                  "where", "which", "how many", "how much", "how far", "how old",
                  "define", "tell me about")
        if t.startswith(starts) and len(t.split()) >= 4:
            return True
        if t.endswith("?") and len(t.split()) >= 5:
            return True
        return False

    def _maybe_web_search(self, user_input: str) -> str:
        if not (self.cfg.web_search_enabled and self.cfg.web_search_auto):
            return ""
        # Weather questions -> live data from wttr.in (search engines only return
        # article links, not current numbers).
        low = user_input.lower()
        if any(k in low for k in ("weather", "forecast", "temperature",
                                  "how hot", "how cold", "how's the weather")):
            try:
                from modules.ai.web_search import get_weather
                w = get_weather(user_input, timeout=self.cfg.web_search_timeout)
                if w:
                    self.logger.info("Weather lookup used for: %s", user_input[:60])
                    return f"Current weather: {w}"
            except Exception as exc:
                self.logger.info("Weather skipped: %s", exc)
        if not self._needs_web(user_input):
            return ""
        try:
            from modules.ai.web_search import search_web, format_results
            results = search_web(
                user_input,
                max_results=self.cfg.web_search_max_results,
                region=self.cfg.web_search_region,
                timeout=self.cfg.web_search_timeout,
            )
            formatted = format_results(results, limit=self.cfg.web_search_max_results)
            if formatted:
                self.logger.info("Web search used for: %s", user_input[:60])
                return f"Web search results:\n{formatted}"
        except Exception as exc:
            self.logger.info("Web search skipped: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # Hybrid cloud escalation
    # ------------------------------------------------------------------
    def _should_escalate(self, text: str) -> bool:
        if not self._escalation_client:
            return False
        t = text.lower()
        words = t.split()
        if len(words) > 25:
            return True
        return any(k in t for k in (
            "explain", "why", "analyze", "compare", "step by step", "write a",
            "write me", "code", "plan", "calculate", "prove", "summarize",
            "in detail", "pros and cons",
        ))

    def _query_escalation(self, full_input: str) -> str:
        """Answer a hard question via the cloud (hybrid mode). Returns '' on failure."""
        try:
            history = self._conversation_history[-self._max_history:]
            if self._escalation_provider == "anthropic":
                resp = self._escalation_client.messages.create(
                    model=self.cfg.anthropic_model,
                    system=self.system_prompt,
                    messages=history,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                )
                self.logger.info("Answered via cloud escalation (anthropic)")
                return resp.content[0].text.strip()
            else:
                messages = [{"role": "system", "content": self.system_prompt}] + history
                resp = self._escalation_client.chat.completions.create(
                    model=self.cfg.openai_model,
                    messages=messages,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                )
                self.logger.info("Answered via cloud escalation (openai)")
                return resp.choices[0].message.content.strip()
        except Exception as exc:
            self.logger.info("Cloud escalation failed, using local: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Online backends
    # ------------------------------------------------------------------
    def _init_online(self) -> bool:
        provider = self.cfg.online_provider.lower()

        # Groq — free/fast, OpenAI-compatible. Reuses the openai client + the
        # openai backend/query path, just with Groq's base_url, key and model.
        if provider == "groq" and OPENAI_AVAILABLE and self.cfg.groq_api_key:
            try:
                self._openai_client = openai.OpenAI(
                    api_key=self.cfg.groq_api_key, base_url=self.cfg.groq_base_url)
                self.cfg.openai_model = self.cfg.groq_model  # _query_openai uses this
                self._backend = "openai"
                self.logger.info("AI engine: Groq (%s)", self.cfg.groq_model)
                return True
            except Exception as exc:
                self.logger.warning("Groq init failed: %s", exc)

        if provider == "openai" and OPENAI_AVAILABLE and self.cfg.openai_api_key:
            try:
                kwargs = {"api_key": self.cfg.openai_api_key}
                if self.cfg.openai_base_url:
                    kwargs["base_url"] = self.cfg.openai_base_url
                self._openai_client = openai.OpenAI(**kwargs)
                self._backend = "openai"
                self.logger.info("AI engine: OpenAI (%s)", self.cfg.openai_model)
                return True
            except Exception as exc:
                self.logger.warning("OpenAI init failed: %s", exc)

        if provider == "anthropic" and ANTHROPIC_AVAILABLE and self.cfg.anthropic_api_key:
            try:
                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.cfg.anthropic_api_key
                )
                self._backend = "anthropic"
                self.logger.info("AI engine: Anthropic (%s)", self.cfg.anthropic_model)
                return True
            except Exception as exc:
                self.logger.warning("Anthropic init failed: %s", exc)

        # Try the other provider as fallback
        if provider != "openai" and OPENAI_AVAILABLE and self.cfg.openai_api_key:
            try:
                self._openai_client = openai.OpenAI(api_key=self.cfg.openai_api_key)
                self._backend = "openai"
                self.logger.info("AI engine: OpenAI fallback (%s)", self.cfg.openai_model)
                return True
            except Exception:
                pass

        if provider != "anthropic" and ANTHROPIC_AVAILABLE and self.cfg.anthropic_api_key:
            try:
                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.cfg.anthropic_api_key
                )
                self._backend = "anthropic"
                self.logger.info("AI engine: Anthropic fallback (%s)", self.cfg.anthropic_model)
                return True
            except Exception:
                pass

        return False

    def _query_openai(self, user_input: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._conversation_history[-self._max_history:])

        try:
            response = self._openai_client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            self.logger.error("OpenAI query failed: %s", exc)
            # Try offline fallback
            if self._llama:
                return self._query_llama(user_input)
            return self._rule_based_response(user_input)

    def _query_anthropic(self, user_input: str) -> str:
        messages = self._conversation_history[-self._max_history:]

        try:
            response = self._anthropic_client.messages.create(
                model=self.cfg.anthropic_model,
                system=self.system_prompt,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
            return response.content[0].text.strip()
        except Exception as exc:
            self.logger.error("Anthropic query failed: %s", exc)
            if self._llama:
                return self._query_llama(user_input)
            return self._rule_based_response(user_input)

    # ------------------------------------------------------------------
    # Hailo-10H NPU backend (via hailo-ollama REST API)
    # ------------------------------------------------------------------
    def _hailo_loaded_models(self) -> List[str]:
        """Models actually pulled/compiled and ready to chat (via /api/tags).

        Note: /hailo/v1/list is only the *downloadable catalogue*; a model there
        is not necessarily installed. /api/tags reflects what can be used now.
        """
        try:
            url = f"{self.cfg.hailo_ollama_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            return [m.get("name") or m.get("model") for m in data.get("models", [])
                    if (m.get("name") or m.get("model"))]
        except Exception:
            return []

    def _init_hailo_ollama(self) -> bool:
        """Select an actually-available hailo-ollama model on the Hailo-10H."""
        loaded = self._hailo_loaded_models()
        if not loaded:
            self.logger.info(
                "hailo-ollama has no installed models yet "
                "(pull one, e.g. curl -X POST %s/hailo/v1/pull -d '{\"model\":\"...\"}')",
                self.cfg.hailo_ollama_url,
            )
            return False

        if self.cfg.hailo_ollama_model not in loaded:
            self.logger.info(
                "Configured NPU model '%s' not installed; using '%s'. "
                "Installed: %s",
                self.cfg.hailo_ollama_model, loaded[0], ", ".join(loaded),
            )
            self.cfg.hailo_ollama_model = loaded[0]

        self._backend = "hailo_ollama"
        self.logger.info(
            "AI engine: Hailo-10H NPU (%s) via hailo-ollama",
            self.cfg.hailo_ollama_model,
        )
        return True

    def _query_hailo_ollama(self, user_input: str) -> str:
        """Send a chat request to the hailo-ollama REST API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._conversation_history[-self._max_history:])

        payload = {
            "model": self.cfg.hailo_ollama_model,
            "messages": messages,
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                f"{self.cfg.hailo_ollama_url}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode()

            # hailo-ollama may return streaming NDJSON even with stream=false
            full_content = []
            for line in raw.strip().split("\n"):
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                if content:
                    full_content.append(content)

            return "".join(full_content).strip() or "I'm not sure how to respond."
        except Exception as exc:
            self.logger.error("hailo-ollama query failed: %s", exc)
            # Try llama.cpp fallback
            if self._llama:
                return self._query_llama(user_input)
            return self._rule_based_response(user_input)

    # ------------------------------------------------------------------
    # Offline backend
    # ------------------------------------------------------------------
    def _init_offline(self) -> bool:
        if not LLAMA_AVAILABLE:
            self.logger.info("llama-cpp-python not installed — offline LLM unavailable")
            return False

        model_path = Path(self.cfg.offline_model_path)
        if not model_path.exists():
            self.logger.warning(
                "Offline model not found at %s. Download a GGUF model "
                "(e.g. TinyLlama-1.1B-Chat Q4_K_M) and place it there.",
                model_path,
            )
            return False

        try:
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=self.cfg.offline_context_length,
                n_threads=self.cfg.offline_threads,
                n_gpu_layers=self.cfg.offline_gpu_layers,
                verbose=False,
            )
            self._backend = "llama"
            self.logger.info("AI engine: llama.cpp offline (%s)", model_path.name)
            return True
        except Exception as exc:
            self.logger.error("llama.cpp init failed: %s", exc)
            return False

    def _query_llama(self, user_input: str) -> str:
        if not self._llama:
            return self._rule_based_response(user_input)

        # Build a simple chat prompt
        prompt = f"<|system|>\n{self.system_prompt}\n"
        for msg in self._conversation_history[-6:]:  # Keep short for speed
            role = msg["role"]
            prompt += f"<|{role}|>\n{msg['content']}\n"
        prompt += "<|assistant|>\n"

        try:
            with self._lock:
                output = self._llama(
                    prompt,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    stop=["<|user|>", "<|system|>", "\n\n"],
                )
            return output["choices"][0]["text"].strip()
        except Exception as exc:
            self.logger.error("llama.cpp inference error: %s", exc)
            return self._rule_based_response(user_input)

    # ------------------------------------------------------------------
    # Rule-based fallback (no LLM)
    # ------------------------------------------------------------------
    def _rule_based_response(self, text: str) -> str:
        """Keyword-driven response when no LLM is available."""
        text_lower = text.lower()

        if any(w in text_lower for w in ("hello", "hi", "hey")):
            return behavior_config.greeting_message.format(name=behavior_config.robot_name)
        if any(w in text_lower for w in ("bye", "goodbye", "see you")):
            return behavior_config.goodbye_message
        if "name" in text_lower and "your" in text_lower:
            return f"My name is {behavior_config.robot_name}."
        if any(w in text_lower for w in ("time", "clock")):
            from datetime import datetime
            return f"It is {datetime.now().strftime('%H:%M')}."
        if "thank" in text_lower:
            return "You're welcome!"

        return "I understand. How can I help you?"

    # ------------------------------------------------------------------
    # Intent parsing helpers
    # ------------------------------------------------------------------
    def _parse_intent_json(self, raw: str, original_text: str) -> Dict[str, Any]:
        """Try to extract JSON from LLM output; fall back to keywords."""
        try:
            # Find JSON in response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return self._keyword_intent(original_text)

    @staticmethod
    def _device_control_intent(text_lower: str) -> Optional[Dict[str, Any]]:
        """Detect real-world device commands like 'turn on the light'.

        Returns entities {action, target} or None if this isn't a device command.
        Checked before 'movement' so 'turn on ...' isn't mistaken for driving.
        """
        # action phrase -> normalized action
        on_phrases = ("turn on", "switch on", "power on", "activate", "enable")
        off_phrases = ("turn off", "switch off", "power off", "deactivate", "disable")
        open_phrases = ("open",)
        close_phrases = ("close", "shut")

        action = None
        if any(p in text_lower for p in on_phrases):
            action = "on"
        elif any(p in text_lower for p in off_phrases):
            action = "off"
        elif any(p in text_lower for p in open_phrases):
            action = "open"
        elif any(p in text_lower for p in close_phrases):
            action = "close"
        if action is None:
            return None

        # Known controllable things (extend via config device_map).
        devices = ("light", "lights", "lamp", "fan", "door", "plug", "socket",
                   "outlet", "tv", "heater", "ac", "air conditioner", "curtain",
                   "blind", "gate", "pump", "relay", "led")
        target = None
        for d in devices:
            if d in text_lower:
                target = d.rstrip("s") if d.endswith("s") else d
                break
        if target is None:
            return None
        return {"action": action, "target": target}

    @staticmethod
    def _keyword_intent(text: str) -> Dict[str, Any]:
        """Simple keyword-based intent extraction (always available)."""
        text_lower = text.lower()
        intent: Dict[str, Any] = {"type": "conversation", "entities": {}, "confidence": 0.5}

        device = AIEngine._device_control_intent(text_lower)
        if device is not None:
            intent["type"] = "device_control"
            intent["entities"] = device
            intent["confidence"] = 0.9
        elif any(w in text_lower for w in ("move", "go", "come", "follow", "stop", "turn")):
            intent["type"] = "movement"
            intent["confidence"] = 0.8
            for d in ("forward", "backward", "left", "right"):
                if d in text_lower:
                    intent["entities"]["direction"] = d
        elif any(w in text_lower for w in ("what", "who", "where", "when", "how")):
            intent["type"] = "query"
            intent["confidence"] = 0.7
        elif any(w in text_lower for w in ("find", "look", "search", "detect")):
            intent["type"] = "object_interaction"
            intent["confidence"] = 0.7
        elif any(w in text_lower for w in ("learn", "remember", "teach")):
            intent["type"] = "learning"
            intent["confidence"] = 0.8
        elif any(w in text_lower for w in ("shutdown", "restart", "sleep")):
            intent["type"] = "system"
            intent["confidence"] = 0.9
        elif any(w in text_lower for w in ("hello", "hi", "hey", "bye", "goodbye")):
            intent["type"] = "social"
            intent["confidence"] = 0.9

        return intent

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _trim_history(self) -> None:
        if len(self._conversation_history) > self._max_history * 2:
            self._conversation_history = self._conversation_history[-self._max_history:]

    def clear_history(self) -> None:
        self._conversation_history.clear()

    # ------------------------------------------------------------------
    # Memory recall — injects past knowledge into prompts
    # ------------------------------------------------------------------
    def _recall_context(self, user_input: str) -> str:
        """Build a context string from the learning DB."""
        if not self.learning_db:
            return ""

        parts: List[str] = []

        # 1. Relevant past memories
        try:
            memories = self.learning_db.recall_memories(user_input, limit=3)
            if memories:
                mem_texts = [m["content"] for m in memories]
                parts.append(f"Relevant memories: {'; '.join(mem_texts)}")
        except Exception:
            pass

        # (Deliberately NOT auto-injecting recent raw chat lines — that made her
        # drag old topics into unrelated turns. Each conversation starts a clean
        # page; the transcript is still saved to the DB, and curated memories +
        # preferences below give continuity.)

        # 3. User preferences + grudge mood
        try:
            if self._current_user:
                prefs = self.learning_db.get_all_preferences(self._current_user)
                if str(prefs.get('grudge', '0')) == '1':
                    parts.append("MOOD: you are upset with THIS user — they insulted you "
                                 "and have NOT apologised. Be curt, cold and reluctant to "
                                 "help, and let them know you're still upset, until they "
                                 "say sorry.")
                shown = {k: v for k, v in prefs.items() if k != 'grudge'}
                if shown:
                    pref_str = ", ".join(f"{k}={v}" for k, v in shown.items())
                    parts.append(f"User preferences: {pref_str}")
        except Exception:
            pass

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Automatic preference extraction
    # ------------------------------------------------------------------
    def _extract_preferences(self, user_input: str, reply: str) -> None:
        """Use keyword heuristics to auto-save user preferences."""
        if not self.learning_db or not self._current_user:
            return

        text = user_input.lower()
        triggers = {
            "my name is": "name",
            "call me": "name",
            "i like": "likes",
            "i love": "likes",
            "i prefer": "preference",
            "i hate": "dislikes",
            "i don't like": "dislikes",
            "my favorite": "favorite",
        }

        for phrase, pref_key in triggers.items():
            idx = text.find(phrase)
            if idx >= 0:
                value = user_input[idx + len(phrase):].strip().rstrip(".!?,")
                if value:
                    try:
                        self.learning_db.set_preference(
                            self._current_user, pref_key, value
                        )
                        self.logger.info(
                            "Auto-saved preference %s=%s for user %s",
                            pref_key, value, self._current_user,
                        )
                    except Exception:
                        pass
                break  # Save only the first match per turn
