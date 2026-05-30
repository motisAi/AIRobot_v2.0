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

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# System prompt shared across backends
SYSTEM_PROMPT = (
    "You are {name}, a helpful and intelligent robot assistant. "
    "You have cameras, microphones, and sensors. You can see objects, "
    "recognise faces, and understand speech. Keep your answers short, "
    "natural, and friendly. When you receive context about what you see "
    "or hear, use it in your response. If you don't know something, say so. "
    "Your master's name is stored in your security settings — only obey "
    "privileged commands from authenticated users."
)


@dataclass
class AIConfig:
    """Runtime configuration for the AI engine."""

    # Mode: "online", "offline", "auto" (try online, fallback offline)
    mode: str = os.getenv("AI_MODE", "auto")

    # Online provider
    online_provider: str = os.getenv("AI_PROVIDER", "openai")  # openai | anthropic
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    max_tokens: int = 300
    temperature: float = 0.7

    # Offline model (llama.cpp GGUF)
    offline_model_path: str = os.getenv(
        "OFFLINE_MODEL_PATH",
        str(MODELS_DIR / "llm" / "phi-3-mini-4k-instruct.Q4_K_M.gguf"),
    )
    offline_context_length: int = 4096
    offline_threads: int = 4
    offline_gpu_layers: int = 0  # RPi5 has no CUDA; set >0 if you add GPU offload


class AIEngine:
    """Unified interface for online and offline LLM inference."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cfg = config or AIConfig()

        self._backend: Optional[str] = None  # "openai", "anthropic", "llama", None
        self._llama: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None
        self._lock = threading.Lock()

        self._conversation_history: List[Dict[str, str]] = []
        self._max_history = 20  # Keep last N turns

        # Reference to learning DB — set by main.py for memory recall
        self.learning_db = None
        # Current user ID — set by brain when a user is identified
        self._current_user: Optional[str] = None

        self.system_prompt = SYSTEM_PROMPT.format(name=behavior_config.robot_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Initialize the best available backend based on config mode."""
        mode = self.cfg.mode.lower()

        if mode in ("online", "auto"):
            if self._init_online():
                return True
            if mode == "online":
                self.logger.error("Online mode requested but no provider available")
                return False
            # auto → fall through to offline

        if mode in ("offline", "auto"):
            if self._init_offline():
                return True

        # Last resort: rule-based (no LLM at all)
        self._backend = "rules"
        self.logger.warning("No LLM backend available — using rule-based responses only")
        return True

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

        full_input = f"[Context: {ctx_str}] {user_input}" if ctx_str else user_input

        # Add to history
        self._conversation_history.append({"role": "user", "content": full_input})
        self._trim_history()

        # Dispatch to backend
        if self._backend == "openai":
            reply = self._query_openai(full_input)
        elif self._backend == "anthropic":
            reply = self._query_anthropic(full_input)
        elif self._backend == "llama":
            reply = self._query_llama(full_input)
        else:
            reply = self._rule_based_response(user_input)

        self._conversation_history.append({"role": "assistant", "content": reply})

        # Automatically extract and save preferences from the conversation
        self._extract_preferences(user_input, reply)

        return reply

    def analyze_intent(self, text: str) -> Dict[str, Any]:
        """Use the LLM to extract structured intent from speech.

        Returns a dict with keys: type, entities, confidence.
        Falls back to keyword matching when no LLM is available.
        """
        if self._backend in ("openai", "anthropic", "llama"):
            prompt = (
                "Analyze this robot command and return ONLY valid JSON with keys: "
                '"type" (one of: movement, query, learning, system, social, '
                'object_interaction, conversation), '
                '"entities" (relevant parameters as dict), '
                '"confidence" (float 0-1). '
                f'Command: "{text}"'
            )
            raw = self.think(prompt)
            # Remove from conversation history (internal query)
            self._conversation_history = self._conversation_history[:-2]
            return self._parse_intent_json(raw, text)
        else:
            return self._keyword_intent(text)

    # ------------------------------------------------------------------
    # Online backends
    # ------------------------------------------------------------------
    def _init_online(self) -> bool:
        provider = self.cfg.online_provider.lower()

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
    def _keyword_intent(text: str) -> Dict[str, Any]:
        """Simple keyword-based intent extraction (always available)."""
        text_lower = text.lower()
        intent: Dict[str, Any] = {"type": "conversation", "entities": {}, "confidence": 0.5}

        if any(w in text_lower for w in ("move", "go", "come", "follow", "stop", "turn")):
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

        # 2. Recent conversation with this user
        try:
            if self._current_user:
                recent = self.learning_db.get_recent_conversations(
                    limit=4, user_id=self._current_user
                )
                if recent:
                    lines = [f"{r['role']}: {r['content']}" for r in recent]
                    parts.append(f"Recent chat with this user: {' | '.join(lines)}")
        except Exception:
            pass

        # 3. User preferences
        try:
            if self._current_user:
                prefs = self.learning_db.get_all_preferences(self._current_user)
                if prefs:
                    pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items())
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
