"""Two-way Telegram chat with Stella.

Long-polls Telegram for messages from the master's chat and routes them to the
agent brain (so you can ask questions, use tools — "what do you see", "turn on
the light", "remind me in 10 minutes…", "guard on/off" — from your phone,
anywhere the Pi has internet). Replies go back through the bot.

Only the configured TELEGRAM_CHAT_ID is served (everyone else is ignored).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
import urllib.request

logger = logging.getLogger("Telegram")


class TelegramBridge:
    def __init__(self, robot):
        self.robot = robot
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = str(os.getenv("TELEGRAM_CHAT_ID", ""))
        self._stop = threading.Event()
        self._thread = None
        self._offset = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.token and self.chat_id)

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if not self.available or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="telegram")
        self._thread.start()
        logger.info("Telegram two-way chat bridge started")

    def stop(self):
        self._stop.set()

    # -- telegram api ------------------------------------------------------
    def _api(self, method: str, params=None, timeout=40):
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = urllib.parse.urlencode(params or {}).encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _send(self, text: str):
        try:
            self._api("sendMessage", {"chat_id": self.chat_id, "text": text}, timeout=15)
        except Exception as exc:
            logger.warning("send failed: %s", exc)

    # -- poll loop ---------------------------------------------------------
    def _loop(self):
        # Skip any backlog so we only respond to new messages.
        try:
            d = self._api("getUpdates", {"timeout": 0}, timeout=15)
            for u in d.get("result", []):
                self._offset = u["update_id"] + 1
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                d = self._api("getUpdates", {"timeout": 30, "offset": self._offset}, timeout=40)
            except Exception as exc:
                logger.debug("getUpdates error (will retry): %s", str(exc)[:100])
                self._stop.wait(3)
                continue
            for u in d.get("result", []):
                self._offset = u["update_id"] + 1
                try:
                    self._handle(u)
                except Exception as exc:
                    logger.warning("handle error: %s", exc)

    def _handle(self, update: dict):
        m = update.get("message") or update.get("edited_message") or {}
        if str(m.get("chat", {}).get("id", "")) != self.chat_id:
            return  # only the master's chat
        text = m.get("text")
        if not text:
            if m.get("photo"):
                self._send("I can't read photos you send yet — but ask me "
                           "\"what do you see?\" and I'll look through my own camera.")
            return
        logger.info("received from phone: %r", text[:80])
        self._process(text.strip())

    # -- command handling --------------------------------------------------
    def _process(self, text: str):
        low = text.lower().strip()
        brain = self.robot.brain

        # --- Guard arm/disarm: broad keywords, handled DIRECTLY (never needs the
        #     LLM, so it works even when Groq is rate-limited). Also clears a pending Q.
        import time as _t
        ARM = ("guard on", "guard my home", "guard the house", "guard my house",
               "guard home", "arm guard", "arm the house", "protect my home",
               "protect the house", "secure the house", "watch the house", "start guard")
        DISARM = ("guard off", "disarm", "stand down", "turn off guard", "stop guard",
                  "i'm home", "im home", "i am home", "we're home")
        if any(k in low for k in ARM) or low in ("/guard_on", "/guardon", "arm"):
            self.robot._guard_pending = None
            brain.guard_mode = True
            self._send("🛡️ Guard mode ARMED — I'll watch and alert you if I see "
                       "someone I don't recognise.")
            return
        if any(k in low for k in DISARM) or low in ("/guard_off", "/guardoff"):
            self.robot._guard_pending = None
            brain.guard_mode = False
            self._send("Guard mode off. Welcome home.")
            return

        # Stop the hand mirror from the phone (in case voice is busy).
        if any(k in low for k in ("stop copying", "stop mirroring", "stop imitating", "stop copy")):
            hm = getattr(self.robot, "hand_mirror", None)
            if hm is not None:
                hm.set_active(False)
                self._send("Okay, I stopped copying your hand.")
                return

        # --- interactive guard flow: recognise? -> alarm? -> scream ---
        pending = getattr(self.robot, "_guard_pending", None)
        if pending and (_t.time() - getattr(self.robot, "_guard_pending_time", 0)) > 300:
            self.robot._guard_pending = None          # stale question -> forget it
            pending = None
        if pending:
            yes = low in ("yes", "y", "yeah", "yep", "yup", "sure", "affirmative") or low.startswith("yes")
            no = low in ("no", "n", "nope", "nah") or low.startswith("no")
            if pending == "recognize":
                if yes:
                    self.robot._guard_pending = None
                    self._send("Good — someone you know. No alarm. I'll keep watching.")
                    return
                if no:
                    self.robot._guard_pending = "alarm"
                    self._send("🚨 Should I sound the alarm at them? Reply YES to scream, NO to stand down.")
                    return
                self._send("Do you recognise this person? Please reply YES or NO.")
                return
            if pending == "alarm":
                self.robot._guard_pending = None
                if yes:
                    self._send("🚨 Sounding the alarm now!")
                    threading.Thread(target=self.robot._guard_scream, daemon=True).start()
                    return
                self._send("Okay, standing down. I'll keep watching quietly.")
                return

        if low in ("/start", "/help", "help"):
            self._send("Hi Moti! I'm Stella. Text me anything:\n"
                       "• ask questions (I can search the web)\n"
                       "• \"what do you see?\" (I look through my camera)\n"
                       "• \"turn on the light\"\n"
                       "• \"remind me in 10 minutes to…\"\n"
                       "• \"guard on\" / \"guard off\" / \"status\"")
            return
        if low in ("/guard_on", "/guardon", "guard on", "arm", "arm guard"):
            brain.guard_mode = True
            self._send("🛡️ Guard mode ARMED — I'll alert you if I see someone I don't recognise.")
            return
        if low in ("/guard_off", "/guardoff", "guard off", "disarm", "i'm home", "im home"):
            brain.guard_mode = False
            self._send("Guard mode off. Welcome home.")
            return
        if low in ("/status", "status"):
            st = getattr(brain, "state", "?")
            self._send(f"Status — state: {getattr(st, 'name', st)} · "
                       f"guard: {'ON' if brain.guard_mode else 'off'} · "
                       f"user: {getattr(brain, 'current_user_name', None) or 'unknown'}")
            return

        engine = self.robot.ai_engine
        if engine is None:
            self._send("My brain isn't available right now.")
            return

        # Fresh context when a new chat starts (gap since the last text), so old
        # voice/telegram topics don't bleed into an unrelated question.
        import time as _t
        now = _t.time()
        if now - getattr(self, "_last_msg", 0) > 90:
            try:
                engine.clear_history()
            except Exception:
                pass
        self._last_msg = now

        # Serialize with voice conversations to avoid interleaved history.
        with self._lock:
            self.robot._remote_master = True   # allow master-only tools from phone
            prev_user = engine._current_user
            try:
                engine._current_user = getattr(brain, "current_user", None) or "master_001"
                reply = engine.think(text, context={"user": "Moti (texting from phone)"})
            except Exception as exc:
                reply = f"Sorry, I hit an error: {exc}"
            finally:
                self.robot._remote_master = False
                engine._current_user = prev_user
        self._send(reply or "(no reply)")
