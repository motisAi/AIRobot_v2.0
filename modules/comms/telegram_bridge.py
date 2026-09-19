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
import subprocess
import tempfile
import threading
import uuid
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
        self._reply_with_voice = False

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
        if getattr(self, "_reply_with_voice", False) and not text.startswith("🎤"):
            self._send_voice(text)

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
        if not text and (m.get("voice") or m.get("audio")):
            self._handle_voice(m.get("voice") or m.get("audio"))
            return
        if not text:
            if m.get("photo"):
                self._send("I can't read photos you send yet — but ask me "
                           "\"what do you see?\" and I'll look through my own camera.")
            return
        logger.info("received from phone: %r", text[:80])
        self._process(text.strip())

    # -- voice notes from the phone ------------------------------------------
    def _download_file(self, file_id: str) -> bytes:
        info = self._api("getFile", {"file_id": file_id}, timeout=20)
        path = info["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{self.token}/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()

    def _handle_voice(self, media: dict):
        """Voice note (OGG/Opus) -> 16 kHz mono PCM -> the same STT chain her ears use."""
        dur = int(media.get("duration", 0) or 0)
        if dur > 60:
            self._send("That voice message is over a minute — please keep commands under 60 seconds.")
            return
        speech = self.robot.modules.get("speech_recognition") if hasattr(self.robot, "modules") else None
        if speech is None or not hasattr(speech, "_transcribe_pcm16k"):
            self._send("I can't transcribe voice right now (speech module not loaded).")
            return
        try:
            blob = self._download_file(media["file_id"])
            proc = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
                input=blob, capture_output=True, timeout=60)
            pcm = proc.stdout
            if proc.returncode != 0 or len(pcm) < 3200:   # < 0.1 s
                logger.warning("voice decode failed: %s", proc.stderr.decode("utf-8", "ignore")[:160])
                self._send("I couldn't decode that voice message.")
                return
            text = speech._transcribe_pcm16k(pcm)
        except Exception as exc:
            logger.warning("voice transcription failed: %s", exc)
            self._send("Sorry, I couldn't understand that voice message.")
            return
        if not text:
            self._send("I heard the voice note but couldn't make out any words.")
            return
        logger.info("voice from phone (%ds): %r", dur, text[:80])
        self._send(f"🎤 heard: “{text}”")
        self._reply_with_voice = True
        try:
            self._process(text.strip())
        finally:
            self._reply_with_voice = False

    def _send_voice(self, text: str) -> bool:
        """Speak the reply as a Telegram voice note in Stella's own Piper voice."""
        tts = self.robot.modules.get("tts") if hasattr(self.robot, "modules") else None
        if tts is None or not hasattr(tts, "_generate_audio") or not text:
            return False
        wav = ogg = None
        try:
            wav = tts._generate_audio(text[:600])
            if not wav:
                return False
            ogg = os.path.join(tempfile.gettempdir(), f"stella_reply_{uuid.uuid4().hex}.ogg")
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", "libopus",
                                "-b:a", "32k", "-ac", "1", ogg], capture_output=True, timeout=30)
            if r.returncode != 0:
                return False
            boundary = uuid.uuid4().hex
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{self.chat_id}\r\n"
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"voice\"; filename=\"reply.ogg\"\r\n"
                    f"Content-Type: audio/ogg\r\n\r\n").encode() + open(ogg, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendVoice", data=body,
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=30):
                return True
        except Exception as exc:
            logger.debug("voice reply skipped: %s", exc)
            return False
        finally:
            for f in (wav, ogg):
                try:
                    if f and os.path.exists(f):
                        os.remove(f)
                except OSError:
                    pass

    # -- speak-aloud intent --------------------------------------------------
    _OUTLOUD = r"(?:out ?loud|aloud|on (?:the |your )?speakers?|over (?:the )?speakers?|through (?:the )?speakers?|loudly)"

    def _extract_speak_aloud(self, text: str):
        """Return the phrase to speak if this is a 'say ... out loud' command, else None.

        Returns "" if the intent is clear but no phrase was given (ask back).
        """
        import re as _re
        raw = (text or "").strip()
        low = raw.lower()
        prefix = (r"(?:stella[,\s]+|hey stella[,\s]+|ok(?:ay)?[,\s]+|please[,\s]+|"
                  r"can you[,\s]+|could you[,\s]+|would you[,\s]+|i want you to[,\s]+|i need you to[,\s]+)*")
        m = None
        # (a) say/speak/read + an explicit out-loud / speaker cue (cue may be after the verb OR trailing)
        if _re.search(self._OUTLOUD, low) and _re.search(r"\b(say|speak|read)\b", low):
            m = _re.match(prefix + r"(?:say|speak|read(?:\s+this|\s+it)?)"
                          r"(?:\s+" + self._OUTLOUD + r")?[\s:,\-]+(.*)$",
                          raw, _re.IGNORECASE | _re.DOTALL)
        # (b) announce/broadcast — always a speak-aloud, no cue needed
        if m is None and _re.match(prefix + r"(?:announce|broadcast)\b", low):
            m = _re.match(prefix + r"(?:announce|broadcast)(?:\s+that)?[\s:,\-]+(.*)$",
                          raw, _re.IGNORECASE | _re.DOTALL)
        if not m:
            return None
        phrase = m.group(1).strip()
        # drop a trailing "... out loud / on the speaker", then any wrapping quotes/punct
        phrase = _re.sub(r"[\s,]*" + self._OUTLOUD + r"\s*[.!]?$", "", phrase, flags=_re.IGNORECASE).strip()
        phrase = phrase.strip('"“”\' :,').strip()
        return phrase

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

        # --- screen face: show / hide from the phone ---
        if ("face" in low or "screen" in low) and any(v in low for v in
                ("show", "open", "hide", "close", "turn off", "turn on", "wake")):
            f = getattr(self.robot, "face", None)
            if f is not None:
                if any(v in low for v in ("hide", "close", "turn off")):
                    f.hide_face(); self._send("Okay, hiding my face.")
                else:
                    f.show_face(); self._send("Showing my face on the screen.")
                return

        # Stop the hand mirror from the phone (in case voice is busy).
        if any(k in low for k in ("stop copying", "stop mirroring", "stop imitating", "stop copy")):
            hm = getattr(self.robot, "hand_mirror", None)
            if hm is not None:
                hm.set_active(False)
                self._send("Okay, I stopped copying your hand.")
                return

        # --- speak a phrase OUT LOUD on the physical speaker (from the phone) ---
        # Triggers on "say/speak/read ... out loud|on the speaker" or "announce/broadcast ...".
        # Handled directly so it works even when the cloud brain is rate-limited, and so
        # she SPEAKS instead of just typing the words back to the chat.
        phrase = self._extract_speak_aloud(text)
        if phrase is not None:
            if not phrase:
                self._send("What would you like me to say out loud?")
                return
            tts = self.robot.modules.get("tts") if hasattr(self.robot, "modules") else None
            if tts is not None and hasattr(tts, "speak"):
                tts.speak(phrase)
                self._send(f'🔊 Saying it out loud: "{phrase}"')
            else:
                self._send("My speaker isn't available right now.")
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
                       "• \"guard on\" / \"guard off\" / \"status\"\n"
                       "• \"say <something> out loud\" — I speak it on my own speaker\n"
                       "• or just send me a 🎤 voice message — I'll answer in my voice too")
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
