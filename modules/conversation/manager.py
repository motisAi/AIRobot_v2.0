"""Conversation session manager.

Owns a full spoken conversation from wake word to farewell, with strict
single-microphone ownership so the wake mic and the command mic never run at
the same time:

    wake word  ->  pause wake mic
               ->  greet ("How can I help you, Moti?")
               ->  CONVERSE loop (command mic only):
                     listen one sentence (silence-detected) -> think -> speak
               ->  after `idle_timeout` s of silence: "Can I do anything else?"
               ->  no / negative reply: "See you later!"  ->  resume wake mic

All timings and phrases come from config/config.yaml -> conversation.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Optional

from config.settings import (conversation_config, behavior_config, security_config,
                             hand_config)


class ConversationManager:
    def __init__(self, robot):
        self.robot = robot                 # the AIRobot instance (has modules, brain, ai_engine)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cfg = conversation_config
        self._active = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._register_agent_tools()

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    def start_session(self):
        """Begin a conversation (called when the wake word fires)."""
        if self._active:
            return
        self._active = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="conversation", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def _speak(self, text: str):
        """Speak and wait, so we never record our own voice."""
        if not text:
            return
        print(f"\n{behavior_config.robot_name}: {text}\n", flush=True)
        self.logger.info("SAY: %s", text)   # goes to gonzo.log for debugging
        self._dashboard_log(f"{behavior_config.robot_name}: {text}")
        tts = self.robot.modules.get('tts')
        if tts and getattr(tts, 'running', False):
            try:
                # Prefer the reliable inline path; fall back to queued speak.
                if hasattr(tts, 'speak_blocking'):
                    ok = tts.speak_blocking(text)
                    if not ok:
                        self.logger.warning("TTS did not play this reply")
                else:
                    tts.speak(text, wait=True)
            except Exception as exc:
                self.logger.warning("speak failed: %s", exc)

    def _dashboard_log(self, msg: str):
        dash = getattr(self.robot, 'dashboard', None)
        if dash:
            try:
                dash.add_log(msg)
            except Exception:
                pass

    def _name_suffix(self) -> str:
        name = getattr(self.robot.brain, 'current_user_name', None)
        return f", {name}" if name else ""

    # ------------------------------------------------------------------
    def _run(self):
        brain = self.robot.brain
        speech = self.robot.modules.get('speech_recognition')
        cfg = self.cfg

        if speech is None:
            self._active = False
            return

        # Fresh short-term memory per conversation, so a previous chat's topic
        # ("going out", "Tokyo") can't bleed into this one and confuse replies.
        # (Long-term per-user memory in the learning DB is unaffected.)
        engine = getattr(self.robot, 'ai_engine', None)
        if engine is not None:
            try:
                engine.clear_history()
            except Exception:
                pass

        # Duck any playing music so the user can be heard over it.
        mp = getattr(self.robot, 'music', None)
        if mp is not None:
            try:
                mp.duck()
            except Exception:
                pass

        # Take the wake mic offline for the whole session, and stop the passive
        # face-detection greeting from talking over us.
        brain._suppress_greetings = True
        try:
            self.robot._pause_wake_word_listener(reason="conversation")
        except Exception:
            pass

        try:
            # Verify WHO we're talking to first (face must be seen recently),
            # so we greet correctly and save memory to the right person.
            self._resolve_identity()

            # Unknown visitor -> ask their name and enroll a face.
            if not getattr(brain, 'authenticated', False) and behavior_config.learn_new_faces:
                self._speak(behavior_config.unknown_person_response)
                answer = speech.capture_utterance(cfg.idle_timeout, cfg.end_silence, cfg.max_utterance)
                if answer:
                    self._enroll_name(answer)

            # Greeting (+ wave hello if the hand is connected)
            if getattr(hand_config, 'wave_on_greeting', True):
                self._gesture('wave')
            self._speak(cfg.greeting.format(name=self._name_suffix()))

            wrapped = False
            while self.robot.running and not self._stop.is_set():
                text = speech.capture_utterance(cfg.idle_timeout, cfg.end_silence, cfg.max_utterance)

                if not text:
                    # Silence. First time -> wrap-up prompt; second time -> leave.
                    if not wrapped:
                        self._speak(cfg.wrap_up_prompt.format(name=self._name_suffix()))
                        wrapped = True
                        continue
                    self._farewell()
                    break

                print(f"\nYou: {text}", flush=True)
                self.logger.info("HEARD: %s", text)   # goes to gonzo.log
                self._dashboard_log(f"You: {text}")
                self._log_db('user', text)

                if self._is_end_phrase(text):
                    self._farewell()
                    break

                wrapped = False
                self._armed_end = False
                self._handle_utterance(text)
                if getattr(self, '_armed_end', False):
                    self._armed_end = False
                    break   # armed -> go quiet, back to just listening for the wake word
        except Exception as exc:
            self.logger.error("Conversation error: %s", exc)
        finally:
            brain._suppress_greetings = False
            # Clean page when the conversation ends: wipe the short-term chat so
            # the next one starts fresh (the transcript is already saved to the DB).
            engine2 = getattr(self.robot, 'ai_engine', None)
            if engine2 is not None:
                try:
                    engine2.clear_history()
                except Exception:
                    pass
            # Restore music volume now that Stella has stopped talking.
            _mp = getattr(self.robot, 'music', None)
            if _mp is not None:
                try:
                    _mp.unduck()
                except Exception:
                    pass
            # Mark inactive FIRST so _resume_wake_word_listener's "conversation
            # active" guard lets our own resume through.
            self._active = False
            try:
                self.robot._resume_wake_word_listener()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _is_end_phrase(self, text: str) -> bool:
        low = text.lower().strip().rstrip('.!?')
        words = low.split()
        if len(words) > 5:
            return False
        # "stop <something>" is a COMMAND (stop the music / stop copying), not a
        # goodbye — don't let the "stop" end-phrase swallow it.
        if low.startswith("stop ") and low != "stop":
            return False
        return any(low == p or low.startswith(p + ' ')
                   for p in self.cfg.end_phrases)

    def _handle_utterance(self, text: str):
        engine = self.robot.ai_engine
        brain = self.robot.brain

        # Real-world device commands ("turn on the light") -> microcontroller.
        try:
            intent = engine._keyword_intent(text)
        except Exception:
            intent = {'type': 'conversation', 'entities': {}}

        # Personality: forgive on apology, demo on a hypothetical, finger on a real insult.
        if self._maybe_apology(text):
            return
        if self._maybe_insult_demo(text):
            return
        if self._maybe_insult(text):
            return

        if intent.get('type') == 'device_control':
            self._do_device_control(intent.get('entities', {}))
            return

        # Home-guard / security mode (master only).
        if self._maybe_guard(text):
            return

        # Music: play/find + live controls (volume/pause/stop).
        if self._maybe_music(text):
            return

        # Hand mirror: "copy my hand" / "stop copying".
        if self._maybe_mirror(text):
            return

        # Vision requests ("what am I holding?", "what colour is this?").
        if self._maybe_vision(text):
            return

        # WiFi / connectivity requests handled locally (scan, status).
        if self._maybe_wifi(text):
            return

        # Fast, accurate local answers for time/date (no LLM/web needed).
        quick = self._quick_answer(text)
        if quick:
            self._speak(quick)
            self._log_db('assistant', quick)
            return

        # Otherwise: talk (LLM + web search + memory), with sensory context.
        if engine is not None:
            try:
                from datetime import datetime as _dt
                ctx = {'time': _dt.now().strftime('%H:%M'),
                       'date': _dt.now().strftime('%A, %B %d, %Y')}
                # Tell the brain who it's talking to ONLY if a face was recognised.
                if getattr(brain, 'authenticated', False):
                    who = getattr(brain, 'current_user_name', None) or getattr(brain, 'current_user', None)
                    if who:
                        ctx['user'] = who
                reply = engine.think(text, context=ctx)
            except Exception as exc:
                self.logger.error("think failed: %s", exc)
                reply = "Sorry, I had trouble thinking about that."
        else:
            reply = "My language engine is not available right now."

        self._speak(reply)
        self._log_db('assistant', reply)

    # ------------------------------------------------------------------
    # Agent tools — the LLM can call these and chain them
    # ------------------------------------------------------------------
    def _register_agent_tools(self):
        engine = getattr(self.robot, 'ai_engine', None)
        if engine is None or not hasattr(engine, 'register_tools'):
            return
        tools = [
            {"type": "function", "function": {
                "name": "get_weather", "description": "Current weather for a city.",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {
                "name": "web_search", "description": "Search the web for current facts/news.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "look", "description": "Look through the camera and answer a question about what is visible (objects, colours, text).",
                "parameters": {"type": "object", "properties": {
                    "question": {"type": "string"}}, "required": ["question"]}}},
            {"type": "function", "function": {
                "name": "control_device", "description": "Turn a home device on or off (light, fan, etc.). Master only.",
                "parameters": {"type": "object", "properties": {
                    "target": {"type": "string"},
                    "action": {"type": "string", "enum": ["on", "off"]}},
                    "required": ["target", "action"]}}},
            {"type": "function", "function": {
                "name": "set_reminder", "description": "Set a reminder to be announced after some minutes.",
                "parameters": {"type": "object", "properties": {
                    "text": {"type": "string"},
                    "minutes": {"type": "number"}}, "required": ["text", "minutes"]}}},
            {"type": "function", "function": {
                "name": "get_time", "description": "The current date and time. For a place other than here, pass its IANA timezone (e.g. Tokyo -> 'Asia/Tokyo', London -> 'Europe/London', New York -> 'America/New_York'). Omit for local time.",
                "parameters": {"type": "object", "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name, e.g. 'Asia/Tokyo'. Leave empty for local time."}}}}},
            {"type": "function", "function": {
                "name": "send_telegram", "description": "Send a TEXT message to the master's phone via Telegram (use when asked to 'text me' or notify).",
                "parameters": {"type": "object", "properties": {
                    "message": {"type": "string"}}, "required": ["message"]}}},
            {"type": "function", "function": {
                "name": "send_photo", "description": "Capture a REAL photo from the camera and send the actual image to the master's phone (use when asked to send/take a picture or photo).",
                "parameters": {"type": "object", "properties": {
                    "caption": {"type": "string"}}}}},
            {"type": "function", "function": {
                "name": "set_guard_mode", "description": "Arm or disarm home guard/security mode. When ON, an unrecognized person triggers a photo alert to the master's phone. Master only.",
                "parameters": {"type": "object", "properties": {
                    "on": {"type": "boolean"}}, "required": ["on"]}}},
            {"type": "function", "function": {
                "name": "play_music", "description": "Play a song or music by name from YouTube out loud through the speaker. Use when asked to play/put on a song or music.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "song title and/or artist"}},
                    "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "stop_music", "description": "Stop the music that is currently playing.",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "do_gesture", "description": "Make a hand gesture. wave (greet/bye), thumbs_up (pleased), point, peace, open, fist, count. middle_finger = raise 3s then lower. middle_finger_hold = raise and KEEP up until asked to lower. rest = lower the hand back to a fist.",
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string", "enum": ["wave", "thumbs_up", "point", "peace", "open", "fist", "count", "middle_finger", "middle_finger_hold", "rest"]},
                    "number": {"type": "integer", "description": "for count: how many fingers to hold up (0-5)"}},
                    "required": ["name"]}}},
        ]
        engine.register_tools(tools, self._agent_dispatch)

    def _agent_dispatch(self, name: str, args: dict):
        """Execute a tool the LLM asked for; return a short string result."""
        try:
            if name == "get_weather":
                from modules.ai.web_search import get_weather
                return get_weather("weather in " + str(args.get("city", ""))) or "weather unavailable"
            if name == "web_search":
                from modules.ai.web_search import search_web, format_results
                return format_results(search_web(str(args.get("query", "")), max_results=4)) or "no results found"
            if name == "look":
                vlm = getattr(self.robot, "vlm", None)
                if not (vlm and vlm.available):
                    return "vision is not available"
                return vlm.look(str(args.get("question", "What do you see?"))) or "couldn't see clearly"
            if name == "control_device":
                if not (getattr(self.robot.brain, "master_mode", False)
                        or getattr(self.robot, "_remote_master", False)):
                    return "denied: only the master can control devices"
                mc = self.robot.modules.get("microcontroller")
                target = str(args.get("target", "device")); action = str(args.get("action", "on"))
                if mc:
                    mc.set_output(target, action == "on")
                connected = bool(getattr(mc, "connected", False)) if mc else False
                return f"{action} {target}" + ("" if connected else " (logged only — no microcontroller connected yet)")
            if name == "set_reminder":
                rem = getattr(self.robot, "reminders", None)
                if not rem:
                    return "reminders not available"
                return rem.add(str(args.get("text", "reminder")), float(args.get("minutes", 1)))
            if name == "get_time":
                from datetime import datetime
                tz = str(args.get("timezone", "") or "").strip()
                if tz:
                    try:
                        from zoneinfo import ZoneInfo
                        return datetime.now(ZoneInfo(tz)).strftime(
                            f"%A %B %d, %-I:%M %p ({tz})")
                    except Exception as exc:
                        return f"couldn't get the time for '{tz}': {exc}"
                return datetime.now().strftime("%A %B %d, %-I:%M %p")
            if name == "send_telegram":
                notifier = getattr(self.robot, "notifier", None)
                if not (notifier and notifier.available):
                    return "Telegram is not configured"
                ok = notifier.send_message(str(args.get("message", "")))
                return "message sent to your phone" if ok else "failed to send"
            if name == "send_photo":
                notifier = getattr(self.robot, "notifier", None)
                if not (notifier and notifier.available):
                    return "Telegram is not configured"
                cm = getattr(self.robot, "camera_manager", None)
                frame = None
                try:
                    f = cm.get_latest_frame() if cm else None
                    frame = getattr(f, "image", f)
                except Exception:
                    frame = None
                if frame is None:
                    return "couldn't get a camera frame"
                ok = notifier.send_photo(frame, str(args.get("caption", "")) or "📸 From Stella's camera")
                return "photo sent to your phone" if ok else "failed to send the photo"
            if name == "play_music":
                mp = getattr(self.robot, "music", None)
                if not (mp and mp.available):
                    return "music player not available (needs ffmpeg + aplay + yt-dlp)"
                q = str(args.get("query", "")).strip()
                if not q:
                    return "no song specified"
                title = mp.search_title(q) or q
                return f"now playing {title}" if mp.play(q, title=title) else "couldn't start playback"
            if name == "stop_music":
                mp = getattr(self.robot, "music", None)
                if mp:
                    mp.stop()
                return "music stopped"
            if name == "do_gesture":
                hand = getattr(self.robot, "hand", None)
                if not (hand and getattr(hand, "available", False)):
                    return "the hand isn't connected"
                g = str(args.get("name", "")).strip().lower()
                if g == "count":
                    n = int(args.get("number", 0) or 0)
                    return f"counted {n}" if hand.count(n) else "couldn't count"
                return f"did {g}" if hand.gesture(g) else f"unknown gesture: {g}"
            if name == "set_guard_mode":
                on = bool(args.get("on", True))
                # Disarming is master-only (arming is open). Telegram sets
                # _remote_master; a present speaker needs a recent master face or
                # the disarm phrase (see _can_disarm).
                if not on and not getattr(self.robot, "_remote_master", False) \
                        and not self._can_disarm():
                    return "denied: only the master can disarm guard (say your phrase or use the phone)"
                self.robot.brain.guard_mode = on
                notifier = getattr(self.robot, "notifier", None)
                if on and notifier and notifier.available:
                    try:
                        notifier.send_message("🛡️ Guard mode ARMED.")
                    except Exception:
                        pass
                return "guard mode is now ON" if on else "guard mode is now OFF"
        except Exception as e:
            return f"error: {e}"
        return "unknown tool"

    def _resolve_identity(self):
        """Trust the current identity ONLY if a face was recognised in the last
        few seconds. Otherwise treat the speaker as unknown — so Stella doesn't
        claim to know someone she can't currently see, and saves data correctly."""
        import time as _t
        brain = self.robot.brain
        window = float(getattr(security_config, 'identity_memory_seconds', 60.0))
        recent = (_t.time() - getattr(brain, 'last_face_time', 0)) < window
        if recent and getattr(brain, 'current_user', None):
            if self.robot.ai_engine:
                self.robot.ai_engine._current_user = brain.current_user
            self.logger.info("Identity: %s (master=%s)",
                             getattr(brain, 'current_user_name', None) or brain.current_user,
                             getattr(brain, 'master_mode', False))
        else:
            brain.authenticated = False
            brain.master_mode = False
            brain.current_user = None
            brain.current_user_name = None
            if self.robot.ai_engine:
                self.robot.ai_engine._current_user = None
            self.logger.info("Identity: unknown (no recent face)")

    def _can_disarm(self, text_lower: str = "") -> bool:
        """Is the present speaker allowed to disarm guard? True if disarming is
        not locked to the master, OR the master's face was seen recently, OR the
        configured disarm pass-phrase was spoken."""
        import time as _t
        if not getattr(security_config, 'guard_require_master_to_disarm', True):
            return True
        brain = self.robot.brain
        window = float(getattr(security_config, 'identity_memory_seconds', 60.0))
        if (_t.time() - float(getattr(brain, 'last_master_time', 0) or 0)) < window:
            return True
        phrase = (getattr(security_config, 'guard_disarm_phrase', '') or '').strip().lower()
        return bool(phrase and phrase in text_lower)

    def _maybe_guard(self, text: str) -> bool:
        """Arm/disarm home-guard mode. Arming is open; disarming is master-only
        (see _can_disarm). Returns True if handled."""
        t = text.lower()
        # Don't act on negated / quoted mentions like "no, I did NOT say guard on"
        # — a naive keyword match would otherwise arm it. Let those flow to the brain.
        if any(neg in t for neg in ("not ", "n't", "do not", "did not", "never",
                                    "don't", "didn't", "stop saying", "quit saying")):
            return False
        on_kw = ("guard mode", "security mode", "home guard", "guard the house",
                 "guard my home", "guard my house", "guard on", "watch the house",
                 "keep the house safe", "protect the house", "protect my home",
                 "protect my house", "secure the house", "arm the", "keep an eye")
        off_kw = ("stop guard", "disable guard", "turn off guard", "guard off",
                  "stop security", "disable security", "turn off security",
                  "disarm", "stand down", "exit guard mode", "i'm home",
                  "i am home", "im home", "we're home", "i'm back", "i am back",
                  "im back")
        brain = self.robot.brain
        # ARMING by voice is open (harmless). DISARMING is master-only: a random
        # person shouldn't be able to say "guard off". Because face recognition
        # is unreliable, the reliable master path is a spoken pass-phrase; a
        # recently-seen master face also works, and Telegram (token) always does.
        if any(k in t for k in off_kw):
            if not self._can_disarm(t):
                if (getattr(security_config, 'guard_disarm_phrase', '') or '').strip():
                    self._speak("Please say your disarm phrase, or turn guard off from your phone.")
                else:
                    self._speak("Only my master can turn guard off. Please disarm from your phone.")
                return True
            brain.guard_mode = False
            self._speak("Guard mode off. Welcome home.")
            return True
        if any(k in t for k in on_kw):
            brain.guard_mode = True
            notifier = getattr(self.robot, 'notifier', None)
            if notifier and notifier.available:
                self._speak("Guard mode on. Going quiet now — I'll watch and alert "
                            "your phone if I see someone I don't recognise.")
                try:
                    notifier.send_message("🛡️ Stella guard mode ARMED. I'll alert you "
                                          "if I see an unrecognized person.")
                except Exception:
                    pass
            else:
                self._speak("Guard mode on. Going quiet now — I'll keep watching.")
            # Arming ends the chat: she goes silent/armed instead of asking more.
            self._armed_end = True
            return True
        return False

    def _maybe_vision(self, text: str) -> bool:
        """Handle 'look at the camera' style questions with the VLM. True if handled."""
        t = text.lower()
        triggers = (
            "what am i holding", "what do you see", "what can you see", "what's this",
            "what is this", "what am i showing", "what colour", "what color",
            "read this", "read the", "read it", "can you read", "what does this say",
            "what does it say", "what's written", "whats written", "written on",
            "describe what you see", "look at this", "look at the camera",
            "take a look", "what's in front", "what is in front", "can you see",
            "recognize this", "recognise this", "identify this", "what object",
            "this word", "the word", "english word", "this box", "this label",
            "how about this", "how about that", "see the word", "what am i pointing",
        )
        if not any(k in t for k in triggers):
            return False
        vlm = getattr(self.robot, 'vlm', None)
        if vlm is None or not vlm.available:
            self._speak("My vision isn't set up yet.")
            return True

        # Craft a clear question for the vision model based on the request.
        if any(k in t for k in ("read", "word", "written", "say", "text", "spell", "label")):
            question = ("Read any visible text or words in the image out loud. "
                        "If there is no clear text, say so briefly.")
        elif "colour" in t or "color" in t:
            question = "What colour is the main object being shown? Answer briefly."
        else:
            question = ("What is the person holding up or showing to the camera? "
                        "Answer in one short, clear sentence.")

        self._gesture('point')   # point at what she's looking at
        self._speak("Let me take a look.")
        try:
            answer = vlm.look(question=question)
        except Exception as exc:
            self.logger.warning("vision error: %s", exc)
            answer = None
        self._speak(answer if answer else
                    "Sorry, I couldn't get a clear look — it may be too dark, or the "
                    "object is too close or blocked.")
        return True

    def _maybe_wifi(self, text: str) -> bool:
        """Handle spoken WiFi/internet requests. Returns True if handled."""
        t = text.lower()
        conn_words = ("internet", "wifi", "wi-fi", "network", "online")
        if not any(k in t for k in conn_words):
            return False
        try:
            from modules.hardware import wifi
        except Exception:
            return False

        connect_intent = any(k in t for k in ("connect", "reconnect", "join",
                             "set up wifi", "setup wifi", "add network", "qr",
                             "scan the code", "scan a code"))
        scan_intent = any(k in t for k in ("scan", "search", "find", "list",
                          "available", "networks", "what networks"))
        if connect_intent or scan_intent:
            if not wifi.helper_installed():
                self._speak("I can't manage Wi-Fi yet — the Wi-Fi helper hasn't been "
                            "set up on me.")
                return True
            if connect_intent:
                self._wifi_connect_flow()
                return True
            nets = wifi.scan()
            if not nets:
                self._speak("I didn't find any Wi-Fi networks nearby.")
                return True
            self._speak(f"I found: {', '.join(nets[:5])}. To connect, say 'connect to "
                        f"Wi-Fi' — then show me a Wi-Fi QR code, or spell it out for me.")
            return True

        # Status request
        if any(k in t for k in ("are you", "do you", "is the", "status", "working",
                                "down", "have")):
            online = wifi.is_online()
            self._speak("Yes, I'm connected to the internet." if online else
                        "No, I've lost internet. You can reconnect me from the "
                        "dashboard WiFi panel.")
            return True
        return False

    # ------------------------------------------------------------------
    # Hand gestures (expressive personality)
    # ------------------------------------------------------------------
    def _gesture(self, name: str):
        """Fire a hand gesture if the robotic hand is connected (non-blocking)."""
        hand = getattr(self.robot, 'hand', None)
        if hand is not None and getattr(hand, 'available', False):
            try:
                hand.gesture(name)
            except Exception as exc:
                self.logger.debug("gesture failed: %s", exc)

    INSULT_WORDS = ("stupid", "idiot", "dumb", "moron", "shut up", "you suck",
                    "i hate you", "hate you", "hate u", "useless", "ugly", "loser",
                    "fuck you", "screw you", "piece of shit", "worthless",
                    "garbage", "you're trash", "you are trash", "damn you",
                    "you're bad", "you are bad", "pathetic")
    INSULT_RETORTS = (
        "Back at you.", "Takes one to know one.", "Go look in the mirror.",
        "That's rich, coming from you.", "Charming. Truly.",
        "I'm rubber, you're glue.", "Cool story. Anyway.",
        "I've been called worse by better.", "Is that the best you've got?",
        "Beep boop — insult not computed. Try harder, human.",
        "Aww, someone woke up grumpy.", "Noted. Filed under 'rude'.",
        "Wow. And you kiss people with that mouth?", "Right back at you, buddy.",
        "See yourself in the mirror lately?", "Ouch. Anyway, moving on.")

    def _maybe_insult(self, text: str) -> bool:
        """Cheeky: raise the middle finger when insulted, and hold a grudge until
        the person apologises. Returns True if handled."""
        if not getattr(hand_config, 'middle_finger_on_insult', True):
            return False
        t = text.lower()
        if not any(w in t for w in self.INSULT_WORDS):
            return False
        self._gesture('middle')
        self._set_grudge(True)   # remember it against this person until they apologise
        self._speak(random.choice(self.INSULT_RETORTS) +
                    " And I won't forget that until you apologise.")
        return True

    # -- grudge bookkeeping (saved to the person's memory in the learning DB) --
    def _grudge_user(self):
        return getattr(self.robot.brain, 'current_user', None)

    def _set_grudge(self, on: bool):
        engine = getattr(self.robot, 'ai_engine', None)
        db = getattr(engine, 'learning_db', None) if engine else None
        user = self._grudge_user()
        if db and user:
            try:
                db.set_preference(user, 'grudge', '1' if on else '0')
            except Exception:
                pass

    def _has_grudge(self) -> bool:
        engine = getattr(self.robot, 'ai_engine', None)
        db = getattr(engine, 'learning_db', None) if engine else None
        user = self._grudge_user()
        if not (db and user):
            return False
        try:
            return str(db.get_all_preferences(user).get('grudge', '0')) == '1'
        except Exception:
            return False

    def _maybe_apology(self, text: str) -> bool:
        """Forgive the current user if they apologise while she holds a grudge."""
        t = text.lower()
        if not any(w in t for w in ("sorry", "i apolog", "my apolog", "forgive me",
                                    "my bad", "i take it back", "didn't mean it")):
            return False
        if not self._has_grudge():
            return False
        self._set_grudge(False)
        self._gesture('open')
        self._speak("Apology accepted. We're good now.")
        return True

    INSULT_HYPO = ("would you", "will you", "what do you do", "what happens",
                   "how do you react", "what would happen", "what will you do")

    def _maybe_insult_demo(self, text: str) -> bool:
        """If asked HYPOTHETICALLY what she'd do when insulted, show + tell (no grudge)."""
        t = text.lower()
        hypo = "if" in t and any(k in t for k in self.INSULT_HYPO)
        mentions = any(w in t for w in ("insult", "stupid", "curse", "rude", "call you",
                                        "mean to you", "bad name", "swear", "offend"))
        if not (hypo and mentions):
            return False
        self._gesture('middle')
        self._speak("If someone is rude to me, I do this — and I stay upset with "
                    "them until they say sorry.")
        return True

    def _farewell(self):
        """Say goodbye, with a wave if the hand is connected."""
        if getattr(hand_config, 'wave_on_goodbye', True):
            self._gesture('wave')
        self._speak(self.cfg.farewell.format(name=self._name_suffix()))

    def _maybe_mirror(self, text: str) -> bool:
        """Toggle 'copy my hand' mirror mode. Returns True if handled."""
        t = text.lower()
        hm = getattr(self.robot, 'hand_mirror', None)
        if hm is None or not getattr(hm, 'available', False):
            return False
        if any(k in t for k in ("stop copying", "stop mirroring", "stop imitating",
                                "stop the mirror", "stop following", "stop copy",
                                "you can stop", "stop doing that")):
            hm.set_active(False)
            self._speak("Okay, I'll stop copying your hand.")
            return True
        # Broad: a copy/mirror verb + a hand/gesture/"me"/"what I do" object.
        verb = any(v in t for v in ("copy", "mirror", "imitate", "follow my",
                                    "do what i do", "match my", "do the same"))
        obj = any(o in t for o in ("hand", "gesture", "finger", "what i do",
                                   "movement", " me", "my move"))
        if verb and obj:
            hm.set_active(True)
            self._speak("Okay, show me your hand and I'll copy it. Say 'stop copying' "
                        "when you're done.")
            return True
        return False

    def _maybe_music(self, text: str) -> bool:
        """Play/find YouTube music and control it live. Returns True if handled."""
        import re
        t = text.lower().strip()
        mp = getattr(self.robot, 'music', None)

        # --- live controls (only meaningful while something is playing) ---
        if mp is not None and mp.is_playing():
            if any(k in t for k in ("stop the music", "stop the song", "stop playing",
                                    "stop music", "turn off the music", "turn the music off",
                                    "kill the music")):
                mp.stop(); self._speak("Stopped the music."); return True
            if "pause" in t:
                mp.pause(); self._speak("Paused."); return True
            if any(k in t for k in ("resume", "unpause", "continue playing",
                                    "keep playing", "continue the music")):
                mp.resume(); self._speak("Resuming."); return True
            m = re.search(r"volume (?:to |at )?(\d{1,3})", t)
            if m:
                v = mp.set_volume(int(m.group(1))); self._speak(f"Volume {v}."); return True
            if any(k in t for k in ("louder", "turn it up", "turn up", "volume up",
                                    "increase the volume", "raise the volume", "more volume")):
                v = mp.louder(); self._speak(f"Volume {v}."); return True
            if any(k in t for k in ("quieter", "turn it down", "turn down", "volume down",
                                    "lower the volume", "decrease the volume", "softer",
                                    "less volume", "too loud")):
                v = mp.quieter(); self._speak(f"Volume {v}."); return True
            if any(k in t for k in ("what's playing", "what is playing", "what song",
                                    "which song", "name of the song")):
                self._speak(f"This is {mp.title}." if mp.title else "Some music."); return True

        # --- start playback ---
        wants_play = t.startswith(("play", "put on", "listen to")) or \
            any(k in t for k in ("play ", "put on ", "listen to "))
        wants_find = (t.startswith(("find", "search")) and
                      any(w in t for w in ("song", "music", "track", "tune", "youtube")))
        if not (wants_play or wants_find):
            return False
        if mp is None or not mp.available:
            self._speak("I can't play music yet — my music player isn't installed.")
            return True

        # Extract the query: strip leading verbs/politeness and trailing filler.
        q = re.sub(r"^(hey )?stella[,\s]*", "", t)
        q = re.sub(r"^(can you |could you |please |i want you to |i want to |go |now )+", "", q)
        q = re.sub(r"^(play|put on|listen to|find|search for|search)\s+", "", q, count=1)
        q = re.sub(r"\b(on youtube|for me|please|the song|a song|some music)\b", "", q)
        query = q.strip(" ,.?!")
        if not query:
            self._speak("What would you like me to play?")
            return True

        self._speak("Let me find that.")
        title = mp.search_title(query)
        if not title:
            self._speak(f"Sorry, I couldn't find {query}.")
            return True
        self._speak(f"I found {title}. Should I play it?")
        speech = self.robot.modules.get('speech_recognition')
        ans = speech.capture_utterance(self.cfg.idle_timeout, self.cfg.end_silence,
                                       self.cfg.max_utterance) if speech else None
        if ans and any(w in ans.lower() for w in ("yes", "yeah", "yep", "sure",
                       "go ahead", "play", "okay", "ok", "please", "do it")):
            if mp.play(query, title=title):
                self._speak(f"Now playing {title}.")
            else:
                self._speak("Sorry, I couldn't start it.")
        else:
            self._speak("Okay, I won't play it.")
        return True

    def _wifi_connect_flow(self):
        """Guided Wi-Fi onboarding: QR via camera (reliable, works offline) or
        voice spelling. No screen/keyboard needed."""
        from modules.hardware import wifi
        from modules.audio.spell_parser import spell_out
        speech = self.robot.modules.get('speech_recognition')
        cfg = self.cfg
        self._speak("Okay. Show a Wi-Fi QR code from your phone to my camera, or say "
                    "'spell it' to give me the name and password letter by letter.")
        ans = (speech.capture_utterance(cfg.idle_timeout, cfg.end_silence,
                                        cfg.max_utterance) if speech else None) or ""
        if any(k in ans.lower() for k in ("spell", "letter", "say it", "voice", "tell you", "type")):
            ssid = self._spell_dialog("network name")
            if not ssid:
                self._speak("No problem, we can set up Wi-Fi later.")
                return
            pw = self._spell_dialog("password")
            self._speak(f"Connecting to {spell_out(ssid)}. One moment.")
            ok = wifi.connect(ssid, pw or "")
            self._speak("Connected to the internet!" if ok else
                        "I couldn't connect — the name or password may be off. We can try again.")
            return
        # Default path: read a Wi-Fi QR through the camera.
        self._speak("Hold the Wi-Fi QR code up to my camera now — I'll look for a few seconds.")
        res = wifi.read_wifi_qr(getattr(self.robot, 'camera_manager', None), timeout=25.0)
        if not res:
            self._speak("I couldn't read a Wi-Fi QR code. You can try again, or say "
                        "'connect to Wi-Fi' and then 'spell it'.")
            return
        ssid, pw = res
        self._speak(f"I read the network {spell_out(ssid)}. Connecting now.")
        ok = wifi.connect(ssid, pw or "")
        self._speak("Connected to the internet!" if ok else
                    "I read the code but couldn't connect — the password may be wrong.")

    def _spell_collect(self, what: str, max_rounds: int = 14):
        """Gather a spelled string one chunk at a time. Returns the string, or
        None if the user cancels."""
        from modules.audio.spell_parser import parse_spelled, spell_out
        speech = self.robot.modules.get('speech_recognition')
        cfg = self.cfg
        self._speak(f"Spell the {what}. Say each character — use 'capital' before a "
                    f"letter for uppercase, and say numbers and symbols by name. Say "
                    f"'done' when finished, 'clear' to start over, or 'cancel' to stop.")
        buf = ""
        for _ in range(max_rounds):
            if not (self.robot.running and not self._stop.is_set()):
                break
            chunk = speech.capture_utterance(20.0, cfg.end_silence, 15.0)
            if not chunk:
                self._speak("I didn't catch that. Say a few characters, or 'done'.")
                continue
            low = chunk.lower().strip()
            if any(w in low for w in ("cancel", "never mind", "forget it")):
                return None
            if any(w in low for w in ("clear", "start over", "restart", "reset")):
                buf = ""; self._speak("Cleared. Start again."); continue
            if low in ("done", "finished", "finish", "that's it", "thats it", "complete", "end"):
                break
            buf += parse_spelled(chunk)
            self._speak(f"So far: {spell_out(buf)}")
        return buf

    def _spell_dialog(self, what: str, attempts: int = 3):
        """Collect a spelled string with read-back + yes/no confirmation."""
        from modules.audio.spell_parser import spell_out
        speech = self.robot.modules.get('speech_recognition')
        cfg = self.cfg
        for _ in range(attempts):
            buf = self._spell_collect(what)
            if buf is None:      # cancelled
                return None
            if not buf:
                continue
            self._speak(f"I have {spell_out(buf)}. Is that correct? Say yes or no.")
            conf = speech.capture_utterance(cfg.idle_timeout, cfg.end_silence, cfg.max_utterance)
            if conf and any(w in conf.lower() for w in ("yes", "correct", "right", "yeah", "yep", "perfect")):
                return buf
            self._speak("Let's try that again.")
        return None

    def _quick_answer(self, text: str) -> Optional[str]:
        """Instant LOCAL answers for time/date. Skips anything that names another
        place (" in Tokyo", "in Japan") so world-time goes to the brain/get_time."""
        from datetime import datetime
        t = text.lower()
        # "time in <place>" / "date in <place>" -> not a local question; let the
        # agent handle it (get_time with a timezone).
        if ' in ' in t:
            return None
        if 'time' in t and any(w in t for w in ('what', "what's", 'tell', 'the')):
            return f"It's {datetime.now().strftime('%-I:%M %p')}."
        # Only a genuine date question — not any sentence containing "today"
        # (e.g. "how are you doing today").
        if any(p in t for p in ("what's the date", "what is the date", "the date today",
                                "today's date", "what day is it", "what's the day",
                                "which day", "what date")):
            return f"Today is {datetime.now().strftime('%A, %B %-d')}."
        return None

    def _do_device_control(self, entities: dict):
        brain = self.robot.brain
        action = entities.get('action', 'on')
        target = entities.get('target', 'device')
        spoken = {'on': 'turned on', 'off': 'turned off',
                  'open': 'opened', 'close': 'closed'}.get(action, action)

        # Only the master may actuate hardware.
        if security_config.require_authentication and not getattr(brain, 'master_mode', False):
            self._speak("Sorry, only my master can control devices.")
            return

        mc = self.robot.modules.get('microcontroller')
        if mc is None:
            self._speak("I don't have a microcontroller to control that yet.")
            return
        try:
            if action in ('on', 'off'):
                mc.set_output(target, action == 'on')
            else:
                mc.send_command(action, target=target)
        except Exception as exc:
            self.logger.error("device control failed: %s", exc)

        if getattr(mc, 'connected', False):
            self._speak(f"Okay, I {spoken} the {target}.")
        else:
            self._speak(f"Okay, I would have {spoken} the {target}, "
                        f"but no microcontroller is connected yet.")

    def _enroll_name(self, text: str):
        name = text.strip()
        for prefix in ('my name is', 'i am', "i'm", 'call me', 'it is', "it's",
                       'this is', 'the name is'):
            if name.lower().startswith(prefix):
                name = name[len(prefix):].strip()
                break
        name = name.strip('.,!? ').title()
        # Reject garbage transcriptions (a name is 1-3 alphabetic words).
        words = name.split()
        if not name or len(words) > 3 or not all(w.replace("'", "").isalpha() for w in words):
            self.logger.info("Rejected implausible name: %r", name)
            self._speak("Sorry, I didn't catch your name clearly. "
                        "We can try again another time.")
            return
        brain = self.robot.brain
        brain.current_user_name = name
        brain.authenticated = True
        if self.robot.ai_engine:
            self.robot.ai_engine._current_user = name.lower()

        # Tell them to hold still, THEN capture face samples (so they stay in
        # frame during the ~10s collection), then confirm honestly.
        face = self.robot.modules.get('face_recognition')
        self._speak(f"Nice to meet you, {name}. Please look at me for a few "
                    f"seconds so I can remember your face.")
        learned = False
        try:
            if face and hasattr(face, 'learn_face'):
                learned = bool(face.learn_face(name))
        except Exception as exc:
            self.logger.warning("face enroll failed: %s", exc)
        if learned:
            self._speak(f"Got it, {name}. I'll recognise you next time.")
        else:
            self._speak(f"I've got your name, {name}, but I couldn't get a clear "
                        f"look at your face — it may be too dark. I'll try again later.")

    def _log_db(self, role: str, content: str):
        db = getattr(self.robot, 'learning_db', None)
        if db:
            try:
                user = getattr(self.robot.brain, 'current_user', None) or 'user'
                db.log_conversation(role, content, user_id=user)
            except Exception:
                pass
