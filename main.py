"""
AI Robot Main Entry Point
=========================
Main application that initializes all modules and runs the robot system.
Handles module coordination, error recovery, and graceful shutdown.

Target platform: Raspberry Pi 5 + Hailo AI Accelerator + Ubuntu Server 24.04
"""

import sys
import signal
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import os
import psutil
import threading

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def _silence_alsa_warnings():
    """Silence the harmless 'ALSA lib ...' C-library chatter on the Pi.

    ALSA prints warnings to stderr for every PCM it can't open while PyAudio
    enumerates devices. They're noise, not errors. We install a no-op error
    handler at the C level. Kept in a module global so it isn't garbage
    collected."""
    global _ALSA_ERR_HANDLER
    try:
        from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int
        proto = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        _ALSA_ERR_HANDLER = proto(lambda *a: None)
        CDLL("libasound.so.2").snd_lib_error_set_handler(_ALSA_ERR_HANDLER)
    except Exception:
        pass


_ALSA_ERR_HANDLER = None
_silence_alsa_warnings()

# Load .env file if present (API keys, etc.)
_env_file = PROJECT_ROOT / '.env'
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _, _val = _line.partition('=')
                os.environ.setdefault(_key.strip(), _val.strip())

# Import configuration — platform detection and Hailo availability are
# handled inside config.settings automatically.
from config.settings import (
    config,
    system_config,
    hardware_config,
    security_config,
    behavior_config,
    ai_config,
    web_search_config,
    music_config,
    hand_config,
    microcontroller_config,
    navigation_config,
)

# Import core modules
from core.robot_brain import RobotBrain, RobotEvent

# Shared hardware managers
from modules.hardware.camera_manager import CameraManager
from modules.hardware.audio_manager import AudioManager

# Vision modules
from modules.vision.face_recognition import FaceRecognitionModule
from modules.vision.object_detection import ObjectDetectionModule
from modules.vision.vlm import VLM  # cloud vision: "what am I holding?"

# Audio modules
from modules.audio.wake_word import WakeWordModule
from modules.audio.speech_recognition import SpeechRecognitionModule
from modules.audio.text_to_speech import TextToSpeechModule

# AI modules
from modules.ai.ai_engine import AIEngine
from modules.ai.learning_db import LearningDB
from modules.ai.reminders import ReminderManager

# Hardware controllers
from modules.hardware.microcontroller import MicrocontrollerController

# Navigation / environment learning (future wheels + sensors)
from modules.navigation.navigator import Navigator

# Conversation session manager (wake -> multi-turn dialogue -> farewell)
from modules.conversation.manager import ConversationManager

# Web dashboard
from modules.web.dashboard import WebDashboard


class AIRobot:
    """
    Main AI Robot application class.
    Manages all modules and coordinates the robot system.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the AI Robot
        
        Args:
            config_file: Optional path to configuration file
        """
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("=" * 60)
        self.logger.info(f"   {behavior_config.robot_name} AI ROBOT SYSTEM")
        self.logger.info("=" * 60)
        
        # Load custom configuration if provided, else prefer config.yaml
        # (the editable control panel), falling back to legacy config.json.
        yaml_config = PROJECT_ROOT / "config" / "config.yaml"
        json_config = PROJECT_ROOT / "config" / "config.json"
        if config_file:
            config.load_from_file(config_file)
            self.logger.info(f"Loaded settings from {config_file}")
        elif yaml_config.exists():
            config.load_from_file(str(yaml_config))
            self.logger.info("Loaded settings from config/config.yaml")
        elif json_config.exists():
            config.load_from_file(str(json_config))
            self.logger.info("Loaded settings from config/config.json")
        
        # Validate configuration
        if not config.validate():
            self.logger.error("Configuration validation failed!")
            sys.exit(1)
        
        # System state
        self.running = False
        self.modules: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        
        # Shared hardware managers
        self.camera_manager = CameraManager()
        self.audio_manager = AudioManager()
        
        # AI subsystems
        self.ai_engine = AIEngine()
        self.learning_db = LearningDB()
        
        # Initialize robot brain
        self.brain = RobotBrain()
        
        # Performance monitoring
        self.start_time = time.time()
        self.performance_monitor = None
        
        # Error handling
        self.error_count = 0
        self.max_errors = 10
        
        # Shutdown event
        self.shutdown_event = threading.Event()
        self.keyword_listener_paused = False
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Console handler with color
        try:
            import colorlog
            console_handler = colorlog.StreamHandler()
            console_handler.setFormatter(
                colorlog.ColoredFormatter(
                    '%(log_color)s' + log_format,
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'bold_red'
                    }
                )
            )
        except ImportError:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
        
        # File handler — FIXED filename so it's always easy to find/tail.
        # Rotates so it never grows unbounded (keeps a few old runs).
        from logging.handlers import RotatingFileHandler
        log_dir = PROJECT_ROOT / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "gonzo.log", maxBytes=2_000_000, backupCount=3,
        )
        file_handler.setFormatter(logging.Formatter(log_format))

        # Configure root logger (force=True needed because imports may have
        # already called logging.warning(), which auto-adds a default handler
        # and makes basicConfig() a no-op without force)
        root = logging.getLogger()
        root.setLevel(getattr(logging, system_config.log_level))
        # Remove any auto-added handlers
        root.handlers.clear()
        root.addHandler(console_handler)
        root.addHandler(file_handler)

        # Silence noisy third-party loggers that flood the log with request
        # spam (Flask access logs, HTTP client, DuckDuckGo search internals).
        for noisy in ('werkzeug', 'httpx', 'httpcore', 'ddgs', 'primp',
                      'urllib3', 'PIL', 'transitions.core'):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # Mark a clear start-of-run boundary in the log.
        logging.getLogger('AIRobot').info(
            "===== NEW RUN %s =====", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    def initialize_modules(self):
        """Initialize all robot modules with shared hardware managers."""
        self.logger.info("Initializing modules...")
        
        # --- Wire AI subsystems to brain ---
        self.brain.ai_engine = self.ai_engine
        self.brain.learning_db = self.learning_db
        
        # Give AI engine access to the learning DB for memory recall
        self.ai_engine.learning_db = self.learning_db
        
        # Start persistent systems first
        self.learning_db.start()
        self.ai_engine.start()
        # Warm up the NPU model in the background so the first question is fast.
        threading.Thread(target=self.ai_engine.warmup, daemon=True).start()
        
        # --- Start shared Camera Manager ---
        try:
            self.logger.info("Starting shared Camera Manager...")
            self.camera_manager.start()
            self.logger.info("✓ Camera Manager started")
        except Exception as e:
            self.logger.error(f"✗ Camera Manager failed: {e}")
        
        # --- Vision: Face Recognition (subscribes to shared camera) ---
        try:
            self.logger.info("Initializing Face Recognition...")
            self.modules['face_recognition'] = FaceRecognitionModule(
                brain=self.brain,
                camera_manager=self.camera_manager,
            )
            self.brain.modules['vision'] = self.modules['face_recognition']
            self.logger.info("✓ Face Recognition initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Face Recognition: {e}")
        
        # --- Vision: Object Detection (subscribes to shared camera) ---
        try:
            self.logger.info("Initializing Object Detection (Hailo / OpenCV fallback)...")
            self.modules['object_detection'] = ObjectDetectionModule(
                camera_manager=self.camera_manager,
                brain=self.brain,
            )
            self.brain.modules['object_detection'] = self.modules['object_detection']
            self.logger.info("✓ Object Detection initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Object Detection: {e}")
        
        # --- Audio Modules ---
        try:
            self.logger.info("Initializing Wake Word Detection...")
            self.modules['wake_word'] = WakeWordModule(self.brain)
            self.logger.info("✓ Wake Word Detection initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Wake Word: {e}")
        
        try:
            self.logger.info("Initializing Speech Recognition...")
            self.modules['speech_recognition'] = SpeechRecognitionModule(self.brain)
            self.brain.modules['speech'] = self.modules['speech_recognition']
            self.logger.info("✓ Speech Recognition initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Speech Recognition: {e}")
        
        try:
            self.logger.info("Initializing Text-to-Speech...")
            self.modules['tts'] = TextToSpeechModule(self.brain)
            self.brain.modules['audio'] = self.modules['tts']
            self.brain.modules['tts'] = self.modules['tts']
            self.logger.info("✓ Text-to-Speech initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize TTS: {e}")
        
        # --- Microcontroller bridge (ESP32 / Pi Zero / Arduino) ---
        # Always present as brain.modules['hardware'] so real-world commands
        # ("turn on the light") always have a sink. When
        # microcontroller.connected is false it runs in log-only mode.
        try:
            self.logger.info("Initializing Microcontroller bridge...")
            self.modules['microcontroller'] = MicrocontrollerController(microcontroller_config)
            self.modules['microcontroller'].connect()
            self.brain.modules['hardware'] = self.modules['microcontroller']
            if microcontroller_config.connected:
                self.logger.info(
                    "✓ Microcontroller bridge live (%s transport)",
                    microcontroller_config.transport,
                )
            else:
                self.logger.info("✓ Microcontroller bridge in log-only mode (not connected)")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Microcontroller bridge: {e}")

        # --- Navigation / environment learning (future wheels + sensors) ---
        try:
            self.modules['navigation'] = Navigator(
                microcontroller=self.modules.get('microcontroller'),
                cfg=navigation_config,
                brain=self.brain,
            )
            self.modules['navigation'].start()
            self.brain.modules['navigation'] = self.modules['navigation']
            if navigation_config.enabled:
                self.logger.info("✓ Navigation enabled")
            else:
                self.logger.info("✓ Navigation hooks ready (disabled in config)")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Navigation: {e}")

        # --- Phone notifications (Telegram) ---
        from modules.hardware.notify import Notifier
        self.notifier = Notifier()
        self._last_guard_alert = 0.0
        self._unknown_streak = 0
        self._guard_pending = None   # interactive guard: 'recognize' | 'alarm' | None
        self.logger.info("Notifier: %s",
                         "ready (Telegram)" if self.notifier.available
                         else "off (set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID in .env)")

        # --- Motion / presence guard (alerts on movement or a body, NO face
        #     needed). Face recognition only suppresses alerts when it sees you.
        self.motion_guard = None
        try:
            from modules.vision.motion_guard import MotionGuard
            self.motion_guard = MotionGuard(
                self.camera_manager, self.brain, self._on_guard_motion,
                enabled=getattr(security_config, 'guard_motion_detection', True),
                sensitivity=getattr(security_config, 'guard_sensitivity', 'medium'),
                cooldown=getattr(security_config, 'guard_alert_cooldown', 30.0),
                detect_person=getattr(security_config, 'guard_detect_person', True),
            )
            self.motion_guard.start()
            self.logger.info("✓ Motion guard ready (sensitivity=%s)",
                             getattr(security_config, 'guard_sensitivity', 'medium'))
        except Exception as e:
            self.logger.error("✗ Motion guard init failed: %s", e)

        # --- Reminders / timers ---
        self.reminders = ReminderManager(
            speak_cb=self._announce,
            store_path=str(PROJECT_ROOT / "data" / "reminders.json"))

        # --- Vision-language model (cloud, optional) ---
        self.vlm = VLM(camera_manager=self.camera_manager)
        if self.vlm.available:
            self.logger.info("✓ Vision ready (Moondream) — ask 'what am I holding?'")
        else:
            self.logger.info("Vision off (set MOONDREAM_API_KEY in .env to enable)")

        # --- Music player (YouTube via mpv + yt-dlp), routed to Stella's speaker ---
        self.music = None
        try:
            from modules.hardware.music import MusicPlayer
            tts_mod = self.modules.get('tts')
            self.music = MusicPlayer(
                alsa_device=getattr(tts_mod, 'alsa_device', None),
                ytdlp_path=str(PROJECT_ROOT / 'venv' / 'bin' / 'yt-dlp'),
                default_volume=getattr(music_config, 'default_volume', 70),
                duck_volume=getattr(music_config, 'duck_volume', 25),
                volume_step=getattr(music_config, 'volume_step', 20))
            self.logger.info("✓ Music player ready" if self.music.available
                             else "Music off (needs ffmpeg + aplay + yt-dlp)")
            # Let the TTS free the speaker from the music while Stella speaks.
            if tts_mod is not None:
                tts_mod.music = self.music
        except Exception as e:
            self.logger.error("✗ Music player init failed: %s", e)

        # --- Audio arbiter: one owner of the speaker at a time (speech preempts
        #     music, and future alert chimes go through the same coordinator). ---
        try:
            from modules.hardware.audio_arbiter import AudioArbiter
            self.audio_arbiter = AudioArbiter()
            if self.music is not None:
                self.audio_arbiter.add_duckable(self.music)
            if tts_mod is not None:
                tts_mod.arbiter = self.audio_arbiter
            self.logger.info("✓ Audio arbiter ready")
        except Exception as e:
            self.logger.error("✗ Audio arbiter init failed: %s", e)
            self.audio_arbiter = None

        # --- Robotic hand (ESP32 + PCA9685 gestures over serial) ---
        self.hand = None
        try:
            if getattr(hand_config, 'enabled', False):
                from modules.hardware.hand import Hand
                self.hand = Hand(
                    port=getattr(hand_config, 'serial_port', '/dev/ttyUSB0'),
                    baud=getattr(hand_config, 'baud', 115200),
                    enabled=True)
                self.logger.info("✓ Robotic hand ready" if self.hand.available
                                 else "Hand enabled but serial not connected")
            else:
                self.logger.info("Robotic hand disabled (set hand.enabled: true)")
        except Exception as e:
            self.logger.error("✗ Hand init failed: %s", e)

        # --- Hand mirror (MediaPipe): Stella copies your hand when asked ---
        self.hand_mirror = None
        try:
            if self.hand is not None and getattr(self.hand, 'available', False):
                from modules.vision.hand_mirror import HandMirror
                self.hand_mirror = HandMirror(self.camera_manager, self.hand)
                self.hand_mirror.start()
                self.logger.info("✓ Hand mirror ready" if self.hand_mirror.available
                                 else "Hand mirror off (mediapipe not available)")
        except Exception as e:
            self.logger.error("✗ Hand mirror init failed: %s", e)

        # --- Conversation session manager ---
        self.conversation = ConversationManager(self)
        self.logger.info("✓ Conversation manager ready")

        # --- Two-way Telegram chat (text Stella from your phone) ---
        self._remote_master = False
        try:
            from modules.comms.telegram_bridge import TelegramBridge
            self.telegram = TelegramBridge(self)
            if self.telegram.available:
                self.logger.info("✓ Telegram two-way chat ready")
            else:
                self.telegram = None
        except Exception as e:
            self.logger.error(f"Telegram bridge init failed: {e}")
            self.telegram = None
        
        # --- Web Dashboard (only when remote access is explicitly enabled) ---
        self.dashboard = None
        if security_config.remote_access_enabled:
            try:
                self.logger.info("Starting Web Dashboard...")
                self.dashboard = WebDashboard(
                    camera_manager=self.camera_manager,
                    brain=self.brain,
                    learning_db=self.learning_db,
                    robot_name=behavior_config.robot_name,
                    ai_engine=self.ai_engine,
                )
                self.dashboard.start()
                self.logger.info("✓ Web Dashboard at http://0.0.0.0:5000")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Web Dashboard: {e}")
                self.dashboard = None
        else:
            self.logger.info(
                "Web Dashboard disabled (set security.remote_access_enabled: true to enable)"
            )
        
        self.logger.info(f"Initialized {len(self.modules)} modules")
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register event handlers for inter-module communication"""
        
        # Face detection events
        self.brain.register_event_handler(
            'face_detected',
            self._handle_face_detected
        )
        
        # Object detection events
        self.brain.register_event_handler(
            'object_detected',
            self._handle_object_detected
        )
        
        # Wake word events
        self.brain.register_event_handler(
            'wake_word_detected',
            self._handle_wake_word
        )
        
        # Speech events
        self.brain.register_event_handler(
            'speech_recognized',
            self._handle_speech
        )
        
        self.brain.register_event_handler(
            'dialogue_idle',
            self._handle_dialogue_idle
        )
        
        self.brain.register_event_handler(
            'speech_complete',
            self._handle_speech_complete
        )
        
        self.brain.register_event_handler(
            'speech_listen_failed',
            self._handle_speech_listen_failed
        )
        
        # System events
        self.brain.register_event_handler(
            'battery_low',
            self._handle_battery_low
        )
        
        self.brain.register_event_handler(
            'emergency_stop',
            self._handle_emergency_stop
        )
        
        # Speak events — route to TTS
        self.brain.register_event_handler(
            'speak',
            self._handle_speak
        )
    
    def _handle_face_detected(self, event: RobotEvent):
        """Handle face detection event"""
        face_data = event.data
        face_id = face_data.get('face_id')
        self.logger.debug(f"Face detected: {face_data.get('name', 'Unknown')}")

        # Authenticate if master
        if face_data.get('is_master'):
            self._unknown_streak = 0
            # Coming home while armed -> recognise you and auto-disarm + greet,
            # so you never have to fight the "only my master can disarm" wall.
            if getattr(self.brain, 'guard_mode', False):
                self.brain.guard_mode = False
                self.logger.info("Master recognised while armed — auto-disarming guard")
                self._announce("Welcome home. Guard mode is now off.")
                if getattr(self, 'notifier', None) and self.notifier.available:
                    threading.Thread(
                        target=self.notifier.send_message,
                        args=("✅ Welcome home — I recognised you, guard disarmed.",),
                        daemon=True).start()
            self.brain.emit_event(RobotEvent(
                type='user_authenticated',
                source='main',
                data={
                    'user_id': face_data['face_id'],
                    'method': 'face'
                },
                priority=2
            ))
            return

        # Reset the phantom-face streak whenever a real known face appears.
        if face_id and face_id != 'unknown':
            self._unknown_streak = 0

        # GUARD MODE: an unrecognized face triggers a silent snapshot + phone
        # alert. Filter flicker/phantoms (e.g. a face in a painting): require the
        # unknown to persist a few detections, and skip if the master was just seen.
        if face_id == 'unknown' and getattr(self.brain, 'guard_mode', False):
            now = time.time()
            if now - getattr(self.brain, 'last_master_time', 0) < 20:
                return  # master is around — not an intruder
            self._unknown_streak = getattr(self, '_unknown_streak', 0) + 1
            if self._unknown_streak >= 3 and now - self._last_guard_alert > 30:
                threading.Thread(target=self._guard_alert,
                                 kwargs={"reason": "an unrecognized face"},
                                 daemon=True).start()
            return

        # Unknown person -> proactively start a conversation so we actually
        # LISTEN (command mic) to their name and enroll them. A passive spoken
        # prompt alone has no listener in this architecture.
        if face_id == 'unknown' and behavior_config.learn_new_faces:
            conv = getattr(self, 'conversation', None)
            if conv is not None and not conv.active:
                now = time.time()
                if now - getattr(self, '_last_unknown_session', 0) > 60:
                    self._last_unknown_session = now
                    self.logger.info("Unknown face — starting greet/enroll conversation")
                    conv.start_session()

    def _on_guard_motion(self, reason: str, frame=None):
        """Motion guard callback — alert about movement/a body (no face needed)."""
        threading.Thread(target=self._guard_alert,
                          kwargs={"reason": reason, "frame": frame},
                          daemon=True).start()

    def _guard_alert(self, reason: str = "an unrecognized person", frame=None):
        """Capture a snapshot and push a guard alert to the master's phone.

        Shared by the motion guard and the face path; one cooldown throttles
        both so a person moving in front of the camera can't spam alerts.
        """
        import time as _t
        now = _t.time()
        if now - self._last_guard_alert < 30:
            return
        self._last_guard_alert = now
        stamp = _t.strftime("%H:%M:%S")
        self.logger.warning("GUARD: %s detected at %s", reason, stamp)
        if frame is None:
            cm = getattr(self, 'camera_manager', None)
            if cm is not None:
                try:
                    f = cm.get_latest_frame()
                    frame = getattr(f, 'image', f)
                except Exception:
                    frame = None
        caption = (f"⚠️ Guard: {reason} detected at {stamp}.\n"
                   f"Do you recognise this person? Reply YES or NO.")
        sent = False
        if getattr(self, 'notifier', None) and self.notifier.available:
            if frame is not None:
                sent = self.notifier.send_photo(frame, caption)
            if not sent:
                sent = self.notifier.send_message(caption)
        if sent:
            self._guard_pending = 'recognize'   # await the master's YES/NO on Telegram
            self._guard_pending_time = now
        else:
            self.logger.warning("GUARD alert not sent (Telegram not configured?)")

    def _guard_scream(self):
        """Sound a LOUD, aggressive alarm (master authorised it from the phone).

        Uses espeak-ng as a harsh shouting voice (not Stella's calm voice),
        amplified/distorted via sox and maxed on the output, repeated — meant to
        scare an intruder, not to sound polite.
        """
        import subprocess, tempfile, os
        self.logger.warning("GUARD: alarm authorised — SCREAMING")
        # stop music and push output volume to max where a control exists
        try:
            if getattr(self, 'music', None):
                self.music.stop()
        except Exception:
            pass
        for ctrl in ('Master', 'PCM', 'Speaker'):
            try:
                subprocess.run(['amixer', 'sset', ctrl, '100%'], capture_output=True, timeout=3)
            except Exception:
                pass
        tts = self.modules.get('tts')
        dev = getattr(tts, 'alsa_device', None) if tts else None
        phrases = ["THIEF! GET OUT NOW!",
                   "INTRUDER! LEAVE IMMEDIATELY!",
                   "GET AWAY, OR I AM CALLING THE POLICE!",
                   "GET OUT! GET OUT NOW!"]
        for i in range(5):
            text = phrases[i % len(phrases)]
            wav = loud = None
            try:
                wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                # -a 200 max amplitude, fast, low/angry pitch
                subprocess.run(['espeak-ng', '-a', '200', '-s', '175', '-p', '15',
                                '-w', wav, text], capture_output=True, timeout=10)
                loud = wav + '.loud.wav'
                r = subprocess.run(['sox', wav, loud, 'gain', '-n', '-0.1', 'vol', '4.0'],
                                   capture_output=True, timeout=10)  # normalize + overdrive
                play = loud if (r.returncode == 0 and os.path.exists(loud)) else wav
                # Try the pinned device, then ALSA default, then plain — HDMI can
                # report a spurious "busy", so fall through until one plays.
                for d in (dev, 'default', None):
                    c = ['aplay', '-q'] + (['-D', d] if d else []) + [play]
                    if subprocess.run(c, capture_output=True, timeout=12).returncode == 0:
                        break
            except Exception as exc:
                self.logger.warning("scream failed: %s", exc)
                break
            finally:
                for f in (wav, loud):
                    if f:
                        try:
                            os.unlink(f)
                        except Exception:
                            pass
    
    def _handle_speak(self, event: RobotEvent):
        """Route speak events to the TTS module."""
        text = event.data.get('text', '') if event.data else ''
        if not text:
            return
        # Print the conversation to the console so it's visible even without a
        # speaker (e.g. on the monitor's text console or over SSH).
        print(f"\n{behavior_config.robot_name}: {text}\n", flush=True)
        tts = self.modules.get('tts')
        if tts and tts.running:
            tts.speak(text)
        else:
            self.logger.info(f"[SPEAK] {text}")
    
    def _handle_object_detected(self, event: RobotEvent):
        """Handle object detection event — log to learning DB."""
        obj_data = event.data
        label = obj_data.get('label', 'unknown')
        confidence = obj_data.get('confidence', 0)
        self.logger.debug(f"Object detected: {label} ({confidence:.0%})")
        
        if self.learning_db:
            self.learning_db.save_object(label, confidence=confidence)
    
    def _handle_wake_word(self, event: RobotEvent):
        """Handle wake word detection — start a multi-turn conversation session."""
        if getattr(self, 'conversation', None) and self.conversation.active:
            return  # already conversing
        self.logger.info("Wake word detected — starting conversation session")
        # The ConversationManager pauses the wake mic, greets, and runs the
        # command-mic dialogue loop until the conversation ends.
        self.conversation.start_session()
    
    def _handle_speech(self, event: RobotEvent):
        """Handle recognized speech"""
        text = event.data.get('text', '')
        self.logger.info(f"Speech recognized: {text}")
        print(f"\nYou: {text}", flush=True)

        # Process in brain
        self.brain.speech_received(text)
        
    def _handle_dialogue_idle(self, event: RobotEvent):
        """Resume wake-word listening once the brain returns to IDLE."""
        self.logger.debug("Dialogue cycle finished; resuming wake-word listener")
        self._resume_wake_word_listener()

    def _handle_speech_complete(self, event: RobotEvent):
        """Log text-to-speech completions for easier debugging."""
        utterance = event.data.get('text') if event.data else None
        self.logger.debug(f"Text-to-speech finished: {utterance}")
        self._resume_wake_word_listener()

    def _handle_speech_listen_failed(self, event: RobotEvent):
        """Recover keyword listening when a speech capture session fails."""
        reason = event.data.get('reason', 'unknown') if event.data else 'unknown'
        self.logger.warning(f"Speech capture failed ({reason}); resuming wake-word listener")
        self._resume_wake_word_listener()

    def _handle_battery_low(self, event: RobotEvent):
        """Handle low battery warning"""
        self.logger.warning("Low battery warning received")
        
        # Could trigger charging behavior
    
    def _handle_emergency_stop(self, event: RobotEvent):
        """Handle emergency stop"""
        self.logger.critical("EMERGENCY STOP TRIGGERED!")
        
        # Stop all modules immediately
        self.emergency_shutdown()

    def _pause_wake_word_listener(self, reason: str = "manual"):
        """Pause the wake-word listener to avoid crosstalk with active dialogue."""
        if self.keyword_listener_paused:
            return
        wake_module = self.modules.get('wake_word')
        if not wake_module:
            return
        try:
            wake_module.pause_listening(reason=reason)
            self.keyword_listener_paused = True
        except Exception as exc:
            self.logger.error(f"Failed to pause wake-word listener: {exc}")
    
    def _resume_wake_word_listener(self):
        """Resume passive wake-word listening once dialogue concludes."""
        # While a conversation is active, ONLY the ConversationManager may
        # resume the wake mic (at the very end). Stray resumes from TTS
        # 'speech_complete' events during the session would reopen the wake mic
        # and fight the command mic.
        if getattr(self, 'conversation', None) and self.conversation.active:
            return
        if not self.keyword_listener_paused:
            return
        wake_module = self.modules.get('wake_word')
        if not wake_module:
            self.keyword_listener_paused = False
            return
        try:
            wake_module.resume_listening()
            self.keyword_listener_paused = False
        except Exception as exc:
            self.logger.error(f"Failed to resume wake-word listener: {exc}")
    
    def _boost_input_gains(self):
        """Raise USB microphone capture gain to a usable level.

        USB mics often power up at 0% capture, which makes speech recognition
        return empty transcripts. We set the capture controls to max on the USB
        input cards (best-effort — ignored if the control doesn't exist). Runs
        every startup so it survives reboots without needing 'alsactl store'.
        """
        import subprocess
        for card in (0, 1):
            for ctrl in ("Mic", "Capture"):
                try:
                    subprocess.run(["amixer", "-c", str(card), "sset", ctrl, "100%", "cap"],
                                   capture_output=True, timeout=5)
                except Exception:
                    pass
        self.logger.info("Microphone capture gains set to max")

    def start_modules(self):
        """Start all initialized modules"""
        self.logger.info("Starting modules...")

        # Ensure USB mics are at usable capture gain (fixes empty transcripts).
        self._boost_input_gains()

        # Start vision modules (they subscribe to shared camera)
        if 'face_recognition' in self.modules:
            try:
                self.modules['face_recognition'].start()
                self.logger.info("✓ Face Recognition started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Face Recognition: {e}")
        
        if 'object_detection' in self.modules:
            try:
                self.modules['object_detection'].start()
                self.logger.info("✓ Object Detection started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Object Detection: {e}")
        
        if 'wake_word' in self.modules:
            try:
                self.modules['wake_word'].start()
                self.logger.info("✓ Wake Word Detection started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Wake Word Detection: {e}")
        
        if 'speech_recognition' in self.modules:
            try:
                self.modules['speech_recognition'].start()
                self.logger.info("✓ Speech Recognition service started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Speech Recognition: {e}")
        
        if 'tts' in self.modules:
            try:
                self.modules['tts'].start()
                self.logger.info("✓ Text-to-Speech engine started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Text-to-Speech: {e}")
        
        # Microcontroller bridge is already connected in initialize_modules().

        # Start robot brain
        self.brain.start()
        self.logger.info("✓ Robot Brain started")

        # Reminders announcer + internet watchdog + Telegram chat.
        try:
            self.reminders.start()
        except Exception as e:
            self.logger.error(f"reminders start failed: {e}")
        if getattr(self, 'telegram', None):
            try:
                self.telegram.start()
            except Exception as e:
                self.logger.error(f"telegram start failed: {e}")
        threading.Thread(target=self._network_monitor_loop, daemon=True).start()

        self.logger.info("All modules started")

    def _announce(self, msg: str):
        """Speak a proactive message (reminders, alerts) out loud + to console."""
        print(f"\n{behavior_config.robot_name}: {msg}\n", flush=True)
        tts = self.modules.get('tts')
        if tts and getattr(tts, 'running', False):
            try:
                tts.speak(msg)
            except Exception:
                pass

    def _network_monitor_loop(self):
        """Announce (once) when the internet connection is lost, and point the
        user at the dashboard WiFi panel to reconnect."""
        try:
            from modules.hardware import wifi
        except Exception:
            return
        was_online = True
        while self.running:
            online = wifi.is_online()
            if was_online and not online:
                conv = getattr(self, 'conversation', None)
                if not (conv and conv.active):   # don't talk over a conversation
                    msg = ("I've lost my internet connection. You can reconnect me "
                           "from the dashboard WiFi panel.")
                    print(f"\n{behavior_config.robot_name}: {msg}\n", flush=True)
                    tts = self.modules.get('tts')
                    if tts and getattr(tts, 'running', False):
                        try:
                            tts.speak(msg)
                        except Exception:
                            pass
            was_online = online
            time.sleep(30)
    
    def start_performance_monitor(self):
        """Start performance monitoring thread"""
        def monitor():
            while self.running:
                try:
                    # Get system stats
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    # Get process stats
                    process = psutil.Process()
                    process_cpu = process.cpu_percent()
                    process_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Update brain health status
                    self.brain.health_status.update({
                        'cpu_usage': cpu_percent,
                        'memory_usage': memory_percent,
                        'process_cpu': process_cpu,
                        'process_memory_mb': process_memory
                    })
                    
                    # Log if high usage
                    if cpu_percent > 80:
                        self.logger.warning(f"High CPU usage: {cpu_percent}%")
                    if memory_percent > 80:
                        self.logger.warning(f"High memory usage: {memory_percent}%")
                    
                    # Check temperature (Raspberry Pi specific)
                    try:
                        temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
                        if temp_file.exists():
                            temp = int(temp_file.read_text()) / 1000
                            self.brain.health_status['temperature'] = temp
                            
                            if temp > 85:
                                self.logger.warning(f"High temperature: {temp}°C")
                    except:
                        pass
                    
                    # Sleep before next check
                    time.sleep(system_config.health_check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Performance monitor error: {e}")
                    time.sleep(5)
        
        self.performance_monitor = threading.Thread(target=monitor, daemon=True)
        self.performance_monitor.start()
        self.logger.info("Performance monitor started")
    
    def run(self):
        """Main run loop"""
        self.running = True
        
        self.logger.info("=" * 60)
        self.logger.info(f"   {behavior_config.robot_name} is now ONLINE!")
        self.logger.info("=" * 60)
        
        # Start performance monitoring
        self.start_performance_monitor()
        
        # Main loop
        try:
            while self.running:
                # Check for shutdown signal
                if self.shutdown_event.is_set():
                    break
                
                # Get robot status
                status = self.brain.get_status()
                
                # Log status periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    self.logger.debug(f"Status: {status['state']}, "
                                    f"User: {status.get('current_user', 'None')}, "
                                    f"Uptime: {status['uptime']:.1f}s")
                
                # Check error threshold
                if self.error_count > self.max_errors:
                    self.logger.error("Maximum error count exceeded, shutting down")
                    break
                
                # Small delay to prevent CPU spinning
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.critical(f"Critical error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating shutdown sequence...")
        
        self.running = False
        
        # Stop brain
        self.brain.stop()
        
        # Stop all modules
        for name, module in self.modules.items():
            try:
                if hasattr(module, 'stop'):
                    module.stop()
                    self.logger.info(f"✓ {name} stopped")
            except Exception as e:
                self.logger.error(f"✗ Error stopping {name}: {e}")
        
        # Stop web dashboard
        if getattr(self, 'dashboard', None):
            try:
                self.dashboard.stop()
                self.logger.info("✓ Web Dashboard stopped")
            except Exception as e:
                self.logger.error(f"✗ Error stopping Dashboard: {e}")
        
        # Stop shared hardware managers
        try:
            self.camera_manager.stop()
            self.logger.info("✓ Camera Manager stopped")
        except Exception as e:
            self.logger.error(f"✗ Error stopping Camera Manager: {e}")
        
        try:
            for role in list(self.audio_manager._leases.keys()):
                self.audio_manager.release(role)
            self.logger.info("✓ Audio Manager released")
        except Exception as e:
            self.logger.error(f"✗ Error releasing Audio Manager: {e}")
        
        # Close learning DB
        try:
            self.learning_db.close()
            self.logger.info("✓ Learning DB closed")
        except Exception as e:
            self.logger.error(f"✗ Error closing Learning DB: {e}")
        
        # Save final state
        self._save_state()
        
        # Calculate runtime
        runtime = time.time() - self.start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        
        self.logger.info("=" * 60)
        self.logger.info(f"   {behavior_config.robot_name} SHUTDOWN COMPLETE")
        self.logger.info(f"   Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.logger.info("=" * 60)
    
    def emergency_shutdown(self):
        """Emergency shutdown - stop everything immediately"""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        self.running = False
        self.shutdown_event.set()
        
        # Force stop all modules
        for name, module in self.modules.items():
            try:
                if hasattr(module, 'emergency_stop'):
                    module.emergency_stop()
                elif hasattr(module, 'stop'):
                    module.stop()
            except:
                pass  # Ignore errors during emergency shutdown
    
    def _save_state(self):
        """Save current state to file"""
        state = {
            'shutdown_time': time.time(),
            'runtime': time.time() - self.start_time,
            'brain_status': self.brain.get_status(),
            'error_count': self.error_count,
            'modules': list(self.modules.keys())
        }
        
        try:
            state_file = PROJECT_ROOT / "data" / "last_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")


def signal_handler(signum, frame):
    """Handle system signals"""
    print("\nShutdown signal received")
    sys.exit(0)


def main():
    """Main entry point"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Robot System')
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        type=str,
        default=None
    )
    parser.add_argument(
        '--debug',
        help='Enable debug mode',
        action='store_true'
    )
    parser.add_argument(
        '--test',
        help='Run in test mode',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        system_config.debug_mode = True
        system_config.log_level = 'DEBUG'
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run robot
    robot = None
    
    try:
        # Initialize robot
        robot = AIRobot(config_file=args.config)
        
        # Test mode
        if args.test:
            print("\n" + "=" * 60)
            print("   RUNNING IN TEST MODE")
            print("=" * 60)
            
            # Initialize modules
            robot.initialize_modules()
            
            # Run tests
            run_system_tests(robot)
            
            print("\n" + "=" * 60)
            print("   TEST COMPLETE")
            print("=" * 60)
            
        else:
            # Normal operation
            # Initialize modules
            robot.initialize_modules()
            
            # Start modules
            robot.start_modules()
            
            # Run main loop
            robot.run()
    
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if robot:
            robot.shutdown()


def run_system_tests(robot: AIRobot):
    """
    Run system tests
    
    Args:
        robot: Robot instance to test
    """
    print("\nRunning system tests...")
    
    # Test 1: Configuration
    print("\n1. Configuration Test:")
    if config.validate():
        print("   ✓ Configuration valid")
    else:
        print("   ✗ Configuration invalid")
    
    # Test 2: Module initialization
    print("\n2. Module Initialization:")
    for name, module in robot.modules.items():
        print(f"   ✓ {name} initialized")
    
    # Test 3: Camera test
    print("\n3. Camera Test:")
    if 'face_recognition' in robot.modules:
        face_module = robot.modules['face_recognition']
        if face_module.initialize_camera():
            print("   ✓ Camera accessible")
            
            # Try to capture a frame
            time.sleep(1)
            frame = face_module.get_current_frame()
            if frame is not None:
                print(f"   ✓ Frame captured: {frame.shape}")
            else:
                print("   ✗ Failed to capture frame")
        else:
            print("   ✗ Camera not accessible")
    
    # Test 4: Brain state machine
    print("\n4. Brain State Machine:")
    brain = robot.brain
    print(f"   Initial state: {brain.state.name}")
    
    # Test state transitions
    brain.startup_complete()
    print(f"   After startup: {brain.state.name}")
    
    brain.wake_word_heard()
    print(f"   After wake word: {brain.state.name}")
    
    brain.return_idle()
    print(f"   Return to idle: {brain.state.name}")
    
    # Test 5: Event system
    print("\n5. Event System:")
    test_event = RobotEvent(
        type='test_event',
        source='test',
        data={'test': True},
        priority=5
    )
    brain.emit_event(test_event)
    print("   ✓ Event emitted successfully")
    
    # Test 6: Memory system
    print("\n6. Memory System:")
    brain.add_memory(
        content="Test memory",
        memory_type='short_term',
        importance=0.5
    )
    memories = brain.recall_memory("Test")
    if memories:
        print(f"   ✓ Memory stored and recalled: {len(memories)} items")
    else:
        print("   ✗ Memory recall failed")
    
    # Test 7: Performance check
    print("\n7. Performance Check:")
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    print(f"   CPU Usage: {cpu_percent}%")
    print(f"   Memory Usage: {memory_percent}%")
    print(f"   Disk Usage: {disk_percent}%")
    
    # Check temperature (Raspberry Pi)
    try:
        temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_file.exists():
            temp = int(temp_file.read_text()) / 1000
            print(f"   Temperature: {temp}°C")
    except:
        print("   Temperature: N/A")
    
    # Test 8: File system
    print("\n8. File System:")
    data_dirs = [
        "data/models",
        "data/faces",
        "data/voices",
        "data/logs"
    ]
    
    for dir_path in data_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path} exists")
        else:
            print(f"   ✗ {dir_path} missing")
    
    # Test 9: Hardware interfaces (if available)
    print("\n9. Hardware Interfaces:")
    
    # Check for ESP32
    esp32_port = Path(hardware_config.esp32_port)
    if esp32_port.exists():
        print(f"   ✓ ESP32 port found: {esp32_port}")
    else:
        print(f"   ✗ ESP32 port not found: {esp32_port}")
    
    # Check for GSM / 4G modem (SIM7600X)
    gsm_port = Path(hardware_config.sim7600x_port)
    if gsm_port.exists():
        print(f"   ✓ GSM port found: {gsm_port}")
    else:
        print(f"   ✗ GSM port not found: {gsm_port}")
    
    # Test 10: Network connectivity
    print("\n10. Network Test:")
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("   ✓ Internet connection available")
    except:
        print("   ✗ No internet connection")
    
    print("\nTest Summary:")
    print("All basic systems checked. Review results above.")


def create_systemd_service():
    """
    Create systemd service file for auto-start on boot
    This should be run with sudo
    """
    import getpass
    user = getpass.getuser()
    python_bin = sys.executable  # the venv interpreter running this process
    service_content = f"""[Unit]
Description=Gonzo AI Robot
After=network-online.target hailo-ollama.service
Wants=network-online.target

[Service]
Type=simple
ExecStart={python_bin} {PROJECT_ROOT}/main.py
Restart=on-failure
RestartSec=5
User={user}
WorkingDirectory={PROJECT_ROOT}
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = "/etc/systemd/system/ai-robot.service"
    
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"Service file created at {service_file}")
        print("To enable auto-start on boot, run:")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable ai-robot.service")
        print("  sudo systemctl start ai-robot.service")
        
    except PermissionError:
        print("Permission denied. Run with sudo to create service file:")
        print(f"  sudo python3 {__file__} --create-service")


if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1 and sys.argv[1] == '--create-service':
        create_systemd_service()
    else:
        main()