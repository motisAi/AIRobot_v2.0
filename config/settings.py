"""
Configuration Settings Module
=============================
Central configuration file for all robot parameters.
Loads environment variables and provides system-wide settings.

"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# YAML is the preferred human-editable config format. Guarded so the app
# still runs (JSON only) if PyYAML is not installed.
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    yaml = None
    YAML_AVAILABLE = False

from .platforms import detect_platform, PLATFORM_OVERRIDES

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Default base config file. YAML is preferred; falls back to legacy JSON.
DEFAULT_CONFIG_YAML = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_CONFIG_JSON = PROJECT_ROOT / "config" / "config.json"


def default_config_path() -> Optional[str]:
    """Return the base config file to auto-load (YAML preferred, JSON fallback)."""
    if DEFAULT_CONFIG_YAML.exists() and YAML_AVAILABLE:
        return str(DEFAULT_CONFIG_YAML)
    if DEFAULT_CONFIG_JSON.exists():
        return str(DEFAULT_CONFIG_JSON)
    if DEFAULT_CONFIG_YAML.exists():
        # YAML present but PyYAML missing — warn later during load.
        return str(DEFAULT_CONFIG_YAML)
    return None


def detect_hailo_device() -> bool:
    """
    Detect if Hailo AI accelerator is physically installed and available.
    
    Returns:
        bool: True if Hailo device is detected, False otherwise
    """
    try:
        # Check for Hailo device in /dev
        hailo_devices = list(Path('/dev').glob('hailo*'))
        if hailo_devices:
            logging.info(f"Hailo device(s) detected: {hailo_devices}")
            return True
            
        # Check for Hailo via lspci (PCIe devices)
        import subprocess
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        if 'hailo' in result.stdout.lower():
            logging.info("Hailo PCIe device detected via lspci")
            return True
            
        # Try importing hailo SDK
        try:
            import hailo_platform
            logging.info("Hailo SDK available")
            return True
        except ImportError:
            pass
            
        logging.warning("No Hailo device detected")
        return False
        
    except Exception as e:
        logging.error(f"Error detecting Hailo device: {e}")
        return False


def detect_sim7600x_module() -> bool:
    """
    Detect if SIM7600X module is connected and responsive.
    
    Returns:
        bool: True if SIM7600X is detected, False otherwise
    """
    try:
        # Check for SIM7600X on GPIO pins 0&1 (UART0)
        # Typically appears as /dev/ttyS0 or /dev/ttyAMA0
        potential_ports = ['/dev/ttyS0', '/dev/ttyAMA0', '/dev/serial0']
        
        for port in potential_ports:
            if Path(port).exists():
                try:
                    import serial
                    # Test communication with AT commands
                    ser = serial.Serial(port, 115200, timeout=2)
                    ser.write(b'AT\r\n')
                    response = ser.read(100).decode('utf-8', errors='ignore')
                    ser.close()
                    
                    if 'OK' in response:
                        logging.info(f"SIM7600X detected on {port}")
                        return True
                        
                except Exception as e:
                    logging.debug(f"No response from {port}: {e}")
                    continue
                    
        logging.warning("SIM7600X module not detected")
        return False
        
    except Exception as e:
        logging.error(f"Error detecting SIM7600X: {e}")
        return False


# Hardware detection results
HAILO_AVAILABLE = detect_hailo_device()
SIM7600X_AVAILABLE = detect_sim7600x_module()


@dataclass
class ModelConfig:
    """AI Model configurations"""
    
    # Face Recognition Settings
    face_model: str = "VGG-Face"  # Options: VGG-Face, Facenet, OpenFace, DeepFace
    face_backend: str = "opencv"  # Options: opencv, ssd, dlib, mtcnn
    face_distance_metric: str = "cosine"  # Options: cosine, euclidean, euclidean_l2
    face_recognition_threshold: float = 0.4  # Lower = more strict
    face_detection_confidence: float = 0.7
    
    # Object Detection Settings (YOLO with Hailo)
    object_model: str = "yolov8n"  # Nano version for speed
    object_confidence_threshold: float = 0.5
    object_nms_threshold: float = 0.4
    object_max_detections: int = 100
    object_classes_filter: list = field(default_factory=lambda: [])  # Empty = all classes
    
    # Speech Recognition
    # stt_mode: auto (Google online, fall back to offline Vosk) | google | vosk
    stt_mode: str = "auto"
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_language: str = "en"
    whisper_device: str = "cpu"  # cuda if available
    whisper_compute_type: str = "int8"  # int8 for speed, float16 for accuracy
    whisper_beam_size: int = 5
    whisper_patience: float = 1.0
    
    # Text-to-Speech
    # Engine: piper (neural, offline, natural — recommended) | pyttsx3 | espeak
    tts_engine: str = "piper"
    piper_binary: str = str(PROJECT_ROOT / "data" / "models" / "piper" / "piper" / "piper")
    piper_voice: str = str(PROJECT_ROOT / "data" / "models" / "piper" / "en_US-amy-medium.onnx")
    piper_length_scale: float = 1.0   # >1 slower, <1 faster speech
    piper_pitch: float = 0.0          # semitones to shift voice (needs sox): + up, - down
    piper_noise_scale: float = 0.667  # expressiveness: 0.3 = flat/steady .. 1.0 = varied
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"  # (Coqui, unused)
    tts_vocoder: str = "vocoder_models/en/ljspeech/hifigan_v2"
    tts_speaker_wav: Optional[str] = None  # Path to speaker voice sample
    tts_language: str = "en"
    tts_speed: float = 1.0  # Speech speed multiplier
    
    # Wake Word Detection
    wake_word: str = "gonzo"
    wake_word_sensitivity: float = 0.5  # 0-1, higher = more sensitive
    wake_word_model_path: str = str(PROJECT_ROOT / "data" / "models" / "wake_word.ppn")
    picovoice_access_key: str = os.getenv("PICOVOICE_ACCESS_KEY", "")
    # Engine: auto | vosk | porcupine | energy.
    #   vosk      -> free, offline, detects the actual word "gonzo" (recommended)
    #   porcupine -> Picovoice (needs a paid/registered key)
    #   energy    -> loudness trigger (no signup, but not word-specific)
    wake_word_engine: str = "auto"
    vosk_model_path: str = str(PROJECT_ROOT / "data" / "models" / "vosk-small-en")
    # Phrases that count as the wake word (Vosk). Fuzzy 'gonz*' also matches.
    wake_word_phrases: list = field(default_factory=lambda: ["gonzo", "hey gonzo"])
    
    # Voice Identification
    voice_embedding_size: int = 512
    voice_similarity_threshold: float = 0.85
    voice_sample_duration: int = 3  # seconds


@dataclass
class HardwareConfig:
    """Hardware interface configurations"""
    
    # Camera Settings
    camera_index: int = 0  # 0 for /dev/video0
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 30
    camera_buffer_size: int = 1
    camera_format: str = "MJPEG"
    # Image controls (None = leave the camera's default). Bump brightness/gain
    # for low light; keep auto_exposure = 3 (auto) so it adapts.
    camera_auto_exposure: Optional[int] = 3   # 3 = auto/aperture-priority, 1 = manual
    camera_brightness: Optional[int] = None    # e.g. 40-120 for low light
    camera_contrast: Optional[int] = None
    camera_gain: Optional[int] = None          # e.g. 100-200 for low light
    camera_gamma: Optional[int] = None
    # Software low-light boost (adaptive; brightens dark frames for face recog
    # without washing out well-lit ones). Recommended on.
    camera_low_light_boost: bool = True
    
    # Microphone Settings
    microphone_device_index: Optional[int] = None  # None = default device
    microphone_channels: int = 1
    microphone_rate: int = 48000
    microphone_chunk: int = 1024
    microphone_timeout: float = 0.8
    microphone_phrase_time_limit: float = 5.0
    wake_word_microphone_name: Optional[str] = "Auto Focus Camera"
    speech_microphone_name: Optional[str] = "USB PnP Sound Device"
    wake_word_device_index: Optional[int] = None  # resolved by name at runtime
    speech_device_index: Optional[int] = None  # resolved by name at runtime
    speech_microphone_rate: int = 44100  # USB PnP Sound Device supports 44100Hz

    # Audio OUTPUT (where the robot speaks). Change this to move sound from the
    # HDMI monitor to a speaker plugged into the Pi.
    #   audio_output_device: "hdmi"  -> HDMI monitor/TV speakers (default now)
    #                        "usb"   -> a USB speaker/USB-audio dongle on the Pi
    #                        "analog"-> 3.5mm/I2S DAC hat
    #                        "auto"  -> first non-HDMI device, else HDMI
    #                        "default" -> ALSA default device
    audio_output_device: str = "hdmi"
    audio_output_card: Optional[int] = None   # explicit ALSA card index (overrides device)
    audio_output_name: Optional[str] = None    # match output device by name substring
    tts_output_volume: float = 1.0             # 0.0 - 1.0

    # ESP32 Serial Communication
    esp32_port: str = "/dev/ttyUSB0"  # Might be /dev/ttyACM0
    esp32_baudrate: int = 115200
    esp32_timeout: float = 1.0
    esp32_retry_attempts: int = 3
    
    # SIM7600X 4G Module (Waveshare) - Connected to GPIO pins 0&1 (UART0)
    sim7600x_port: str = "/dev/ttyAMA1"  # Primary port for SIM7600X on GPIO 0&1
    sim7600x_alt_ports: list = field(default_factory=lambda: ["/dev/ttyAMA0", "/dev/serial0"])  # Alternative ports
    sim7600x_baudrate: int = 115200
    sim7600x_pin: Optional[str] = None  # SIM PIN if required
    sim7600x_apn: str = "internet"  # Change based on your carrier (e.g., "hologram" for Hologram)
    sim7600x_timeout: float = 10.0
    sim7600x_retry_attempts: int = 3
    sim7600x_power_pin: int = 6  # GPIO pin to control power (if wired)
    sim7600x_reset_pin: int = 5  # GPIO pin to control reset (if wired)
    sim7600x_status_pin: int = 13  # GPIO pin to read status (if wired)
    
    # Network Settings for SIM7600X
    network_mode: str = "auto"  # auto, lte, gsm, 3g
    preferred_network: str = "lte"
    roaming_enabled: bool = True
    
    # Hailo AI Accelerator
    hailo_device_id: int = 0
    hailo_power_mode: str = "performance"  # performance, balanced, power_save
    hailo_batch_size: int = 1
    
    # GPIO Pins (Raspberry Pi)
    gpio_mode: str = "BCM"  # BCM or BOARD
    pin_status_led: int = 18
    pin_emergency_stop: int = 25
    pin_motor_enable: int = 24
    
    # Servo Configuration (Robotic Hand)
    servo_pins: list = field(default_factory=lambda: [5, 6, 13, 19, 26, 21])
    servo_min_pulse: int = 500
    servo_max_pulse: int = 2500
    servo_frequency: int = 50
    is_esp_connected: bool = False
    uart_device_map: Dict[str, str] = field(
        default_factory=lambda: {
            "esp32": "/dev/ttyUSB0",
            "sim7600x": "/dev/ttyAMA1",
        }
    )


@dataclass
class SystemConfig:
    """System-level configurations"""
    
    platform_name: str = "generic"
    use_tensor_rt: bool = False
    data_dir: str = str(PROJECT_ROOT / "data")
    # Performance Settings
    max_threads: int = 8
    vision_thread_count: int = 2
    audio_thread_count: int = 2
    enable_gpu: bool = False
    enable_hailo: bool = HAILO_AVAILABLE  # Automatically detect Hailo availability
    enable_sim7600x: bool = SIM7600X_AVAILABLE  # Automatically detect SIM7600X availability
    
    # Processing Optimization
    frame_skip: int = 3  # Process every Nth frame
    batch_processing: bool = True
    model_quantization: bool = True
    cache_enabled: bool = True
    cache_size_mb: int = 512
    
    # Timing Configuration
    main_loop_delay: float = 0.01  # 10ms
    vision_process_interval: float = 0.1  # 100ms
    audio_process_interval: float = 0.05  # 50ms
    sensor_read_interval: float = 0.5  # 500ms
    health_check_interval: float = 5.0  # 5 seconds
    
    # Memory Management
    max_memory_percent: float = 80.0
    clear_cache_threshold: float = 70.0
    max_log_size_mb: int = 100
    max_recording_seconds: int = 30
    
    # Behavior Settings
    idle_timeout: float = 300.0  # 5 minutes
    patrol_mode_enabled: bool = False
    auto_learning_enabled: bool = True
    conversation_timeout: float = 60.0
    
    # Debug and Logging
    debug_mode: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_dir: str = str(PROJECT_ROOT / "data" / "logs")
    enable_profiling: bool = False


@dataclass
class SecurityConfig:
    """Security and privacy configurations"""
    
    # Authentication
    require_authentication: bool = True
    master_user_id: str = os.getenv("MASTER_USER_ID", "master_001")
    max_auth_attempts: int = 3
    auth_timeout_seconds: float = 30.0
    
    # Privacy
    store_recordings: bool = False
    store_faces: bool = True
    anonymize_logs: bool = False
    
    # Network Security
    enable_ssl: bool = True
    api_key: str = os.getenv("API_KEY", "generate_random_key_here")
    allowed_origins: list = field(default_factory=lambda: ["http://localhost:*"])
    
    # Access Control
    remote_access_enabled: bool = False
    require_physical_button: bool = False
    emergency_stop_enabled: bool = True

    # Home guard / security mode. When armed (by voice, phone, or the dashboard),
    # Stella alerts on MOVEMENT or a human body — she does NOT need to recognise
    # a face. Face recognition only suppresses alerts when it sees the master.
    guard_motion_detection: bool = True     # alert on movement/body, not just faces
    guard_sensitivity: str = "medium"       # low | medium | high (how little movement triggers)
    guard_detect_person: bool = True        # label alerts "a person" vs "movement" (HOG)
    guard_alert_cooldown: float = 30.0      # min seconds between phone alerts
    # Disarming is master-only (arming isn't — it's harmless). Because face
    # recognition is unreliable, the RELIABLE master path to disarm by voice is a
    # spoken pass-phrase. Telegram (token-secured) always works too.
    guard_require_master_to_disarm: bool = True  # false = any present speaker can disarm
    guard_disarm_phrase: str = ""           # e.g. "sunflower" -> say "guard off sunflower"

    # How long (seconds) Stella keeps trusting a recognised identity after the
    # last time she saw that face. Short = she forgets you between glances (and
    # asks your name again); longer bridges the gaps in sparse face detection.
    identity_memory_seconds: float = 60.0


@dataclass
class BehaviorConfig:
    """Robot behavior and personality settings"""
    
    # Personality
    robot_name: str = os.getenv("ROBOT_NAME", "RoboAI")
    personality_type: str = "helpful"  # helpful, playful, professional
    response_style: str = "concise"  # concise, detailed, chatty
    # Language Stella listens & speaks in: "en" (English) | "he" (Hebrew).
    # One switch flips speech-to-text and the voice. Her brain (the LLM) already
    # understands both, so this mainly controls hearing + speaking.
    language: str = "en"
    
    # Interaction Settings
    greeting_message: str = "Hello! I'm {name}. How can I help you?"
    goodbye_message: str = "Goodbye! Have a great day!"
    unknown_person_response: str = "Hello! I don't think we've met. What's your name?"
    
    # Learning Behavior
    learn_new_faces: bool = True
    learn_new_objects: bool = True
    remember_conversations: bool = True
    max_memories: int = 1000
    
    # Movement Behavior
    obstacle_detection_range: float = 30.0  # cm
    max_speed: float = 0.5  # m/s
    turn_speed: float = 45.0  # degrees/s
    safe_distance: float = 50.0  # cm
    auto_charge: bool = False


@dataclass
class AIConfig:
    """Conversation 'brain' configuration.

    mode:
      "local"  -> use the on-device Hailo-10H NPU LLM (hailo-ollama). Free,
                  offline, private. Default.
      "online" -> use a cloud provider (OpenAI/Anthropic) as primary.
      "hybrid" -> local by default, escalate hard questions to the cloud when
                  online (see ``allow_cloud_escalation``).
    Secrets (API keys) always come from the .env file, never from YAML.
    """

    mode: str = os.getenv("AI_MODE", "local")           # local | online | hybrid
    allow_cloud_escalation: bool = False                 # used when mode == "hybrid"

    # Cloud provider
    online_provider: str = os.getenv("AI_PROVIDER", "groq")  # groq | openai | anthropic
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    # Smaller/faster Groq model (same key) used automatically when the big model
    # is rate-limited (HTTP 429), so Stella degrades gracefully instead of dying.
    groq_fast_model: str = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # Fallback chain order (tried left-to-right; 'hailo' = local NPU always works)
    fallback_order: list = field(default_factory=lambda: ["groq", "gemini", "hailo"])
    # Agent mode: let the LLM call tools (web, weather, camera, devices, reminders)
    agent_enabled: bool = True

    # On-device Hailo-10H NPU LLM (hailo-ollama REST server)
    hailo_ollama_url: str = os.getenv("HAILO_OLLAMA_URL", "http://localhost:8000")
    hailo_ollama_model: str = os.getenv("HAILO_OLLAMA_MODEL", "qwen2.5-instruct:1.5b")

    # CPU offline fallback (llama.cpp GGUF) — optional
    offline_model_path: str = os.getenv("OFFLINE_MODEL_PATH", "")

    # Generation
    max_tokens: int = 400
    temperature: float = 0.7
    conversation_memory_turns: int = 20


@dataclass
class HandConfig:
    """Robotic hand (ESP32 + PCA9685 gesture firmware over serial)."""
    enabled: bool = False               # set true once the ESP32 hand firmware is flashed
    serial_port: str = "/dev/ttyUSB0"
    baud: int = 115200
    wave_on_greeting: bool = True        # wave hello when Stella greets someone
    wave_on_goodbye: bool = True         # wave when saying goodbye
    middle_finger_on_insult: bool = True # cheeky: raise the middle finger when insulted


@dataclass
class MusicConfig:
    """YouTube music playback (via mpv + yt-dlp)."""
    default_volume: int = 70    # 0-100 (mpv allows up to 130)
    duck_volume: int = 25       # volume while Stella talks to you over the music
    volume_step: int = 20       # how much each louder/quieter command changes


@dataclass
class WebSearchConfig:
    """Lets the robot answer questions from the live web (like Claude/Copilot)."""

    enabled: bool = True
    provider: str = "duckduckgo"   # duckduckgo (free, no key) | none
    max_results: int = 4
    timeout: float = 8.0
    region: str = "wt-wt"
    # Let the AI decide when to search (tool-use). If False, never searches.
    allow_auto_search: bool = True


@dataclass
class MicrocontrollerConfig:
    """Generic bridge to an external microcontroller (ESP32 / Pi Zero / Arduino).

    When ``connected`` is False, device commands ("turn on the light") are
    accepted and logged but not transmitted — so you can develop the whole
    voice pipeline before any hardware is wired. Flip ``connected: true`` and
    pick a transport to start sending real commands.
    """

    connected: bool = False                 # microcontroller_connected
    transport: str = "serial"               # serial | network | null
    # Serial transport (ESP32/Arduino over USB-UART)
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    serial_timeout: float = 1.0
    protocol: str = "json"                  # json (line-delimited) | esp32_binary
    # Network transport (Pi Zero / ESP32 over Wi-Fi)
    host: str = "192.168.11.50"
    port: int = 8080
    network_path: str = "/command"          # HTTP path for network transport
    ack_timeout: float = 2.0
    # Friendly-name -> device id understood by the microcontroller firmware.
    # e.g. saying "turn on the light" sends {"target": "relay1", "action": "on"}.
    device_map: Dict[str, str] = field(
        default_factory=lambda: {
            "light": "relay1",
            "lamp": "relay1",
            "fan": "relay2",
            "door": "servo1",
        }
    )


@dataclass
class NavigationConfig:
    """Future 'learn my house' / autonomous navigation capabilities.

    All off by default. As you add wheels and sensors to the microcontroller,
    enable the relevant flags — the robot exposes the hooks now so the code is
    ready when the hardware arrives.
    """

    enabled: bool = False
    has_wheels: bool = False
    has_lidar: bool = False
    has_ultrasonic: bool = False
    has_imu: bool = False
    has_wheel_encoders: bool = False
    mapping_enabled: bool = False           # build/save a map of the house
    map_dir: str = str(PROJECT_ROOT / "data" / "maps")
    obstacle_stop_distance_cm: float = 20.0
    cruise_speed: float = 0.3               # m/s
    exploration_enabled: bool = False       # autonomously explore to build a map


@dataclass
class ConversationConfig:
    """Multi-turn voice conversation session behaviour.

    Flow: wake word -> greet -> converse (multi-turn) -> after `idle_timeout`
    seconds of silence, ask the wrap-up prompt -> on no/negative reply, say the
    farewell and go back to listening for the wake word.
    """

    idle_timeout: float = 12.0        # seconds of silence before the wrap-up prompt
    end_silence: float = 1.2          # seconds of silence that ends one utterance
    max_utterance: float = 12.0       # hard cap on a single utterance (seconds)
    greeting: str = "How can I help you{name}?"
    wrap_up_prompt: str = "Can I do anything else for you{name}?"
    farewell: str = "See you later{name}!"
    end_phrases: list = field(default_factory=lambda: [
        "no", "nope", "no thanks", "no thank you", "nothing", "that's all",
        "that is all", "that's it", "bye", "goodbye", "see you", "stop",
        "i'm good", "im good", "im done", "i'm done", "nothing else",
    ])


class RobotConfig:
    """Main configuration class that combines all settings"""
    
    PROJECT_ROOT = PROJECT_ROOT  # Expose as class attribute for modules
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize robot configuration
        
        Args:
            config_file: Optional path to JSON config file for overrides
        """
        self.model = ModelConfig()
        self.hardware = HardwareConfig()
        self.system = SystemConfig()
        self.security = SecurityConfig()
        self.behavior = BehaviorConfig()
        self.ai = AIConfig()
        self.web_search = WebSearchConfig()
        self.music = MusicConfig()
        self.hand = HandConfig()
        self.microcontroller = MicrocontrollerConfig()
        self.navigation = NavigationConfig()
        self.conversation = ConversationConfig()
        self.platform_id = detect_platform()

        # Create necessary directories
        self._create_directories()
        self._apply_platform_profile()

        # Load custom config if provided, otherwise auto-load the base config
        # file (config/config.yaml preferred, config/config.json fallback).
        load_target = config_file or default_config_path()
        if load_target and Path(load_target).exists():
            self.load_from_file(load_target)

        # Keep the legacy ESP flag in sync with the new microcontroller section
        # so older code paths continue to work.
        self.hardware.is_esp_connected = bool(self.microcontroller.connected)

        # Setup logging
        self._setup_logging()
        logging.info("Configuration initialized for platform: %s", self.platform_id)
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            PROJECT_ROOT / "data" / "models",
            PROJECT_ROOT / "data" / "faces",
            PROJECT_ROOT / "data" / "voices",
            PROJECT_ROOT / "data" / "maps",
            PROJECT_ROOT / "data" / "logs",
            PROJECT_ROOT / "data" / "recordings",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging system"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers = [logging.StreamHandler()]
        
        if self.system.log_to_file:
            log_file = Path(self.system.log_dir) / "robot.log"
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.system.log_level),
            format=log_format,
            handlers=handlers
        )

    def _apply_platform_profile(self):
        """Apply platform-specific overrides when helpers are available."""

        helper = PLATFORM_OVERRIDES.get(self.platform_id)
        if helper is None:
            return

        try:
            helper(self)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Failed to apply platform profile '%s': %s", self.platform_id, exc)
    
    def load_from_file(self, filepath: str):
        """
        Load configuration from a YAML or JSON file.

        The file format is chosen by extension: ``.yaml``/``.yml`` use PyYAML,
        anything else is parsed as JSON. The parsed dict is merged onto the
        typed config sections (model/hardware/system/security/behavior/ai/
        web_search/microcontroller/navigation); unknown keys are ignored.

        Args:
            filepath: Path to a YAML or JSON configuration file.
        """
        try:
            suffix = Path(filepath).suffix.lower()
            with open(filepath, 'r', encoding='utf-8') as f:
                if suffix in ('.yaml', '.yml'):
                    if not YAML_AVAILABLE:
                        logging.error(
                            "Config %s is YAML but PyYAML is not installed. "
                            "Run: pip install pyyaml", filepath,
                        )
                        return
                    config_data = yaml.safe_load(f) or {}
                else:
                    config_data = json.load(f)

            if not isinstance(config_data, dict):
                logging.error("Config file %s did not parse to a mapping", filepath)
                return

            # Update configurations
            for section_name, section_data in config_data.items():
                if not isinstance(section_data, dict):
                    continue
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                        else:
                            logging.debug(
                                "Ignoring unknown config key %s.%s", section_name, key,
                            )
                else:
                    logging.debug("Ignoring unknown config section '%s'", section_name)

            logging.info(f"Configuration loaded from {filepath}")

        except Exception as e:
            logging.error(f"Failed to load config from {filepath}: {e}")
    
    def save_to_file(self, filepath: str):
        """
        Save current configuration to a YAML or JSON file (chosen by extension).

        Args:
            filepath: Path to save the configuration to.
        """
        config_data = self.get_all_settings()

        try:
            suffix = Path(filepath).suffix.lower()
            with open(filepath, 'w', encoding='utf-8') as f:
                if suffix in ('.yaml', '.yml') and YAML_AVAILABLE:
                    yaml.safe_dump(config_data, f, sort_keys=False, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=4, default=str)

            logging.info(f"Configuration saved to {filepath}")

        except Exception as e:
            logging.error(f"Failed to save config to {filepath}: {e}")

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as dictionary"""
        return {
            'model': dict(self.model.__dict__),
            'hardware': dict(self.hardware.__dict__),
            'system': dict(self.system.__dict__),
            'security': dict(self.security.__dict__),
            'behavior': dict(self.behavior.__dict__),
            'ai': dict(self.ai.__dict__),
            'web_search': dict(self.web_search.__dict__),
            'music': dict(self.music.__dict__),
            'hand': dict(self.hand.__dict__),
            'microcontroller': dict(self.microcontroller.__dict__),
            'navigation': dict(self.navigation.__dict__),
            'conversation': dict(self.conversation.__dict__),
        }
    
    def validate(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # Check required API keys
        if not self.model.picovoice_access_key:
            logging.warning("Picovoice access key not set - wake word detection will use fallback mode")
        
        # Check hardware ports exist
        if self.hardware.is_esp_connected:
            esp_port = self.hardware.uart_device_map.get("esp32", self.hardware.esp32_port)
            if not Path(esp_port).exists():
                logging.warning(f"ESP32 port {esp_port} not found")
        
        # Check camera availability
        if self.hardware.camera_index < 0:
            errors.append("Invalid camera index")
        
        # Check model files
        if not Path(self.model.wake_word_model_path).parent.exists():
            Path(self.model.wake_word_model_path).parent.mkdir(parents=True, exist_ok=True)
        
        if errors:
            for error in errors:
                logging.error(f"Config validation error: {error}")
            return False
        
        return True


# Global configuration instance
config = RobotConfig()

# Export configuration sections for easy access
model_config = config.model
hardware_config = config.hardware
system_config = config.system
security_config = config.security
behavior_config = config.behavior
ai_config = config.ai
web_search_config = config.web_search
music_config = config.music
hand_config = config.hand
microcontroller_config = config.microcontroller
navigation_config = config.navigation
conversation_config = config.conversation


if __name__ == "__main__":
    """Test configuration loading and validation"""
    
    # Print current configuration
    print("Robot Configuration")
    print("=" * 50)
    
    for section_name, section_config in config.get_all_settings().items():
        print(f"\n{section_name.upper()} Configuration:")
        for key, value in section_config.items():
            print(f"  {key}: {value}")
    
    # Validate configuration
    if config.validate():
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration has errors")
    
    # Save example configuration
    config.save_to_file("config_example.json")