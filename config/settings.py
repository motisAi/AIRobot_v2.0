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

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


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
    
    # Speech Recognition (Whisper)
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_language: str = "en"
    whisper_device: str = "cpu"  # cuda if available
    whisper_compute_type: str = "int8"  # int8 for speed, float16 for accuracy
    whisper_beam_size: int = 5
    whisper_patience: float = 1.0
    
    # Text-to-Speech (Coqui TTS)
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    tts_vocoder: str = "vocoder_models/en/ljspeech/hifigan_v2"
    tts_speaker_wav: Optional[str] = None  # Path to speaker voice sample
    tts_language: str = "en"
    tts_speed: float = 1.0  # Speech speed multiplier
    
    # Wake Word Detection (Picovoice Porcupine)
    wake_word: str = "hey robot"
    wake_word_sensitivity: float = 0.5  # 0-1, higher = more sensitive
    wake_word_model_path: str = str(PROJECT_ROOT / "data" / "models" / "wake_word.ppn")
    picovoice_access_key: str = os.getenv("PICOVOICE_ACCESS_KEY", "")
    
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
    
    # Microphone Settings
    microphone_device_index: Optional[int] = None  # None = default device
    microphone_channels: int = 1
    microphone_rate: int = 16000
    microphone_chunk: int = 1024
    microphone_timeout: float = 0.8
    microphone_phrase_time_limit: float = 5.0
    
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
    sim7600x_apn: str = "sphone.pelephone.net.il"  # Pelephone APN
    sim7600x_username: str = "pcl@3g"  # Pelephone username
    sim7600x_password: str = "pcl"  # Pelephone password
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


@dataclass
class SystemConfig:
    """System-level configurations"""
    
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


@dataclass
class BehaviorConfig:
    """Robot behavior and personality settings"""
    
    # Personality
    robot_name: str = os.getenv("ROBOT_NAME", "RoboAI")
    personality_type: str = "helpful"  # helpful, playful, professional
    response_style: str = "concise"  # concise, detailed, chatty
    
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


class RobotConfig:
    """Main configuration class that combines all settings"""
    
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
        
        # Create necessary directories
        self._create_directories()
        
        # Load custom config if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Setup logging
        self._setup_logging()
    
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
    
    def load_from_file(self, filepath: str):
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to JSON configuration file
        """
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            logging.info(f"Configuration loaded from {filepath}")
        
        except Exception as e:
            logging.error(f"Failed to load config from {filepath}: {e}")
    
    def save_to_file(self, filepath: str):
        """
        Save current configuration to JSON file
        
        Args:
            filepath: Path to save JSON configuration
        """
        config_data = {
            'model': self.model.__dict__,
            'hardware': self.hardware.__dict__,
            'system': self.system.__dict__,
            'security': self.security.__dict__,
            'behavior': self.behavior.__dict__
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=4, default=str)
            
            logging.info(f"Configuration saved to {filepath}")
        
        except Exception as e:
            logging.error(f"Failed to save config to {filepath}: {e}")
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as dictionary"""
        return {
            'model': self.model.__dict__,
            'hardware': self.hardware.__dict__,
            'system': self.system.__dict__,
            'security': self.security.__dict__,
            'behavior': self.behavior.__dict__
        }
    
    def validate(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # Check required API keys
        if self.model.picovoice_access_key == "":
            errors.append("Picovoice access key not set")
        
        # Check hardware ports exist
        if not Path(self.hardware.esp32_port).exists():
            logging.warning(f"ESP32 port {self.hardware.esp32_port} not found")
        
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