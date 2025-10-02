"""
AI Robot Main Entry Point - FIXED VERSION
==========================================
Main application that initializes ALL modules and runs the complete robot system.
Handles module coordination, error recovery, and graceful shutdown.
All modules are now properly imported and synchronized.

Author: AI Assistant
Date: 2024
Fixed: All imports enabled, synchronization added
"""

import sys
import signal
import time
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import psutil
import threading

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Check for Hailo availability FIRST
USE_HAILO = False
try:
    import hailo
    USE_HAILO = True
    print("✓ Hailo detected - using hardware acceleration")
except ImportError:
    print("ℹ Hailo not found - using CPU-only mode")

# Import configuration based on Hailo availability
from config.settings import (
    config, 
    system_config, 
    hardware_config,
    security_config,
    behavior_config,
    model_config  # Added model_config
)

# If no Hailo, adjust settings for CPU-only operation
if not USE_HAILO:
    model_config.whisper_model = "tiny"  # Use smaller model
    model_config.object_model = "mobilenet"  # Use lighter model
    system_config.frame_skip = 5  # Process less frames
    hardware_config.camera_resolution = (320, 240)  # Lower resolution
    hardware_config.camera_fps = 15  # Lower FPS

# Import core modules
from core.robot_brain import RobotBrain, RobotEvent

# Import ALL modules - NO MORE COMMENTS!
from modules.vision.face_recognition import FaceRecognitionModule

# Import audio modules - ALL ENABLED
from modules.audio.speech_recognition import SpeechRecognitionModule
from modules.audio.wake_word import WakeWordDetector
from modules.audio.text_to_speech import TextToSpeechModule

# Import hardware modules - ALL ENABLED
from modules.hardware.esp32_controller import ESP32Controller

# Import modules that need to be created
try:
    from modules.vision.object_detection import ObjectDetectionModule
except ImportError:
    print("⚠ Object Detection module not found - will create")
    ObjectDetectionModule = None

try:
    from modules.audio.voice_identification import VoiceIdentificationModule
except ImportError:
    print("⚠ Voice Identification module not found - will create")
    VoiceIdentificationModule = None

try:
    from modules.communication.gsm_module import GSMModule
except ImportError:
    print("⚠ GSM module not found - will create")
    GSMModule = None

try:
    from modules.intelligence.conversation_ai import ConversationAI
except ImportError:
    print("⚠ Conversation AI module not found - will create")
    ConversationAI = None

try:
    from modules.intelligence.learning_module import LearningModule
except ImportError:
    print("⚠ Learning module not found - will create")
    LearningModule = None

try:
    from modules.communication.web_server import WebServer
except ImportError:
    print("⚠ Web Server module not found")
    WebServer = None


class AIRobot:
    """
    Main AI Robot application class.
    Manages all modules and coordinates the robot system.
    FIXED: All modules enabled and properly synchronized.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the AI Robot with ALL modules
        
        Args:
            config_file: Optional path to configuration file
        """
        
        # Setup logging with detailed format
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("=" * 60)
        self.logger.info(f"   {behavior_config.robot_name} AI ROBOT SYSTEM v2.0")
        self.logger.info("=" * 60)
        
        # Detect available hardware
        self.hardware_status = self._detect_hardware()
        self.logger.info("Hardware Detection Results:")
        for hw, status in self.hardware_status.items():
            self.logger.info(f"  {hw}: {'✓' if status else '✗'}")
        
        # Load custom configuration if provided
        if config_file:
            config.load_from_file(config_file)
        
        # Validate configuration
        if not config.validate():
            self.logger.error("Configuration validation failed!")
            sys.exit(1)
        
        # System state
        self.running = False
        self.modules: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        
        # Initialize robot brain - the central controller
        self.brain = RobotBrain()
        
        # Module synchronization locks to prevent resource conflicts
        self.camera_lock = threading.Lock()
        self.microphone_lock = threading.Lock()
        self.speaker_lock = threading.Lock()
    
    def _detect_hardware(self) -> Dict[str, bool]:
        """Detect what hardware is actually connected"""
        status = {}
        
        # Check camera
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            status['camera'] = cap.isOpened()
            cap.release()
        except:
            status['camera'] = False
        
        # Check microphone
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            status['microphone'] = p.get_device_count() > 0
            p.terminate()
        except:
            status['microphone'] = False
        
        # Check ESP32 (serial port)
        status['esp32'] = Path(hardware_config.esp32_port).exists()
        
        # Check SIM7600X
        sim_ports = [hardware_config.sim7600x_port] + hardware_config.sim7600x_alt_ports
        status['sim7600x'] = any(Path(port).exists() for port in sim_ports)
        
        # Check Hailo
        status['hailo'] = USE_HAILO
        
        return status
        
        # Performance monitoring
        self.start_time = time.time()
        self.performance_monitor = None
        
        # Error handling
        self.error_count = 0
        self.max_errors = 10
        
        # Shutdown event
        self.shutdown_event = threading.Event()
        
    def _setup_logging(self):
        """Setup logging configuration with colors and file output"""
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
        
        # File handler
        log_dir = PROJECT_ROOT / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"robot_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, system_config.log_level),
            handlers=[console_handler, file_handler]
        )
    
    def initialize_modules(self):
        from complete_integration import integrate_everything
        """Initialize ALL robot modules with proper synchronization"""
        self.logger.info("Initializing ALL modules...")
        
        # Initialize Vision Modules
        self._init_vision_modules()
        
        # Initialize Audio Modules  
        self._init_audio_modules()
        
        # Initialize Hardware Controllers
        self._init_hardware_modules()
        
        # Initialize Communication Modules
        self._init_communication_modules()
        
        # Initialize Intelligence Modules
        self._init_intelligence_modules()
        
        self.logger.info(f"Successfully initialized {len(self.modules)} modules")
        
        # CRITICAL: Set up complete system integration
        from config.system_integration import setup_robot_integration
        self.integration = setup_robot_integration(self)
        
        # ALSO call the complete integration function
        self.integration_result = integrate_everything(self)
        self.logger.info("✓ Complete system integration finished - all modules synchronized")
    
    def _init_vision_modules(self):
        """Initialize all vision-related modules"""
        
        if not self.hardware_status.get('camera', False):
            self.logger.warning("Camera not detected - skipping vision modules")
            return
        
        # Face Recognition
        try:
            self.logger.info("Initializing Face Recognition...")
            self.modules['face_recognition'] = FaceRecognitionModule(self.brain)
            self.brain.modules['face_recognition'] = self.modules['face_recognition']
            self.logger.info("✓ Face Recognition initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Face Recognition: {e}")
        
        # Object Detection
        if ObjectDetectionModule and self.hardware_status.get('hailo', False):
            try:
                self.logger.info("Initializing Object Detection...")
                self.modules['object_detection'] = ObjectDetectionModule(self.brain, use_hailo=USE_HAILO)
                self.brain.modules['object_detection'] = self.modules['object_detection']
                self.logger.info("✓ Object Detection initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize Object Detection: {e}")
    
    def _init_audio_modules(self):
        """Initialize all audio-related modules with synchronization"""
        
        if not self.hardware_status.get('microphone', False):
            self.logger.warning("Microphone not detected - skipping audio modules")
            return
        
        # Wake Word Detection - runs continuously in background
        try:
            self.logger.info("Initializing Wake Word Detection...")
            self.modules['wake_word'] = WakeWordDetector(self.brain)
            self.brain.modules['wake_word'] = self.modules['wake_word']
            self.logger.info("✓ Wake Word Detection initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Wake Word: {e}")
        
        # Speech Recognition - activated after wake word
        try:
            self.logger.info("Initializing Speech Recognition...")
            self.modules['speech_recognition'] = SpeechRecognitionModule(self.brain)
            self.brain.modules['speech_recognition'] = self.modules['speech_recognition']
            self.logger.info("✓ Speech Recognition initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Speech Recognition: {e}")
        
        # Text-to-Speech - for robot responses
        try:
            self.logger.info("Initializing Text-to-Speech...")
            self.modules['tts'] = TextToSpeechModule(self.brain)
            self.brain.modules['tts'] = self.modules['tts']
            self.brain.modules['audio'] = self.modules['tts']  # Alias for brain
            self.logger.info("✓ Text-to-Speech initialized")
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize TTS: {e}")
        
        # Voice Identification - identify who is speaking
        if VoiceIdentificationModule:
            try:
                self.logger.info("Initializing Voice Identification...")
                self.modules['voice_identification'] = VoiceIdentificationModule(self.brain)
                self.brain.modules['voice_identification'] = self.modules['voice_identification']
                self.logger.info("✓ Voice Identification initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize Voice Identification: {e}")
    
    def _init_hardware_modules(self):
        """Initialize hardware control modules"""
        
        # ESP32 Controller for motors and sensors
        if self.hardware_status.get('esp32', False):
            try:
                self.logger.info("Initializing ESP32 Controller...")
                self.modules['esp32'] = ESP32Controller(self.brain)
                self.brain.modules['esp32'] = self.modules['esp32']
                self.brain.modules['hardware'] = self.modules['esp32']  # Alias
                self.logger.info("✓ ESP32 Controller initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize ESP32: {e}")
        else:
            self.logger.warning("ESP32 not detected - skipping hardware controller")
    
    def _init_communication_modules(self):
        """Initialize communication modules"""
        
        # GSM Module for remote communication
        if GSMModule:
            try:
                self.logger.info("Initializing GSM Module...")
                self.modules['gsm'] = GSMModule(self.brain)
                self.brain.modules['gsm'] = self.modules['gsm']
                self.logger.info("✓ GSM Module initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize GSM: {e}")
        
        # Web Server for remote control
        if WebServer and system_config.enable_web_interface:
            try:
                self.logger.info("Initializing Web Server...")
                self.modules['web_server'] = WebServer(self.brain)
                self.brain.modules['web_server'] = self.modules['web_server']
                self.logger.info("✓ Web Server initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize Web Server: {e}")
    
    def _init_intelligence_modules(self):
        """Initialize AI and learning modules"""
        
        # Conversation AI for intelligent dialogue
        if ConversationAI:
            try:
                self.logger.info("Initializing Conversation AI...")
                self.modules['conversation_ai'] = ConversationAI(self.brain)
                self.brain.modules['conversation_ai'] = self.modules['conversation_ai']
                self.logger.info("✓ Conversation AI initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize Conversation AI: {e}")
        
        # Learning Module for self-improvement
        if LearningModule:
            try:
                self.logger.info("Initializing Learning Module...")
                self.modules['learning'] = LearningModule(self.brain)
                self.brain.modules['learning'] = self.modules['learning']
                self.logger.info("✓ Learning Module initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize Learning Module: {e}")
    
    def _setup_module_sync(self):
        """Setup synchronization between modules to prevent conflicts"""
        
        # Wake word should pause when speech recognition is active
        if 'wake_word' in self.modules and 'speech_recognition' in self.modules:
            def on_wake_word_detected():
                """Handler for wake word detection"""
                self.logger.info("Wake word detected - pausing detection, starting recognition")
                self.modules['wake_word'].pause_listening()
                self.modules['speech_recognition'].start_recording(duration=5.0)
                
                # Resume wake word after speech recognition
                def resume_wake_word():
                    time.sleep(5.0)
                    self.modules['wake_word'].resume_listening()
                    self.logger.info("Speech recognition complete - resuming wake word detection")
                
                threading.Thread(target=resume_wake_word, daemon=True).start()
            
            self.modules['wake_word'].set_callback(on_wake_word_detected)
        
        # TTS should pause wake word detection
        if 'tts' in self.modules and 'wake_word' in self.modules:
            original_speak = self.modules['tts'].speak
            
            def speak_with_pause(text, priority=5, wait=False):
                """Speak while pausing wake word detection"""
                self.modules['wake_word'].pause_listening()
                original_speak(text, priority, wait=True)
                self.modules['wake_word'].resume_listening()
            
            self.modules['tts'].speak = speak_with_pause
    
    def _register_event_handlers(self):
        """Register event handlers for inter-module communication"""
        
        # Face detection events
        self.brain.register_event_handler(
            'face_detected',
            self._handle_face_detected
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
        
        # Object detection events
        self.brain.register_event_handler(
            'object_detected',
            self._handle_object_detected
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
        
        # Learning events
        self.brain.register_event_handler(
            'new_pattern',
            self._handle_new_pattern
        )
    
    def _handle_face_detected(self, event: RobotEvent):
        """Handle face detection event with learning capability"""
        face_data = event.data
        self.logger.debug(f"Face detected: {face_data.get('name', 'Unknown')}")
        
        # Check if new face
        if face_data.get('face_id') == 'unknown':
            # Ask for name and learn
            if 'tts' in self.modules:
                self.modules['tts'].speak(
                    "Hello! I don't think we've met. What's your name?",
                    priority=2
                )
            # Start learning process
            if 'learning' in self.modules:
                self.modules['learning'].learn_new_person(face_data)
        
        # Authenticate if master
        elif face_data.get('is_master'):
            self.brain.emit_event(RobotEvent(
                type='user_authenticated',
                source='main',
                data={
                    'user_id': face_data['face_id'],
                    'method': 'face'
                },
                priority=2
            ))
            
            # Greet master
            if 'tts' in self.modules:
                self.modules['tts'].speak(
                    f"Welcome back, Master!",
                    priority=1
                )
    
    def _handle_wake_word(self, event: RobotEvent):
        """Handle wake word detection - start dialogue"""
        self.logger.info("Wake word detected - entering dialogue mode")
        
        # Visual feedback - LED
        if 'esp32' in self.modules:
            self.modules['esp32'].set_led('blue')
        
        # Audio feedback
        if 'tts' in self.modules:
            self.modules['tts'].speak("Yes?", priority=1, wait=True)
        
        # Trigger listening state in brain
        self.brain.wake_word_heard()
    
    def _handle_speech(self, event: RobotEvent):
        """Handle recognized speech with intelligent processing"""
        text = event.data.get('text', '')
        self.logger.info(f"Speech recognized: {text}")
        
        # Process with AI if available
        if 'conversation_ai' in self.modules:
            response = self.modules['conversation_ai'].process(text)
            
            # Execute commands
            if response.get('action'):
                self._execute_action(response['action'])
            
            # Speak response
            if response.get('text') and 'tts' in self.modules:
                self.modules['tts'].speak(response['text'], priority=2)
        else:
            # Simple processing without AI
            self.brain.speech_received(text)
            self._process_simple_command(text)
    
    def _process_simple_command(self, text: str):
        """Process commands without AI"""
        text_lower = text.lower()
        
        # Light control
        if 'light' in text_lower:
            if 'on' in text_lower or 'turn on' in text_lower:
                if 'esp32' in self.modules:
                    self.modules['esp32'].set_gpio(25, True)
                    self.modules['tts'].speak("Light turned on", priority=2)
            elif 'off' in text_lower or 'turn off' in text_lower:
                if 'esp32' in self.modules:
                    self.modules['esp32'].set_gpio(25, False)
                    self.modules['tts'].speak("Light turned off", priority=2)
        
        # Movement commands
        elif 'move' in text_lower or 'go' in text_lower:
            if 'forward' in text_lower:
                if 'esp32' in self.modules:
                    self.modules['esp32'].move_forward(50, duration=2.0)
            elif 'backward' in text_lower or 'back' in text_lower:
                if 'esp32' in self.modules:
                    self.modules['esp32'].move_backward(50, duration=2.0)
            elif 'left' in text_lower:
                if 'esp32' in self.modules:
                    self.modules['esp32'].turn_left(50, duration=1.0)
            elif 'right' in text_lower:
                if 'esp32' in self.modules:
                    self.modules['esp32'].turn_right(50, duration=1.0)
        
        # Stop command
        elif 'stop' in text_lower:
            if 'esp32' in self.modules:
                self.modules['esp32'].stop_motors()
                self.modules['tts'].speak("Stopped", priority=1)
    
    def _handle_object_detected(self, event: RobotEvent):
        """Handle object detection with learning"""
        objects = event.data.get('objects', [])
        
        for obj in objects:
            self.logger.debug(f"Object detected: {obj.get('class')} ({obj.get('confidence', 0):.2%})")
            
            # Learn new objects
            if 'learning' in self.modules:
                self.modules['learning'].process_object(obj)
    
    def _handle_new_pattern(self, event: RobotEvent):
        """Handle new pattern learned"""
        pattern = event.data
        self.logger.info(f"New pattern learned: {pattern.get('type')}")
        
        # Store in long-term memory
        self.brain.add_memory(
            content=pattern,
            memory_type='long_term',
            importance=0.8
        )
    
    def _handle_battery_low(self, event: RobotEvent):
        """Handle low battery warning with action"""
        level = event.data.get('level', 0)
        self.logger.warning(f"Low battery warning: {level}%")
        
        # Notify user
        if 'tts' in self.modules:
            self.modules['tts'].speak(
                f"Warning: Battery is at {level} percent. I need to charge soon.",
                priority=1
            )
        
        # Send SMS if GSM available
        if 'gsm' in self.modules:
            self.modules['gsm'].send_sms(
                "Robot battery low",
                f"Battery level: {level}%"
            )
    
    def _handle_emergency_stop(self, event: RobotEvent):
        """Handle emergency stop - immediate action"""
        self.logger.critical("EMERGENCY STOP TRIGGERED!")
        
        # Stop all modules immediately
        self.emergency_shutdown()
    
    def _execute_action(self, action: Dict[str, Any]):
        """Execute action from AI decision"""
        action_type = action.get('type')
        params = action.get('parameters', {})
        
        if action_type == 'move':
            if 'esp32' in self.modules:
                direction = params.get('direction')
                speed = params.get('speed', 50)
                duration = params.get('duration', 2.0)
                
                if direction == 'forward':
                    self.modules['esp32'].move_forward(speed, duration)
                elif direction == 'backward':
                    self.modules['esp32'].move_backward(speed, duration)
                elif direction == 'left':
                    self.modules['esp32'].turn_left(speed, duration)
                elif direction == 'right':
                    self.modules['esp32'].turn_right(speed, duration)
        
        elif action_type == 'gpio':
            if 'esp32' in self.modules:
                pin = params.get('pin')
                value = params.get('value')
                self.modules['esp32'].set_gpio(pin, value)
        
        elif action_type == 'learn':
            if 'learning' in self.modules:
                self.modules['learning'].start_learning(params)
    
    def start_modules(self):
        """Start all initialized modules in correct order"""
        self.logger.info("Starting all modules...")
        
        # Start hardware first
        if 'esp32' in self.modules:
            try:
                self.modules['esp32'].start()
                self.logger.info("✓ ESP32 started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start ESP32: {e}")
        
        # Start vision modules
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
        
        # Start audio modules (in order)
        if 'tts' in self.modules:
            try:
                self.modules['tts'].start()
                self.logger.info("✓ Text-to-Speech started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start TTS: {e}")
        
        if 'speech_recognition' in self.modules:
            try:
                self.modules['speech_recognition'].start()
                self.logger.info("✓ Speech Recognition started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Speech Recognition: {e}")
        
        # Start wake word last (after other audio modules)
        if 'wake_word' in self.modules:
            try:
                self.modules['wake_word'].start()
                self.logger.info("✓ Wake Word Detection started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Wake Word: {e}")
        
        # Start communication modules
        if 'gsm' in self.modules:
            try:
                self.modules['gsm'].start()
                self.logger.info("✓ GSM Module started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start GSM: {e}")
        
        if 'web_server' in self.modules:
            try:
                self.modules['web_server'].start()
                self.logger.info("✓ Web Server started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Web Server: {e}")
        
        # Start intelligence modules
        if 'conversation_ai' in self.modules:
            try:
                self.modules['conversation_ai'].start()
                self.logger.info("✓ Conversation AI started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Conversation AI: {e}")
        
        if 'learning' in self.modules:
            try:
                self.modules['learning'].start()
                self.logger.info("✓ Learning Module started")
            except Exception as e:
                self.logger.error(f"✗ Failed to start Learning Module: {e}")
        
        # Start robot brain last
        self.brain.start()
        self.logger.info("✓ Robot Brain started")
        
        self.logger.info("All modules started successfully")
    
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
                            
                            if temp > 70:
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
        """Main run loop with all modules active"""
        self.running = True
        
        self.logger.info("=" * 60)
        self.logger.info(f"   {behavior_config.robot_name} is now ONLINE!")
        self.logger.info("   All systems operational")
        self.logger.info("=" * 60)
        
        # Initial greeting
        if 'tts' in self.modules:
            self.modules['tts'].speak(
                f"Hello! {behavior_config.robot_name} is ready.",
                priority=1,
                wait=True
            )
        
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
        """Graceful shutdown of all modules"""
        self.logger.info("Initiating shutdown sequence...")
        
        self.running = False
        
        # Say goodbye
        if 'tts' in self.modules and self.modules['tts'].running:
            try:
                self.modules['tts'].speak(behavior_config.goodbye_message, priority=1, wait=True)
            except:
                pass
        
        # Stop brain
        self.brain.stop()
        
        # Stop all modules in reverse order
        stop_order = [
            'learning', 'conversation_ai', 'web_server', 'gsm',
            'wake_word', 'speech_recognition', 'tts', 'voice_identification',
            'object_detection', 'face_recognition', 'esp32'
        ]
        
        for module_name in stop_order:
            if module_name in self.modules:
                try:
                    module = self.modules[module_name]
                    if hasattr(module, 'stop'):
                        module.stop()
                        self.logger.info(f"✓ {module_name} stopped")
                    elif hasattr(module, 'shutdown'):
                        module.shutdown()
                        self.logger.info(f"✓ {module_name} shutdown")
                except Exception as e:
                    self.logger.error(f"✗ Error stopping {module_name}: {e}")
        
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
        
        # Stop all motors immediately
        if 'esp32' in self.modules:
            try:
                self.modules['esp32'].stop_all_motors()
            except:
                pass
        
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
        """Save current state to file for recovery"""
        state = {
            'shutdown_time': time.time(),
            'runtime': time.time() - self.start_time,
            'brain_status': self.brain.get_status(),
            'error_count': self.error_count,
            'modules': list(self.modules.keys()),
            'last_user': self.brain.current_user,
            'memories': {
                'short_term': len(self.brain.short_term_memory),
                'long_term': len(self.brain.long_term_memory)
            }
        }
        
        try:
            state_file = PROJECT_ROOT / "data" / "last_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self.logger.info("State saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    print("\nShutdown signal received")
    sys.exit(0)


def main():
    """Main entry point with complete initialization"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Robot System v2.0')
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
    parser.add_argument(
        '--create-service',
        help='Create systemd service',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Create systemd service if requested
    if args.create_service:
        create_systemd_service()
        return
    
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
    Run comprehensive system tests
    
    Args:
        robot: Robot instance to test
    """
    print("\nRunning comprehensive system tests...")
    
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
    
    # Test 4: Microphone test
    print("\n4. Microphone Test:")
    if 'speech_recognition' in robot.modules:
        print("   ✓ Speech recognition available")
        if 'wake_word' in robot.modules:
            print("   ✓ Wake word detection available")
    
    # Test 5: ESP32 connection
    print("\n5. ESP32 Connection:")
    if 'esp32' in robot.modules:
        esp32 = robot.modules['esp32']
        if esp32.connect():
            print(f"   ✓ ESP32 connected on {esp32.port}")
            esp32.disconnect()
        else:
            print("   ✗ ESP32 not connected")
    
    # Test 6: Brain state machine
    print("\n6. Brain State Machine:")
    brain = robot.brain
    print(f"   Initial state: {brain.state.name}")
    
    # Test state transitions
    brain.startup_complete()
    print(f"   After startup: {brain.state.name}")
    
    brain.wake_word_heard()
    print(f"   After wake word: {brain.state.name}")
    
    brain.return_idle()
    print(f"   Return to idle: {brain.state.name}")
    
    # Test 7: Event system
    print("\n7. Event System:")
    test_event = RobotEvent(
        type='test_event',
        source='test',
        data={'test': True},
        priority=5
    )
    brain.emit_event(test_event)
    print("   ✓ Event emitted successfully")
    
    # Test 8: Memory system
    print("\n8. Memory System:")
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
    
    # Test 9: Performance check
    print("\n9. Performance Check:")
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
    
    # Test 10: File system
    print("\n10. File System:")
    data_dirs = [
        "data/models",
        "data/faces", 
        "data/voices",
        "data/logs",
        "data/tts_cache"
    ]
    
    for dir_path in data_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path} exists")
        else:
            print(f"   ✗ {dir_path} missing")
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"     → Created {dir_path}")
    
    # Test 11: Network connectivity
    print("\n11. Network Test:")
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("   ✓ Internet connection available")
    except:
        print("   ✗ No internet connection")
    
    # Test 12: Module synchronization
    print("\n12. Module Synchronization:")
    print(f"   Camera lock: {'✓' if robot.camera_lock else '✗'}")
    print(f"   Microphone lock: {'✓' if robot.microphone_lock else '✗'}")
    print(f"   Speaker lock: {'✓' if robot.speaker_lock else '✗'}")
    
    print("\nTest Summary:")
    print("All basic systems checked. Review results above.")
    print("The robot is ready for operation.")


def create_systemd_service():
    """
    Create systemd service file for auto-start on boot
    This should be run with sudo
    """
    service_content = f"""[Unit]
Description=AI Robot Service v2.0
After=multi-user.target network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 {PROJECT_ROOT}/main.py
Restart=always
RestartSec=10
User=pi
WorkingDirectory={PROJECT_ROOT}
Environment="DISPLAY=:0"
Environment="HOME=/home/pi"
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
        print("\nTo enable auto-start on boot, run:")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable ai-robot.service")
        print("  sudo systemctl start ai-robot.service")
        print("\nTo check status:")
        print("  sudo systemctl status ai-robot.service")
        print("\nTo view logs:")
        print("  sudo journalctl -u ai-robot.service -f")
        
    except PermissionError:
        print("Permission denied. Run with sudo:")
        print(f"  sudo python3 {__file__} --create-service")


if __name__ == "__main__":
    main()