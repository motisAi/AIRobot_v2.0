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
import psutil
import threading

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration — platform detection and Hailo availability are
# handled inside config.settings automatically.
from config.settings import (
    config,
    system_config,
    hardware_config,
    security_config,
    behavior_config,
)

# Import core modules
from core.robot_brain import RobotBrain, RobotEvent

# Shared hardware managers
from modules.hardware.camera_manager import CameraManager
from modules.hardware.audio_manager import AudioManager

# Vision modules
from modules.vision.face_recognition import FaceRecognitionModule
from modules.vision.object_detection import ObjectDetectionModule

# Audio modules
from modules.audio.wake_word import WakeWordModule
from modules.audio.speech_recognition import SpeechRecognitionModule
from modules.audio.text_to_speech import TextToSpeechModule

# AI modules
from modules.ai.ai_engine import AIEngine
from modules.ai.learning_db import LearningDB

# Hardware controllers
from modules.hardware.esp32_controller import ESP32Controller

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
        
        # Load custom configuration if provided, else try default config.json
        default_config = PROJECT_ROOT / "config" / "config.json"
        if config_file:
            config.load_from_file(config_file)
        elif default_config.exists():
            config.load_from_file(str(default_config))
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
        
        # File handler
        log_dir = PROJECT_ROOT / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"robot_{time.strftime('%Y%m%d_%H%M%S')}.log"
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
        
        # --- Hardware Controllers ---
        if hardware_config.is_esp_connected:
            try:
                self.logger.info("Initializing ESP32 Controller...")
                self.modules['esp32'] = ESP32Controller(self.brain)
                self.brain.modules['hardware'] = self.modules['esp32']
                self.logger.info("✓ ESP32 Controller initialized")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize ESP32: {e}")
        else:
            self.logger.info("ESP32 flagged as disconnected – simulation mode enabled")
        
        # --- Web Dashboard ---
        try:
            self.logger.info("Starting Web Dashboard...")
            self.dashboard = WebDashboard(
                camera_manager=self.camera_manager,
                brain=self.brain,
                learning_db=self.learning_db,
                robot_name=behavior_config.robot_name,
            )
            self.dashboard.start()
            self.logger.info("✓ Web Dashboard at http://0.0.0.0:5000")
        except Exception as e:
            self.logger.error(f"✗ Failed to start Web Dashboard: {e}")
            self.dashboard = None
        
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
        self.logger.debug(f"Face detected: {face_data.get('name', 'Unknown')}")
        
        # Authenticate if master
        if face_data.get('is_master'):
            self.brain.emit_event(RobotEvent(
                type='user_authenticated',
                source='main',
                data={
                    'user_id': face_data['face_id'],
                    'method': 'face'
                },
                priority=2
            ))
    
    def _handle_speak(self, event: RobotEvent):
        """Route speak events to the TTS module."""
        text = event.data.get('text', '') if event.data else ''
        if not text:
            return
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
        """Handle wake word detection"""
        self.logger.info("Wake word detected — pausing passive listener and starting dialogue")
        self._pause_wake_word_listener(reason="wake_word")
        
        # Trigger listening state in brain
        self.brain.wake_word_heard()
        recognizer = self.modules.get('speech_recognition')
        if recognizer:
            started = recognizer.listen_for_command(timeout=system_config.conversation_timeout)
            if not started:
                self.logger.warning("Speech recognizer busy; resuming wake word listener")
                self._resume_wake_word_listener()
        else:
            self.logger.warning("Speech recognition module not available — resuming wake word listener")
            self._resume_wake_word_listener()
    
    def _handle_speech(self, event: RobotEvent):
        """Handle recognized speech"""
        text = event.data.get('text', '')
        self.logger.info(f"Speech recognized: {text}")
        
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
    
    def start_modules(self):
        """Start all initialized modules"""
        self.logger.info("Starting modules...")
        
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
        
        if 'esp32' in self.modules:
            try:
                self.modules['esp32'].connect()
                self.logger.info("✓ ESP32 Controller connected")
            except Exception as e:
                self.logger.error(f"✗ Failed to connect ESP32: {e}")
        
        # Start robot brain
        self.brain.start()
        self.logger.info("✓ Robot Brain started")
        
        self.logger.info("All modules started")
    
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
    
    # Check for GSM
    gsm_port = Path(hardware_config.gsm_port)
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
    service_content = f"""[Unit]
Description=AI Robot Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 {PROJECT_ROOT}/main.py
Restart=always
User=pi
WorkingDirectory={PROJECT_ROOT}
Environment=DISPLAY=:0
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