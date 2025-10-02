"""
System Integration and Flow Control
====================================
This file defines the complete robot behavior flow and module coordination.
Ensures all modules work together without conflicts.

Author: AI Assistant
Date: 2024
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import os

# Import all modules properly
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from core.robot_brain import RobotBrain, RobotEvent

# Import ALL modules - no try/except, they must exist
from modules.vision.face_recognition import FaceRecognitionModule
from modules.vision.object_detection import ObjectDetectionModule
from modules.audio.wake_word import WakeWordDetector
from modules.audio.speech_recognition import SpeechRecognitionModule
from modules.audio.text_to_speech import TextToSpeechModule
from modules.audio.voice_identification import VoiceIdentificationModule
from modules.hardware.esp32_controller import ESP32Controller
from modules.intelligence.conversation_ai import ConversationAI
from modules.intelligence.learning_module import LearningModule

# Import configuration
from config.settings import config, behavior_config


class RobotSystemIntegration:
    """
    Complete system integration that ensures all modules work together.
    This is the ACTUAL flow of how the robot responds to everything.
    """
    
    def __init__(self, robot_instance):
        """
        Initialize system integration
        
        Args:
            robot_instance: The main AIRobot instance
        """
        
        self.robot = robot_instance
        self.brain = robot_instance.brain
        self.modules = robot_instance.modules
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Resource locks to prevent conflicts
        self.microphone_lock = threading.Lock()
        self.speaker_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        
        # State tracking
        self.conversation_active = False
        self.current_speaker = None
        self.last_command_time = 0
        
    def setup_complete_flow(self):
        """
        Set up the complete interaction flow between all modules.
        This is the MASTER FLOW that controls everything.
        """
        
        self.logger.info("Setting up complete system flow...")
        
        # 1. WAKE WORD → CONVERSATION FLOW
        self._setup_wake_word_flow()
        
        # 2. FACE → AUTHENTICATION FLOW
        self._setup_face_recognition_flow()
        
        # 3. SPEECH → ACTION FLOW
        self._setup_speech_to_action_flow()
        
        # 4. LEARNING INTEGRATION
        self._setup_learning_flow()
        
        # 5. CONTINUOUS MONITORING
        self._setup_monitoring_flow()
        
        self.logger.info("✓ Complete system flow configured")
    
    def _setup_wake_word_flow(self):
        """
        Wake word detection → Full conversation flow
        """
        
        if 'wake_word' not in self.modules:
            return
        
        def on_wake_word_detected():
            """
            COMPLETE FLOW when "Hey Robot" is detected
            """
            
            self.logger.info("🎤 Wake word detected - Starting interaction")
            
            # 1. PAUSE wake word detection (prevent re-triggering)
            self.modules['wake_word'].pause_listening()
            
            # 2. VISUAL feedback
            if 'esp32' in self.modules:
                self.modules['esp32'].set_led('blue')  # Blue = listening
            
            # 3. AUDIO feedback
            if 'tts' in self.modules:
                with self.speaker_lock:
                    self.modules['tts'].speak("Yes?", priority=1, wait=True)
            
            # 4. IDENTIFY speaker by voice
            current_speaker = None
            if 'voice_identification' in self.modules:
                # Get last audio sample from wake word
                # (In practice, would capture next speech)
                current_speaker = self.modules['voice_identification'].current_speaker
                
                if current_speaker and current_speaker != 'unknown':
                    self.current_speaker = current_speaker
                    self.logger.info(f"Speaker identified: {current_speaker}")
            
            # 5. START recording for command
            if 'speech_recognition' in self.modules:
                with self.microphone_lock:
                    self.modules['speech_recognition'].start_recording(duration=5.0)
                    
                    # Wait for recording
                    time.sleep(5.5)
                    
                    # Get recognized text
                    text = self.modules['speech_recognition'].stop_recording()
                    
                    if text:
                        self.logger.info(f"Heard: '{text}'")
                        
                        # 6. PROCESS with AI
                        self._process_speech_with_ai(text, current_speaker)
                    else:
                        # No speech detected
                        if 'tts' in self.modules:
                            self.modules['tts'].speak(
                                "I didn't hear anything. Please try again.",
                                priority=2
                            )
            
            # 7. RESUME wake word detection after delay
            threading.Timer(2.0, lambda: self.modules['wake_word'].resume_listening()).start()
            
            # 8. LED feedback complete
            if 'esp32' in self.modules:
                self.modules['esp32'].set_led('green')  # Green = ready
                threading.Timer(2.0, lambda: self.modules['esp32'].set_led('off')).start()
        
        # Register the callback
        self.modules['wake_word'].set_callback(on_wake_word_detected)
    
    def _setup_face_recognition_flow(self):
        """
        Face detection → Authentication → Personalization flow
        """
        
        def on_face_detected(event: RobotEvent):
            """
            COMPLETE FLOW when a face is detected
            """
            
            face_data = event.data
            face_id = face_data.get('face_id', 'unknown')
            name = face_data.get('name', 'Unknown')
            is_master = face_data.get('is_master', False)
            
            # 1. NEW person detected
            if face_id == 'unknown':
                # Learn new person
                if 'learning' in self.modules and behavior_config.learn_new_faces:
                    # Ask for name
                    if 'tts' in self.modules:
                        self.modules['tts'].speak(
                            "Hello! I don't think we've met. What's your name?",
                            priority=2,
                            wait=True
                        )
                    
                    # Record response
                    if 'speech_recognition' in self.modules:
                        with self.microphone_lock:
                            self.modules['speech_recognition'].start_recording(duration=3.0)
                            time.sleep(3.5)
                            name = self.modules['speech_recognition'].stop_recording()
                            
                            if name:
                                # Learn the face
                                self.modules['face_recognition'].learn_face(name)
                                
                                # Learn in learning module
                                self.modules['learning'].learn_new_person({
                                    'id': f"person_{time.time()}",
                                    'name': name,
                                    'face_features': face_data
                                })
                                
                                # Greet
                                self.modules['tts'].speak(
                                    f"Nice to meet you, {name}! I'll remember you.",
                                    priority=1
                                )
            
            # 2. KNOWN person detected
            else:
                # Authenticate
                self.brain.current_user = face_id
                self.brain.authenticated = True
                
                # Master user special treatment
                if is_master:
                    self.brain.master_mode = True
                    
                    # Special greeting for master
                    if 'tts' in self.modules and time.time() - self.last_command_time > 300:
                        self.modules['tts'].speak(
                            f"Welcome back, Master {name}!",
                            priority=2
                        )
                
                # Load user preferences from learning
                if 'learning' in self.modules:
                    preferences = self.modules['learning'].get_user_preferences(face_id)
                    
                    # Apply preferences
                    if preferences:
                        if 'volume' in preferences and 'tts' in self.modules:
                            volume = 0.9 if preferences['volume'] == 'loud' else 0.5
                            self.modules['tts'].set_voice_settings(volume=volume)
                        
                        if 'speed' in preferences and 'esp32' in self.modules:
                            # Adjust robot speed preference
                            speed = 70 if preferences['speed'] == 'fast' else 40
                            self.modules['esp32'].motor_state['speed'] = speed
                
                # Update learning module
                if 'learning' in self.modules:
                    self.modules['learning'].learn_from_interaction({
                        'type': 'face_recognition',
                        'user_id': face_id,
                        'time': time.time()
                    })
        
        # Register handler
        self.brain.register_event_handler('face_detected', on_face_detected)
    
    def _setup_speech_to_action_flow(self):
        """
        Speech recognition → AI processing → Action execution flow
        """
        
        def on_speech_recognized(event: RobotEvent):
            """
            This is already handled in _process_speech_with_ai
            """
            pass
        
        self.brain.register_event_handler('speech_recognized', on_speech_recognized)
    
    def _process_speech_with_ai(self, text: str, speaker: Optional[str] = None):
        """
        Process speech through AI and execute actions.
        This is the CORE decision-making flow.
        """
        
        self.logger.info(f"Processing: '{text}' from {speaker or 'unknown'}")
        
        # 1. CONVERSATION AI processing
        response = None
        if 'conversation_ai' in self.modules:
            response = self.modules['conversation_ai'].process(
                text,
                user_id=speaker or 'default'
            )
        
        if not response:
            # Fallback to simple processing
            response = self._simple_command_processing(text)
        
        # 2. EXECUTE action if needed
        if response.get('action'):
            self._execute_action(response['action'])
        
        # 3. SPEAK response
        if response.get('text') and 'tts' in self.modules:
            with self.speaker_lock:
                self.modules['tts'].speak(response['text'], priority=2)
        
        # 4. LEARN from interaction
        if 'learning' in self.modules:
            self.modules['learning'].learn_from_interaction({
                'type': 'conversation',
                'user_id': speaker or 'default',
                'input': text,
                'response': response,
                'timestamp': time.time()
            })
        
        self.last_command_time = time.time()
    
    def _simple_command_processing(self, text: str) -> Dict[str, Any]:
        """
        Simple command processing without AI
        """
        
        text_lower = text.lower()
        response = {'text': '', 'action': None}
        
        # MOVEMENT commands
        if 'move' in text_lower or 'go' in text_lower:
            if 'forward' in text_lower:
                response['action'] = {
                    'type': 'move',
                    'parameters': {'direction': 'forward', 'duration': 2.0}
                }
                response['text'] = "Moving forward"
            elif 'back' in text_lower:
                response['action'] = {
                    'type': 'move',
                    'parameters': {'direction': 'backward', 'duration': 2.0}
                }
                response['text'] = "Moving backward"
            elif 'left' in text_lower:
                response['action'] = {
                    'type': 'move',
                    'parameters': {'direction': 'left', 'duration': 1.0}
                }
                response['text'] = "Turning left"
            elif 'right' in text_lower:
                response['action'] = {
                    'type': 'move',
                    'parameters': {'direction': 'right', 'duration': 1.0}
                }
                response['text'] = "Turning right"
        
        # STOP command
        elif 'stop' in text_lower:
            response['action'] = {'type': 'stop'}
            response['text'] = "Stopping"
        
        # LIGHT control
        elif 'light' in text_lower:
            if 'on' in text_lower:
                response['action'] = {
                    'type': 'gpio',
                    'parameters': {'pin': 25, 'value': True}
                }
                response['text'] = "Turning light on"
            elif 'off' in text_lower:
                response['action'] = {
                    'type': 'gpio',
                    'parameters': {'pin': 25, 'value': False}
                }
                response['text'] = "Turning light off"
        
        # FIND object
        elif 'find' in text_lower or 'look for' in text_lower:
            # Extract object name (simple approach)
            words = text_lower.split()
            if 'find' in words:
                idx = words.index('find') + 1
            else:
                idx = words.index('for') + 1
            
            if idx < len(words):
                target = words[idx]
                response['action'] = {
                    'type': 'find',
                    'parameters': {'target': target}
                }
                response['text'] = f"Looking for {target}"
        
        # FOLLOW command
        elif 'follow' in text_lower:
            response['action'] = {'type': 'follow'}
            response['text'] = "I'll follow you"
        
        # Default
        else:
            response['text'] = "I understand. How can I help?"
        
        return response
    
    def _execute_action(self, action: Dict[str, Any]):
        """
        Execute an action based on AI decision.
        This connects to actual hardware control.
        """
        
        action_type = action.get('type')
        params = action.get('parameters', {})
        
        self.logger.info(f"Executing action: {action_type}")
        
        # MOVEMENT actions
        if action_type == 'move' and 'esp32' in self.modules:
            direction = params.get('direction', 'forward')
            duration = params.get('duration', 2.0)
            speed = params.get('speed', 50)
            
            if direction == 'forward':
                self.modules['esp32'].move_forward(speed, duration)
            elif direction == 'backward':
                self.modules['esp32'].move_backward(speed, duration)
            elif direction == 'left':
                self.modules['esp32'].turn_left(speed, duration)
            elif direction == 'right':
                self.modules['esp32'].turn_right(speed, duration)
        
        # STOP action
        elif action_type == 'stop' and 'esp32' in self.modules:
            self.modules['esp32'].stop_motors()
        
        # GPIO control
        elif action_type == 'gpio' and 'esp32' in self.modules:
            pin = params.get('pin')
            value = params.get('value')
            if pin is not None:
                self.modules['esp32'].set_gpio(pin, value)
        
        # FIND object
        elif action_type == 'find' and 'object_detection' in self.modules:
            target = params.get('target')
            if target:
                # Start looking for object
                result = self.modules['object_detection'].find_object(target)
                
                if result and 'tts' in self.modules:
                    self.modules['tts'].speak(
                        f"I found {target} at position {result['center']}",
                        priority=2
                    )
                elif 'tts' in self.modules:
                    self.modules['tts'].speak(
                        f"I cannot find {target} right now",
                        priority=2
                    )
        
        # FOLLOW person
        elif action_type == 'follow':
            # This would require continuous tracking
            self.brain.state = 'FOLLOWING'
            # Start follow behavior (simplified)
        
        # LEARN something
        elif action_type == 'learn' and 'learning' in self.modules:
            self.modules['learning'].learn_from_interaction({
                'type': 'explicit_learning',
                'data': params
            })
    
    def _setup_learning_flow(self):
        """
        Set up continuous learning from all interactions
        """
        
        if 'learning' not in self.modules:
            return
        
        # Learning module already has event handlers registered
        # Just ensure it's getting all events
        
        # Predict next action periodically
        def predict_and_suggest():
            while self.robot.running:
                time.sleep(60)  # Every minute
                
                if self.brain.current_user:
                    context = {
                        'user_id': self.brain.current_user,
                        'time': time.localtime().tm_hour,
                        'state': str(self.brain.state)
                    }
                    
                    prediction = self.modules['learning'].predict_next_action(context)
                    
                    if prediction and prediction['confidence'] > 0.7:
                        self.logger.info(f"Predicted action: {prediction}")
                        # Could proactively suggest
        
        # Start prediction thread
        threading.Thread(target=predict_and_suggest, daemon=True).start()
    
    def _setup_monitoring_flow(self):
        """
        Set up continuous environment monitoring
        """
        
        def monitor_environment():
            """
            Continuous monitoring loop
            """
            
            while self.robot.running:
                # Check for objects periodically
                if 'object_detection' in self.modules:
                    objects = self.modules['object_detection'].get_detected_objects()
                    
                    # Learn from environment
                    if objects and 'learning' in self.modules:
                        for obj in objects:
                            self.modules['learning'].learn_from_interaction({
                                'type': 'environment',
                                'object': obj
                            })
                
                # Clean up tracking
                if 'object_detection' in self.modules:
                    self.modules['object_detection'].cleanup_tracking()
                
                time.sleep(5)  # Every 5 seconds
        
        # Start monitoring thread
        threading.Thread(target=monitor_environment, daemon=True).start()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get complete system status
        """
        
        return {
            'conversation_active': self.conversation_active,
            'current_speaker': self.current_speaker,
            'authenticated_user': self.brain.current_user,
            'master_mode': self.brain.master_mode,
            'brain_state': str(self.brain.state),
            'modules_active': list(self.modules.keys()),
            'last_interaction': time.time() - self.last_command_time
        }


# Integration helper function
def setup_robot_integration(robot_instance):
    """
    Set up complete robot integration.
    Call this from main.py after initializing modules.
    """
    
    integration = RobotSystemIntegration(robot_instance)
    integration.setup_complete_flow()
    
    return integration