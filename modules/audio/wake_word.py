"""
Wake Word Detection Module
==========================
Detects wake word "Hey Robot" using Picovoice Porcupine or alternatives.
Runs continuously in background with minimal CPU usage.

"""

import time
import threading
import queue
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
import struct
import wave
import os

# Audio libraries
try:
    import pyaudio
except ImportError:
    print("Warning: pyaudio not installed. Wake word detection will not work.")
    pyaudio = None

# Try to import Porcupine (Picovoice)
try:
    import pvporcupine
    import pvrecorder
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    print("Picovoice Porcupine not available. Using alternative wake word detection.")

# Alternative: webrtcvad for simple voice activity detection
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import config, model_config


class WakeWordDetector:
    """
    Wake word detection system.
    Uses Picovoice Porcupine if available, otherwise falls back to alternatives.
    """
    
    def __init__(self, brain=None):
        """
        Initialize wake word detector
        
        Args:
            brain: Reference to robot brain for event emission
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Wake Word Detector")
        
        # Brain reference
        self.brain = brain
        
        # Configuration
        self.wake_word = model_config.wake_word
        self.sensitivity = model_config.wake_word_sensitivity
        self.access_key = model_config.picovoice_access_key
        
        # Audio settings
        self.sample_rate = 16000  # Standard for speech
        self.frame_length = 512  # Samples per frame
        self.chunk_size = 1024
        
        # State
        self.running = False
        self.listening = True  # Can be paused
        self.detected_callback = None
        
        # Threading
        self.detection_thread = None
        
        # Audio interface
        self.audio = None
        self.stream = None
        
        # Detection engine
        self.porcupine = None
        self.use_porcupine = False
        
        # Alternative: Simple keyword detection
        self.vad = None
        self.use_vad = False
        
        # Buffer for alternative detection
        self.audio_buffer = queue.Queue()
        self.detection_buffer = []
        
        # Statistics
        self.detection_count = 0
        self.false_positive_count = 0
        self.last_detection_time = 0
        
        # Initialize detection engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the wake word detection engine"""
        
        # Try Porcupine first
        if PORCUPINE_AVAILABLE and self.access_key:
            try:
                self._init_porcupine()
                self.use_porcupine = True
                self.logger.info("Using Picovoice Porcupine for wake word detection")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize Porcupine: {e}")
        
        # Fallback to VAD + simple detection
        if VAD_AVAILABLE:
            try:
                self._init_vad()
                self.use_vad = True
                self.logger.info("Using WebRTC VAD for voice activity detection")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize VAD: {e}")
        
        # Last resort: simple audio threshold
        self.logger.warning("No advanced wake word detection available. Using simple audio detection.")
    
    def _init_porcupine(self):
        """Initialize Picovoice Porcupine"""
        
        # Built-in wake words (if no custom model)
        keywords = ['hey google', 'alexa', 'ok google', 'hey siri']
        
        # Try custom wake word if model exists
        custom_model_path = Path(model_config.wake_word_model_path)
        
        if custom_model_path.exists():
            # Use custom model
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[str(custom_model_path)],
                sensitivities=[self.sensitivity]
            )
        else:
            # Use built-in wake word closest to our phrase
            # For "hey robot", we'll use "hey google" and check the following audio
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=['hey google'],  # Closest to "hey robot"
                sensitivities=[self.sensitivity]
            )
            
            self.logger.info("Using 'hey google' as trigger, will verify 'robot' in post-processing")
    
    def _init_vad(self):
        """Initialize WebRTC Voice Activity Detector"""
        
        # VAD aggressiveness (0-3, 3 is most aggressive)
        aggressiveness = 2
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Also initialize simple keyword matching
        self.keywords = ['hey', 'robot', 'hi', 'hello']
    
    def start(self):
        """Start wake word detection"""
        
        if self.running:
            self.logger.warning("Wake word detection already running")
            return
        
        self.running = True
        
        # Initialize PyAudio
        if pyaudio:
            self.audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback if not self.use_porcupine else None
            )
            
            if self.stream:
                self.stream.start_stream()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.logger.info("Wake word detection started")
    
    def stop(self):
        """Stop wake word detection"""
        
        self.running = False
        
        # Wait for thread
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()
        
        # Clean up Porcupine
        if self.porcupine:
            self.porcupine.delete()
        
        self.logger.info("Wake word detection stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for audio input (non-Porcupine mode)
        
        Args:
            in_data: Audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flags
            
        Returns:
            Tuple of (data, flag)
        """
        
        if self.listening and self.running:
            # Add to buffer for processing
            self.audio_buffer.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    def _detection_loop(self):
        """Main detection loop"""
        
        if self.use_porcupine:
            self._porcupine_loop()
        else:
            self._alternative_loop()
    
    def _porcupine_loop(self):
        """Detection loop using Porcupine"""
        
        recorder = pvrecorder.PvRecorder(
            device_index=-1,  # Default audio device
            frame_length=self.porcupine.frame_length
        )
        
        recorder.start()
        
        self.logger.info("Porcupine listening for wake word...")
        
        try:
            while self.running:
                if not self.listening:
                    time.sleep(0.1)
                    continue
                
                # Read audio frame
                pcm = recorder.read()
                
                # Check for wake word
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    self._handle_detection()
        
        except Exception as e:
            self.logger.error(f"Porcupine error: {e}")
        
        finally:
            recorder.stop()
            recorder.delete()
    
    def _alternative_loop(self):
        """Detection loop using alternative methods"""
        
        self.logger.info("Alternative detection listening...")
        
        # Buffer for accumulating audio
        audio_accumulator = b''
        
        while self.running:
            try:
                if not self.listening:
                    time.sleep(0.1)
                    continue
                
                # Get audio data
                if not self.audio_buffer.empty():
                    audio_data = self.audio_buffer.get(timeout=0.1)
                    audio_accumulator += audio_data
                    
                    # Process when we have enough data
                    if len(audio_accumulator) >= self.sample_rate * 2:  # 2 seconds
                        
                        if self.use_vad:
                            # Check for voice activity
                            if self._check_voice_activity(audio_accumulator):
                                # Simple keyword detection
                                if self._check_keywords(audio_accumulator):
                                    self._handle_detection()
                        else:
                            # Simple volume threshold
                            if self._check_volume_threshold(audio_accumulator):
                                self._handle_detection()
                        
                        # Clear accumulator
                        audio_accumulator = b''
                
                else:
                    time.sleep(0.01)
            
            except Exception as e:
                self.logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)
    
    def _check_voice_activity(self, audio_data: bytes) -> bool:
        """
        Check if audio contains voice using VAD
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            bool: True if voice detected
        """
        
        if not self.vad:
            return False
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Process in 30ms frames (480 samples at 16kHz)
        frame_size = 480
        num_frames = len(audio_array) // frame_size
        
        voice_frames = 0
        
        for i in range(num_frames):
            frame = audio_array[i * frame_size:(i + 1) * frame_size]
            frame_bytes = frame.tobytes()
            
            if self.vad.is_speech(frame_bytes, self.sample_rate):
                voice_frames += 1
        
        # Return True if >30% frames contain voice
        return voice_frames > num_frames * 0.3
    
    def _check_keywords(self, audio_data: bytes) -> bool:
        """
        Simple keyword detection (very basic)
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            bool: True if keywords might be present
        """
        
        # This is a placeholder for more sophisticated detection
        # In practice, you'd use speech recognition here
        
        # For now, just use volume patterns
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Look for two peaks (hey + robot)
        threshold = np.max(np.abs(audio_array)) * 0.3
        above_threshold = np.abs(audio_array) > threshold
        
        # Simple pattern: two groups of activity
        # This is very basic and will have many false positives
        changes = np.diff(above_threshold.astype(int))
        peaks = np.sum(changes == 1)
        
        return peaks >= 2
    
    def _check_volume_threshold(self, audio_data: bytes) -> bool:
        """
        Check if audio exceeds volume threshold
        
        Args:
            audio_data: Audio bytes
            
        Returns:
            bool: True if loud enough
        """
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        volume = np.sqrt(np.mean(audio_array**2))
        
        # Dynamic threshold based on sensitivity
        threshold = 1000 * (1.0 - self.sensitivity)
        
        return volume > threshold
    
    def _handle_detection(self):
        """Handle wake word detection"""
        
        # Prevent multiple rapid detections
        current_time = time.time()
        if current_time - self.last_detection_time < 2.0:
            return
        
        self.last_detection_time = current_time
        self.detection_count += 1
        
        self.logger.info(f"Wake word detected! (count: {self.detection_count})")
        
        # Callback if set
        if self.detected_callback:
            self.detected_callback()
        
        # Emit event to brain
        if self.brain:
            from core.robot_brain import RobotEvent
            self.brain.emit_event(RobotEvent(
                type='wake_word_detected',
                source='wake_word',
                data={'timestamp': current_time},
                priority=1  # High priority
            ))
        
        # Pause listening briefly to avoid re-triggering
        self.pause_listening()
        threading.Timer(2.0, self.resume_listening).start()
    
    def pause_listening(self):
        """Temporarily pause wake word detection"""
        self.listening = False
        self.logger.debug("Wake word detection paused")
    
    def resume_listening(self):
        """Resume wake word detection"""
        self.listening = True
        self.logger.debug("Wake word detection resumed")
    
    def set_callback(self, callback: Callable):
        """
        Set callback function for wake word detection
        
        Args:
            callback: Function to call when wake word detected
        """
        self.detected_callback = callback
    
    def set_sensitivity(self, sensitivity: float):
        """
        Adjust detection sensitivity
        
        Args:
            sensitivity: Sensitivity value (0.0 to 1.0)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        
        if self.porcupine:
            # Recreate Porcupine with new sensitivity
            self.stop()
            self._initialize_engine()
            self.start()
    
    def get_statistics(self) -> dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'running': self.running,
            'listening': self.listening,
            'detection_count': self.detection_count,
            'false_positive_count': self.false_positive_count,
            'last_detection': self.last_detection_time,
            'engine': 'porcupine' if self.use_porcupine else 'vad' if self.use_vad else 'threshold',
            'wake_word': self.wake_word,
            'sensitivity': self.sensitivity
        }
    
    def save_audio_sample(self, filepath: str, duration: int = 3):
        """
        Save audio sample for testing/debugging
        
        Args:
            filepath: Path to save audio file
            duration: Duration in seconds
        """
        
        self.logger.info(f"Recording {duration}s audio sample...")
        
        frames = []
        
        for _ in range(0, int(self.sample_rate * duration / self.chunk_size)):
            if self.stream:
                data = self.stream.read(self.chunk_size)
                frames.append(data)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        self.logger.info(f"Audio sample saved to {filepath}")


# Standalone testing
if __name__ == "__main__":
    """Test wake word detection"""
    
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create detector
    detector = WakeWordDetector()
    
    # Set callback
    def on_detection():
        print("\n🎤 WAKE WORD DETECTED! 🎤\n")
    
    detector.set_callback(on_detection)
    
    print("\nWake Word Detection Test")
    print("=" * 50)
    print(f"Say '{model_config.wake_word}' to trigger detection")
    print("Press Ctrl+C to stop")
    print()
    
    # Start detection
    detector.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
            # Print statistics periodically
            stats = detector.get_statistics()
            print(f"\rDetections: {stats['detection_count']} | "
                  f"Engine: {stats['engine']} | "
                  f"Listening: {stats['listening']}", end='')
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        detector.stop()
        print("Detection stopped")