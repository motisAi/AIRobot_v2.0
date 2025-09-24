"""
Speech Recognition Module
=========================
Converts speech to text using OpenAI Whisper (offline).
Optimized for Raspberry Pi with model selection based on available resources.


"""

import time
import threading
import queue
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import wave
import tempfile
import os

# Audio libraries
try:
    import pyaudio
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("Warning: speech_recognition not installed")

# Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: OpenAI Whisper not installed")

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import config, model_config, hardware_config


class SpeechRecognitionModule:
    """
    Speech recognition using OpenAI Whisper for offline processing.
    Falls back to Google Speech Recognition if Whisper unavailable.
    """
    
    def __init__(self, brain=None):
        """
        Initialize speech recognition module
        
        Args:
            brain: Reference to robot brain for event emission
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Speech Recognition Module")
        
        # Brain reference
        self.brain = brain
        
        # Configuration
        self.model_name = model_config.whisper_model
        self.language = model_config.whisper_language
        self.device = model_config.whisper_device
        
        # Audio settings
        self.sample_rate = hardware_config.microphone_rate
        self.chunk_size = hardware_config.microphone_chunk
        self.timeout = hardware_config.microphone_timeout
        self.phrase_limit = hardware_config.microphone_phrase_time_limit
        
        # State
        self.running = False
        self.recording = False
        self.processing = False
        
        # Whisper model
        self.whisper_model = None
        self.use_whisper = False
        
        # Speech recognizer (fallback)
        self.recognizer = None
        self.microphone = None
        
        # Audio buffer
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        
        # Threading
        self.recognition_thread = None
        self.recording_thread = None
        
        # Callbacks
        self.speech_callback = None
        self.error_callback = None
        
        # Statistics
        self.recognition_count = 0
        self.error_count = 0
        self.average_processing_time = 0
        
        # Initialize recognition engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize speech recognition engine"""
        
        # Try Whisper first
        if WHISPER_AVAILABLE:
            try:
                self._load_whisper_model()
                self.use_whisper = True
                self.logger.info(f"Using Whisper model: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load Whisper: {e}")
        
        # Fallback to speech_recognition
        if SR_AVAILABLE and not self.use_whisper:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone(
                    device_index=hardware_config.microphone_device_index,
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size
                )
                
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                self.logger.info("Using speech_recognition with Google Speech API")
            except Exception as e:
                self.logger.error(f"Failed to initialize speech_recognition: {e}")
        
        if not self.use_whisper and not self.recognizer:
            self.logger.error("No speech recognition engine available!")
    
    def _load_whisper_model(self):
        """Load Whisper model with optimization for Raspberry Pi"""
        
        # Check available memory
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        # Auto-select model based on available memory
        if available_memory < 2:
            actual_model = "tiny"
            self.logger.warning(f"Low memory ({available_memory:.1f}GB), using tiny model")
        elif available_memory < 4:
            actual_model = "base" if self.model_name != "tiny" else "tiny"
        else:
            actual_model = self.model_name
        
        # Load model
        self.logger.info(f"Loading Whisper model '{actual_model}'...")
        
        self.whisper_model = whisper.load_model(
            actual_model,
            device=self.device,
            download_root=str(Path(config.PROJECT_ROOT) / "data" / "models")
        )
        
        self.model_name = actual_model
        self.logger.info(f"Whisper model '{actual_model}' loaded successfully")
    
    def start(self):
        """Start speech recognition"""
        
        if self.running:
            self.logger.warning("Speech recognition already running")
            return
        
        self.running = True
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(target=self._recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        self.logger.info("Speech recognition started")
    
    def stop(self):
        """Stop speech recognition"""
        
        self.running = False
        self.recording = False
        
        # Wait for threads
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        self.logger.info("Speech recognition stopped")
    
    def start_recording(self, duration: Optional[float] = None):
        """
        Start recording audio for recognition
        
        Args:
            duration: Maximum duration in seconds (None for unlimited)
        """
        
        if self.recording:
            self.logger.warning("Already recording")
            return
        
        self.recording = True
        self.audio_buffer = []
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._record_audio,
            args=(duration,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.logger.info("Started recording")
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and process audio
        
        Returns:
            Recognized text or None
        """
        
        if not self.recording:
            self.logger.warning("Not recording")
            return None
        
        self.recording = False
        
        # Wait for recording to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        # Process recorded audio
        if self.audio_buffer:
            return self._process_audio_buffer()
        
        return None
    
    def _record_audio(self, duration: Optional[float] = None):
        """
        Record audio from microphone
        
        Args:
            duration: Maximum duration in seconds
        """
        
        if self.use_whisper:
            self._record_for_whisper(duration)
        else:
            self._record_with_speech_recognition(duration)
    
    def _record_for_whisper(self, duration: Optional[float] = None):
        """Record audio for Whisper processing"""
        
        try:
            import pyaudio
            
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info("Recording audio...")
            
            start_time = time.time()
            
            while self.recording:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Read audio chunk
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_buffer.append(data)
                except Exception as e:
                    self.logger.error(f"Recording error: {e}")
                    break
            
            # Clean up
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            self.logger.info(f"Recording complete ({len(self.audio_buffer)} chunks)")
            
        except Exception as e:
            self.logger.error(f"Recording failed: {e}")
    
    def _record_with_speech_recognition(self, duration: Optional[float] = None):
        """Record using speech_recognition library"""
        
        if not self.recognizer or not self.microphone:
            self.logger.error("Speech recognition not initialized")
            return
        
        try:
            with self.microphone as source:
                self.logger.info("Listening...")
                
                # Record audio
                if duration:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=duration
                    )
                else:
                    audio = self.recogn