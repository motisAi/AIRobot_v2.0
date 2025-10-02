"""
Voice Identification Module
===========================
Identifies speakers by analyzing voice characteristics.
Uses voice embeddings and similarity matching.

Author: AI Robot System
Date: 2024
"""

import numpy as np
import logging
import pickle
import time
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib

class VoiceIdentificationModule:
    """
    Voice identification using voice embeddings
    """
    
    def __init__(self, brain):
        """
        Initialize voice identification
        
        Args:
            brain: Robot brain instance
        """
        self.brain = brain
        self.logger = logging.getLogger("VoiceID")
        
        # Voice profiles storage
        self.voice_profiles = {}
        self.profiles_file = Path("data/voices/voice_profiles.pkl")
        self.profiles_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
        
        # Current processing
        self.current_speaker = None
        self.confidence_threshold = 0.7
        
        self.logger.info("Voice Identification initialized")
    
    def _load_profiles(self):
        """Load voice profiles from disk"""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, 'rb') as f:
                    self.voice_profiles = pickle.load(f)
                self.logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")
            else:
                self.voice_profiles = {}
                self.logger.info("No existing voice profiles found")
        except Exception as e:
            self.logger.error(f"Error loading voice profiles: {e}")
            self.voice_profiles = {}
    
    def _save_profiles(self):
        """Save voice profiles to disk"""
        try:
            with open(self.profiles_file, 'wb') as f:
                pickle.dump(self.voice_profiles, f)
            self.logger.info("Voice profiles saved")
        except Exception as e:
            self.logger.error(f"Error saving voice profiles: {e}")
    
    def extract_voice_features(self, audio_data) -> np.ndarray:
        """
        Extract voice features from audio data
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Voice feature vector
        """
        try:
            # Placeholder feature extraction
            # In a real implementation, you would use:
            # - MFCC features
            # - Speaker embeddings (x-vector, d-vector)
            # - Deep learning models like SpeakerNet
            
            # Simple statistical features as placeholder
            features = []
            
            if isinstance(audio_data, (list, np.ndarray)):
                audio_array = np.array(audio_data, dtype=np.float32)
                
                # Basic statistical features
                features.extend([
                    np.mean(audio_array),
                    np.std(audio_array),
                    np.max(audio_array),
                    np.min(audio_array),
                    np.median(audio_array)
                ])
                
                # Spectral features (placeholder)
                if len(audio_array) > 1:
                    fft = np.fft.fft(audio_array)
                    fft_mag = np.abs(fft)
                    features.extend([
                        np.mean(fft_mag),
                        np.std(fft_mag),
                        np.max(fft_mag)
                    ])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting voice features: {e}")
            return np.array([])
    
    def register_speaker(self, speaker_name: str, audio_samples: List) -> bool:
        """
        Register a new speaker with voice samples
        
        Args:
            speaker_name: Name of the speaker
            audio_samples: List of audio samples for training
            
        Returns:
            True if registration successful
        """
        try:
            if not audio_samples:
                self.logger.error("No audio samples provided")
                return False
            
            # Extract features from all samples
            features_list = []
            for sample in audio_samples:
                features = self.extract_voice_features(sample)
                if len(features) > 0:
                    features_list.append(features)
            
            if not features_list:
                self.logger.error("Could not extract features from samples")
                return False
            
            # Create voice profile (average features)
            voice_profile = {
                'name': speaker_name,
                'features': np.mean(features_list, axis=0),
                'samples_count': len(features_list),
                'created_at': time.time()
            }
            
            self.voice_profiles[speaker_name] = voice_profile
            self._save_profiles()
            
            self.logger.info(f"Registered speaker: {speaker_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering speaker: {e}")
            return False
    
    def identify_speaker(self, audio_data) -> Optional[Dict[str, Any]]:
        """
        Identify speaker from audio data
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Speaker identification result with confidence
        """
        try:
            if not self.voice_profiles:
                return None
            
            # Extract features from input audio
            input_features = self.extract_voice_features(audio_data)
            if len(input_features) == 0:
                return None
            
            # Compare with all registered profiles
            best_match = None
            best_similarity = 0
            
            for speaker_name, profile in self.voice_profiles.items():
                # Calculate similarity (cosine similarity)
                similarity = self._calculate_similarity(input_features, profile['features'])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_name
            
            # Check if confidence is above threshold
            if best_similarity >= self.confidence_threshold:
                result = {
                    'speaker': best_match,
                    'confidence': best_similarity,
                    'timestamp': time.time()
                }
                
                self.current_speaker = best_match
                self.logger.info(f"Identified speaker: {best_match} (confidence: {best_similarity:.2f})")
                
                # Notify brain
                if hasattr(self.brain, 'process_event'):
                    self.brain.process_event({
                        'type': 'speaker_identified',
                        'data': result
                    })
                
                return result
            else:
                self.current_speaker = None
                return None
                
        except Exception as e:
            self.logger.error(f"Error identifying speaker: {e}")
            return None
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate similarity between two feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Ensure same length
            min_len = min(len(features1), len(features2))
            if min_len == 0:
                return 0.0
            
            f1 = features1[:min_len]
            f2 = features2[:min_len]
            
            # Cosine similarity
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_registered_speakers(self) -> List[str]:
        """Get list of registered speakers"""
        return list(self.voice_profiles.keys())
    
    def remove_speaker(self, speaker_name: str) -> bool:
        """
        Remove a speaker profile
        
        Args:
            speaker_name: Name of speaker to remove
            
        Returns:
            True if removal successful
        """
        try:
            if speaker_name in self.voice_profiles:
                del self.voice_profiles[speaker_name]
                self._save_profiles()
                self.logger.info(f"Removed speaker: {speaker_name}")
                return True
            else:
                self.logger.warning(f"Speaker not found: {speaker_name}")
                return False
        except Exception as e:
            self.logger.error(f"Error removing speaker: {e}")
            return False