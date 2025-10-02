"""
Learning Module
==============
Handles machine learning, pattern recognition, and adaptive behavior.
Learns from user interactions and environmental data.

Author: AI Robot System
Date: 2024
"""

import logging
import time
import json
import pickle
import threading
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict, deque

class LearningModule:
    """
    Machine learning and adaptation module
    """
    
    def __init__(self, brain):
        """
        Initialize learning module
        
        Args:
            brain: Robot brain instance
        """
        self.brain = brain
        self.logger = logging.getLogger("Learning")
        
        # Data storage
        self.data_dir = Path("data/models")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning data
        self.user_patterns = defaultdict(list)
        self.interaction_history = deque(maxlen=1000)
        self.behavior_patterns = {}
        self.preferences = {}
        
        # Learning models
        self.models = {}
        
        # Learning configuration
        self.learning_enabled = True
        self.min_data_points = 10
        self.confidence_threshold = 0.7
        
        # Background learning
        self.learning_thread = None
        self.learning_running = False
        
        # Load existing data
        self._load_learning_data()
        
        self.logger.info("Learning Module initialized")
    
    def start_learning(self):
        """Start background learning process"""
        if not self.learning_running:
            self.learning_running = True
            self.learning_thread = threading.Thread(target=self._learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            self.logger.info("Learning process started")
    
    def stop_learning(self):
        """Stop background learning process"""
        if self.learning_running:
            self.learning_running = False
            if self.learning_thread:
                self.learning_thread.join(timeout=5)
            self.logger.info("Learning process stopped")
    
    def record_interaction(self, interaction_data: Dict[str, Any]):
        """
        Record user interaction for learning
        
        Args:
            interaction_data: Dictionary containing interaction details
        """
        try:
            if not self.learning_enabled:
                return
            
            # Add timestamp
            interaction_data['timestamp'] = time.time()
            
            # Store interaction
            self.interaction_history.append(interaction_data)
            
            # Extract patterns
            self._extract_patterns(interaction_data)
            
            self.logger.debug(f"Recorded interaction: {interaction_data.get('type', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error recording interaction: {e}")
    
    def _extract_patterns(self, interaction: Dict[str, Any]):
        """Extract patterns from interaction data"""
        try:
            interaction_type = interaction.get('type', 'unknown')
            user_id = interaction.get('user_id', 'default')
            
            # Time-based patterns
            timestamp = interaction.get('timestamp', time.time())
            hour = time.localtime(timestamp).tm_hour
            day_of_week = time.localtime(timestamp).tm_wday
            
            pattern_key = f"{user_id}_{interaction_type}"
            
            pattern_data = {
                'hour': hour,
                'day_of_week': day_of_week,
                'data': interaction.get('data', {}),
                'timestamp': timestamp
            }
            
            self.user_patterns[pattern_key].append(pattern_data)
            
            # Keep only recent patterns
            if len(self.user_patterns[pattern_key]) > 100:
                self.user_patterns[pattern_key] = self.user_patterns[pattern_key][-100:]
            
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
    
    def learn_user_preferences(self, user_id: str = 'default') -> Dict[str, Any]:
        """
        Learn user preferences from interaction history
        
        Args:
            user_id: User identifier
            
        Returns:
            Learned preferences dictionary
        """
        try:
            if user_id not in self.preferences:
                self.preferences[user_id] = {}
            
            # Analyze conversation preferences
            conversation_patterns = self.user_patterns.get(f"{user_id}_conversation", [])
            if len(conversation_patterns) >= self.min_data_points:
                self.preferences[user_id]['conversation'] = self._analyze_conversation_preferences(conversation_patterns)
            
            # Analyze time preferences
            all_patterns = []
            for key, patterns in self.user_patterns.items():
                if key.startswith(f"{user_id}_"):
                    all_patterns.extend(patterns)
            
            if len(all_patterns) >= self.min_data_points:
                self.preferences[user_id]['timing'] = self._analyze_timing_preferences(all_patterns)
            
            # Analyze interaction preferences
            if len(all_patterns) >= self.min_data_points:
                self.preferences[user_id]['interaction'] = self._analyze_interaction_preferences(all_patterns)
            
            self.logger.info(f"Updated preferences for user {user_id}")
            return self.preferences[user_id]
            
        except Exception as e:
            self.logger.error(f"Error learning user preferences: {e}")
            return {}
    
    def _analyze_conversation_preferences(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation preferences"""
        try:
            preferences = {
                'preferred_topics': [],
                'response_length': 'medium',
                'formality': 'casual',
                'interaction_frequency': 'normal'
            }
            
            # Analyze topics from conversation data
            topics = []
            response_lengths = []
            
            for pattern in patterns:
                data = pattern.get('data', {})
                
                # Extract topics (placeholder - would need NLP)
                if 'input' in data:
                    input_text = data['input'].lower()
                    # Simple keyword extraction
                    if any(word in input_text for word in ['weather', 'temperature', 'forecast']):
                        topics.append('weather')
                    elif any(word in input_text for word in ['time', 'clock', 'schedule']):
                        topics.append('time')
                    elif any(word in input_text for word in ['music', 'song', 'play']):
                        topics.append('music')
                
                # Analyze response preferences
                if 'response' in data:
                    response_len = len(data['response'].split())
                    response_lengths.append(response_len)
            
            # Determine preferred topics
            if topics:
                topic_counts = defaultdict(int)
                for topic in topics:
                    topic_counts[topic] += 1
                preferences['preferred_topics'] = list(topic_counts.keys())
            
            # Determine preferred response length
            if response_lengths:
                avg_length = np.mean(response_lengths)
                if avg_length < 10:
                    preferences['response_length'] = 'short'
                elif avg_length > 30:
                    preferences['response_length'] = 'long'
                else:
                    preferences['response_length'] = 'medium'
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation preferences: {e}")
            return {}
    
    def _analyze_timing_preferences(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze timing preferences"""
        try:
            preferences = {
                'active_hours': [],
                'preferred_days': [],
                'interaction_duration': 'normal'
            }
            
            # Analyze active hours
            hours = [pattern.get('hour', 12) for pattern in patterns]
            if hours:
                hour_counts = defaultdict(int)
                for hour in hours:
                    hour_counts[hour] += 1
                
                # Find most active hours (top 25%)
                sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
                top_hours = [hour for hour, count in sorted_hours[:len(sorted_hours)//4]]
                preferences['active_hours'] = sorted(top_hours)
            
            # Analyze preferred days
            days = [pattern.get('day_of_week', 0) for pattern in patterns]
            if days:
                day_counts = defaultdict(int)
                for day in days:
                    day_counts[day] += 1
                
                # Convert to day names
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                preferred_days = []
                for day, count in day_counts.items():
                    if count > len(patterns) / 10:  # More than 10% of interactions
                        preferred_days.append(day_names[day % 7])
                
                preferences['preferred_days'] = preferred_days
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error analyzing timing preferences: {e}")
            return {}
    
    def _analyze_interaction_preferences(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze interaction preferences"""
        try:
            preferences = {
                'preferred_modalities': [],
                'response_speed': 'normal',
                'help_level': 'moderate'
            }
            
            # Analyze interaction types
            interaction_types = []
            for pattern in patterns:
                data = pattern.get('data', {})
                if 'input' in data:
                    input_text = data['input'].lower()
                    
                    # Classify interaction type
                    if any(word in input_text for word in ['help', 'how', 'what', 'explain']):
                        interaction_types.append('help_seeking')
                    elif any(word in input_text for word in ['do', 'control', 'turn', 'start']):
                        interaction_types.append('command')
                    else:
                        interaction_types.append('conversation')
            
            # Determine preferred interaction types
            if interaction_types:
                type_counts = defaultdict(int)
                for itype in interaction_types:
                    type_counts[itype] += 1
                
                preferences['preferred_modalities'] = list(type_counts.keys())
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error analyzing interaction preferences: {e}")
            return {}
    
    def predict_user_intent(self, input_data: Dict[str, Any], user_id: str = 'default') -> Dict[str, Any]:
        """
        Predict user intent based on learned patterns
        
        Args:
            input_data: Current input data
            user_id: User identifier
            
        Returns:
            Predicted intent with confidence score
        """
        try:
            intent_result = {
                'intent': 'unknown',
                'confidence': 0.0,
                'suggestions': []
            }
            
            # Get user patterns
            user_prefs = self.preferences.get(user_id, {})
            
            # Simple intent classification based on keywords
            input_text = input_data.get('text', '').lower()
            
            # Define intent patterns
            intent_patterns = {
                'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
                'question': ['what', 'how', 'when', 'where', 'why', 'who'],
                'command': ['turn', 'start', 'stop', 'control', 'do', 'make'],
                'help': ['help', 'assist', 'support', 'guide'],
                'information': ['tell', 'explain', 'describe', 'about'],
                'farewell': ['goodbye', 'bye', 'see you', 'farewell']
            }
            
            # Calculate intent scores
            intent_scores = {}
            for intent, keywords in intent_patterns.items():
                score = sum(1 for keyword in keywords if keyword in input_text)
                if score > 0:
                    intent_scores[intent] = score / len(keywords)
            
            # Determine best intent
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                confidence = intent_scores[best_intent]
                
                intent_result['intent'] = best_intent
                intent_result['confidence'] = min(confidence, 1.0)
            
            # Add suggestions based on user preferences
            conv_prefs = user_prefs.get('conversation', {})
            if conv_prefs.get('preferred_topics'):
                intent_result['suggestions'] = conv_prefs['preferred_topics'][:3]
            
            return intent_result
            
        except Exception as e:
            self.logger.error(f"Error predicting user intent: {e}")
            return {'intent': 'unknown', 'confidence': 0.0, 'suggestions': []}
    
    def adapt_behavior(self, context: Dict[str, Any]):
        """
        Adapt robot behavior based on learned patterns
        
        Args:
            context: Current context information
        """
        try:
            user_id = context.get('user_id', 'default')
            user_prefs = self.preferences.get(user_id, {})
            
            # Adapt conversation style
            conv_prefs = user_prefs.get('conversation', {})
            if conv_prefs:
                # Adjust response parameters based on preferences
                adaptations = {
                    'response_length': conv_prefs.get('response_length', 'medium'),
                    'formality': conv_prefs.get('formality', 'casual'),
                    'preferred_topics': conv_prefs.get('preferred_topics', [])
                }
                
                # Notify brain of adaptations
                if hasattr(self.brain, 'process_event'):
                    self.brain.process_event({
                        'type': 'behavior_adaptation',
                        'data': adaptations
                    })
                
                self.logger.info(f"Adapted behavior for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error adapting behavior: {e}")
    
    def _learning_loop(self):
        """Background learning loop"""
        while self.learning_running:
            try:
                # Periodic learning updates
                if len(self.interaction_history) >= self.min_data_points:
                    # Update user preferences
                    for user_id in set(interaction.get('user_id', 'default') 
                                     for interaction in self.interaction_history):
                        self.learn_user_preferences(user_id)
                    
                    # Save learning data periodically
                    self._save_learning_data()
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
                time.sleep(30)
    
    def _save_learning_data(self):
        """Save learning data to disk"""
        try:
            # Save patterns
            patterns_file = self.data_dir / "user_patterns.pkl"
            with open(patterns_file, 'wb') as f:
                pickle.dump(dict(self.user_patterns), f)
            
            # Save preferences
            preferences_file = self.data_dir / "preferences.json"
            with open(preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
            
            # Save interaction history
            history_file = self.data_dir / "interaction_history.pkl"
            with open(history_file, 'wb') as f:
                pickle.dump(list(self.interaction_history), f)
            
            self.logger.debug("Learning data saved")
            
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    def _load_learning_data(self):
        """Load learning data from disk"""
        try:
            # Load patterns
            patterns_file = self.data_dir / "user_patterns.pkl"
            if patterns_file.exists():
                with open(patterns_file, 'rb') as f:
                    loaded_patterns = pickle.load(f)
                    self.user_patterns.update(loaded_patterns)
            
            # Load preferences
            preferences_file = self.data_dir / "preferences.json"
            if preferences_file.exists():
                with open(preferences_file, 'r') as f:
                    self.preferences = json.load(f)
            
            # Load interaction history
            history_file = self.data_dir / "interaction_history.pkl"
            if history_file.exists():
                with open(history_file, 'rb') as f:
                    loaded_history = pickle.load(f)
                    self.interaction_history.extend(loaded_history)
            
            self.logger.info("Learning data loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_interactions': len(self.interaction_history),
            'user_count': len(self.preferences),
            'pattern_types': len(self.user_patterns),
            'learning_enabled': self.learning_enabled,
            'data_points_threshold': self.min_data_points
        }
    
    def reset_learning_data(self, user_id: str = None):
        """
        Reset learning data
        
        Args:
            user_id: Optional user ID to reset specific user data
        """
        try:
            if user_id:
                # Reset specific user data
                if user_id in self.preferences:
                    del self.preferences[user_id]
                
                # Remove user patterns
                keys_to_remove = [key for key in self.user_patterns.keys() if key.startswith(f"{user_id}_")]
                for key in keys_to_remove:
                    del self.user_patterns[key]
                
                self.logger.info(f"Reset learning data for user {user_id}")
            else:
                # Reset all data
                self.user_patterns.clear()
                self.interaction_history.clear()
                self.preferences.clear()
                
                self.logger.info("Reset all learning data")
            
            self._save_learning_data()
            
        except Exception as e:
            self.logger.error(f"Error resetting learning data: {e}")