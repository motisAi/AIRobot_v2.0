"""
Conversation AI Module
=====================
Handles intelligent conversation using various AI models.
Supports local and cloud-based AI services.

Author: AI Robot System
Date: 2024
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

class ConversationContext:
    """Conversation context and history"""
    
    def __init__(self, max_history=20):
        self.messages = []
        self.max_history = max_history
        self.current_topic = None
        self.user_preferences = {}
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time()
        }
        
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_recent_messages(self, count: int = 10) -> List[Dict]:
        """Get recent messages"""
        return self.messages[-count:] if self.messages else []


class ConversationAI:
    """
    Main conversation AI module
    """
    
    def __init__(self, brain):
        """
        Initialize conversation AI
        
        Args:
            brain: Robot brain instance
        """
        self.brain = brain
        self.logger = logging.getLogger("ConversationAI")
        
        # Conversation state
        self.context = ConversationContext()
        self.is_active = False
        
        # AI response generation
        self._generate_ai_response = self._default_ai_response
        
        # Response callbacks
        self.response_callbacks: List[Callable] = []
        
        # Configuration
        self.response_timeout = 30  # seconds
        self.enable_context = True
        
        self.logger.info("Conversation AI initialized")
    
    def start_conversation(self, user_input: str = None):
        """
        Start or continue conversation
        
        Args:
            user_input: Optional initial user input
        """
        try:
            self.is_active = True
            self.logger.info("Conversation started")
            
            if user_input:
                response = self.process_input(user_input)
                return response
            else:
                # Start with greeting
                greeting = self._generate_greeting()
                self.context.add_message('assistant', greeting)
                
                # Notify callbacks
                self._notify_response_callbacks(greeting)
                
                return greeting
                
        except Exception as e:
            self.logger.error(f"Error starting conversation: {e}")
            return "I'm sorry, I'm having trouble starting our conversation."
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response
        
        Args:
            user_input: User's message
            
        Returns:
            AI response
        """
        try:
            if not user_input.strip():
                return "I didn't hear anything. Could you please repeat that?"
            
            self.logger.info(f"Processing input: {user_input}")
            
            # Add user message to context
            self.context.add_message('user', user_input)
            
            # Check for special commands
            if self._is_command(user_input):
                response = self._handle_command(user_input)
            else:
                # Generate AI response
                response = self._generate_ai_response(user_input, self.context)
            
            # Add response to context
            self.context.add_message('assistant', response)
            
            # Notify callbacks
            self._notify_response_callbacks(response)
            
            # Notify brain
            if hasattr(self.brain, 'process_event'):
                self.brain.process_event({
                    'type': 'conversation_response',
                    'data': {
                        'input': user_input,
                        'response': response,
                        'timestamp': time.time()
                    }
                })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return "I'm sorry, I encountered an error while processing your message."
    
    def _default_ai_response(self, text: str, context: ConversationContext) -> str:
        """
        Default AI response generator (fallback)
        
        Args:
            text: User input text
            context: Conversation context
            
        Returns:
            Generated response
        """
        # Simple rule-based responses as fallback
        text_lower = text.lower()
        
        # Greetings
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! How can I help you today?"
        
        # Questions about robot
        elif any(word in text_lower for word in ['who are you', 'what are you', 'your name']):
            return "I'm an AI robot assistant. I'm here to help you with various tasks!"
        
        # Help requests
        elif any(word in text_lower for word in ['help', 'what can you do', 'capabilities']):
            return "I can help you with conversation, answer questions, control smart devices, and much more. What would you like to do?"
        
        # Time/date questions
        elif any(word in text_lower for word in ['time', 'what time', 'clock']):
            current_time = time.strftime("%I:%M %p")
            return f"The current time is {current_time}."
        
        elif any(word in text_lower for word in ['date', 'what date', 'today']):
            current_date = time.strftime("%A, %B %d, %Y")
            return f"Today is {current_date}."
        
        # Weather (placeholder)
        elif 'weather' in text_lower:
            return "I don't have access to current weather data right now, but I can help you with other things!"
        
        # Farewell
        elif any(word in text_lower for word in ['goodbye', 'bye', 'see you', 'farewell']):
            return "Goodbye! It was nice talking with you!"
        
        # Default response
        else:
            responses = [
                "That's interesting! Tell me more about that.",
                "I understand. What else would you like to know?",
                "Can you elaborate on that?",
                "That's a good point. How can I help you with that?",
                "I see. Is there anything specific I can assist you with?"
            ]
            
            # Use a simple hash to pick response consistently
            response_index = hash(text) % len(responses)
            return responses[response_index]
    
    def _generate_greeting(self) -> str:
        """Generate a greeting message"""
        greetings = [
            "Hello! I'm your AI assistant. How can I help you today?",
            "Hi there! What can I do for you?",
            "Good to see you! How may I assist you?",
            "Hello! I'm ready to help. What's on your mind?"
        ]
        
        # Use time-based selection
        greeting_index = int(time.time()) % len(greetings)
        return greetings[greeting_index]
    
    def _is_command(self, text: str) -> bool:
        """Check if input is a special command"""
        command_prefixes = ['/', '!', 'robot ']
        return any(text.lower().startswith(prefix) for prefix in command_prefixes)
    
    def _handle_command(self, text: str) -> str:
        """Handle special commands"""
        text_lower = text.lower()
        
        if 'stop' in text_lower or 'end' in text_lower:
            self.end_conversation()
            return "Ending our conversation. Goodbye!"
        
        elif 'clear' in text_lower or 'reset' in text_lower:
            self.context = ConversationContext()
            return "Conversation history cleared. Starting fresh!"
        
        elif 'status' in text_lower:
            return self._get_status_info()
        
        else:
            return "I didn't understand that command. Try 'status', 'clear', or 'stop'."
    
    def _get_status_info(self) -> str:
        """Get conversation status information"""
        message_count = len(self.context.messages)
        return f"Conversation active with {message_count} messages in history. AI system operational."
    
    def end_conversation(self):
        """End the current conversation"""
        self.is_active = False
        self.logger.info("Conversation ended")
        
        # Notify brain
        if hasattr(self.brain, 'process_event'):
            self.brain.process_event({
                'type': 'conversation_ended',
                'data': {'timestamp': time.time()}
            })
    
    def set_ai_generator(self, generator_func: Callable):
        """
        Set custom AI response generator
        
        Args:
            generator_func: Function that takes (text, context) and returns response
        """
        self._generate_ai_response = generator_func
        self.logger.info("Custom AI generator set")
    
    def add_response_callback(self, callback: Callable):
        """Add callback for when responses are generated"""
        self.response_callbacks.append(callback)
    
    def _notify_response_callbacks(self, response: str):
        """Notify all response callbacks"""
        for callback in self.response_callbacks:
            try:
                callback(response)
            except Exception as e:
                self.logger.error(f"Error in response callback: {e}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.context.messages.copy()
    
    def save_conversation(self, filename: str = None):
        """Save conversation history to file"""
        try:
            if not filename:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"
            
            # Ensure logs directory exists
            log_dir = Path("data/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = log_dir / filename
            
            conversation_data = {
                'timestamp': time.time(),
                'messages': self.context.messages,
                'metadata': {
                    'message_count': len(self.context.messages),
                    'current_topic': self.context.current_topic
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            self.logger.info(f"Conversation saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
            return None
    
    def load_conversation(self, filepath: str) -> bool:
        """Load conversation history from file"""
        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
            
            self.context.messages = conversation_data.get('messages', [])
            self.context.current_topic = conversation_data.get('metadata', {}).get('current_topic')
            
            self.logger.info(f"Conversation loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading conversation: {e}")
            return False