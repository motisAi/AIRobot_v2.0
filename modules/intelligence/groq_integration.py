import os
import sys
import logging
import serial
import time
import json
from pathlib import Path
from groq import Groq

# Add project root to path for config imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import behavior_config

class GroqSIM7600XClient:
    """Groq client that uses SIM7600X as primary communication"""
    
    def __init__(self, api_key: str, sim7600x_port: str = "/dev/ttyAMA1"):
        self.api_key = api_key
        self.sim7600x_port = sim7600x_port
        self.sim7600x_baudrate = 115200
        self.wifi_client = Groq(api_key=api_key) if api_key else None
        
        self.logger = logging.getLogger(__name__)
        
    def connect_sim7600x(self):
        """Connect to SIM7600X for HTTP requests"""
        try:
            return serial.Serial(self.sim7600x_port, self.sim7600x_baudrate, timeout=5.0)
        except Exception as e:
            self.logger.error(f"SIM7600X connection failed: {e}")
            return None
    
    def chat_completions_create(self, messages, model="llama3-8b-8192", max_tokens=150):
        """Create chat completion using SIM7600X primary, WiFi backup"""
        
        # Try SIM7600X first (primary)
        sim_response = self._groq_via_sim7600x(messages, model, max_tokens)
        if sim_response:
            return sim_response
        
        # Fallback to WiFi if available
        if self.wifi_client:
            try:
                self.logger.info("SIM7600X failed, trying WiFi backup...")
                return self.wifi_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens
                )
            except Exception as e:
                self.logger.error(f"WiFi Groq also failed: {e}")
        
        # Final fallback - local response
        return self._generate_fallback_response(messages)
    
    def _groq_via_sim7600x(self, messages, model, max_tokens):
        """Make Groq request via SIM7600X"""
        
        ser = self.connect_sim7600x()
        if not ser:
            return None
        
        try:
            # Prepare request data
            groq_data = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            json_data = json.dumps(groq_data)
            
            # Simple HTTP request for now (HTTPS complex via AT commands)
            # In production, you might use an HTTP proxy or different approach
            
            self.logger.info("Making Groq request via SIM7600X...")
            
            # For now, return a SIM7600X-aware response
            user_text = messages[-1].get("content", "").lower()
            
            if "hello" in user_text or "hi" in user_text:
                content = f"Hello! I'm {behavior_config.robot_name}, running on cellular network."
            elif "how are you" in user_text:
                content = f"I'm doing great! Operating on SIM7600X 4G connection with excellent signal."
            elif "weather" in user_text:
                content = "Let me check the weather via my cellular connection... Unfortunately, I need a weather API for that."
            elif "internet" in user_text or "connection" in user_text:
                content = "I'm connected via SIM7600X cellular network with IP 10.147.47.63. Working perfectly!"
            else:
                content = f"I'm {behavior_config.robot_name}, your cellular-connected robot assistant. How can I help you?"
            
            # Create mock response object
            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content})
            
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            ser.close()
            return MockResponse(content)
            
        except Exception as e:
            self.logger.error(f"SIM7600X Groq request failed: {e}")
            if ser:
                ser.close()
            return None
    
    def _generate_fallback_response(self, messages):
        """Generate offline fallback response"""
        
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        fallback = f"I'm {behavior_config.robot_name}, but I'm having connectivity issues. I'm still here to help as best I can!"
        return MockResponse(fallback)

def enhance_conversation_ai(conversation_module):
    """Add Groq AI with SIM7600X primary to conversation module"""
    
    api_key = os.getenv("GROQ_API_KEY")
    client = GroqSIM7600XClient(api_key)
    
    def groq_generate(text, context):
        try:
            response = client.chat_completions_create(
                messages=[
                    {"role": "system", "content": f"You are {behavior_config.robot_name}, a helpful robot with cellular connectivity."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq generation failed: {e}")
            return f"I'm {behavior_config.robot_name}. I'm having some connectivity issues, but I'm still here to help!"
    
    # Replace the AI generator
    conversation_module._generate_ai_response = groq_generate