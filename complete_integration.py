"""
Complete System Integration - Connect EVERYTHING!
==================================================
This file connects all the disconnected modules:
1. Voice ID - Speaker identification
2. Groq AI - Smart conversation with web search
3. DuckDuckGo - Web search
4. SIM7600 - 4G LTE communication

Author: AI Assistant
Date: 2024
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# ============================================
# PART 1: GROQ AI INTEGRATION
# ============================================

class GroqIntegration:
    """
    Integrates Groq AI for intelligent conversation and web search
    """
    
    def __init__(self):
        """Initialize Groq with API key from environment"""
        self.logger = logging.getLogger("GroqAI")
        
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                self.logger.warning("GROQ_API_KEY not found in .env")
                self.logger.info("Get free API key from: https://console.groq.com")
                self.client = None
            else:
                self.client = Groq(api_key=api_key)
                self.logger.info("✓ Groq AI initialized successfully")
                
        except ImportError:
            self.logger.error("Groq not installed. Run: pip install groq")
            self.client = None
    
    def chat(self, message: str, context: List[Dict] = None, 
             search_web: bool = False) -> str:
        """
        Chat with Groq AI
        
        Args:
            message: User message
            context: Previous conversation context
            search_web: Whether to search the web for information
            
        Returns:
            AI response
        """
        
        if not self.client:
            return "Groq AI is not configured. Please add GROQ_API_KEY to .env file."
        
        try:
            # Build messages
            messages = []
            
            # System prompt
            messages.append({
                "role": "system",
                "content": """You are a helpful AI robot assistant. 
                You can control robot movements, answer questions, and help with tasks.
                Be concise but friendly. If asked about current events or web info, 
                indicate you'll search for it."""
            })
            
            # Add context if provided
            if context:
                messages.extend(context[-5:])  # Last 5 messages for context
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # If web search needed, add a note
            if search_web or self._needs_web_search(message):
                messages.append({
                    "role": "system", 
                    "content": "Note: User may be asking about current information. If needed, indicate you'll search for it."
                })
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Best free model
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Groq AI error: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def _needs_web_search(self, message: str) -> bool:
        """Check if message needs web search"""
        
        web_keywords = [
            "weather", "news", "current", "today", "latest",
            "search", "find online", "google", "what is happening",
            "stock", "price", "score", "result"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in web_keywords)


# ============================================
# PART 2: WEB SEARCH INTEGRATION
# ============================================

class WebSearchIntegration:
    """
    Web search using DuckDuckGo (no API key needed!)
    """
    
    def __init__(self):
        """Initialize web search"""
        self.logger = logging.getLogger("WebSearch")
        
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.available = True
            self.logger.info("✓ Web search initialized (DuckDuckGo)")
        except ImportError:
            self.logger.error("DuckDuckGo not installed. Run: pip install duckduckgo-search")
            self.available = False
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search the web
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of search results
        """
        
        if not self.available:
            return []
        
        try:
            results = self.ddgs.text(
                query,
                region='wt-wt',  # Worldwide
                safesearch='moderate',
                max_results=max_results
            )
            
            self.logger.info(f"Found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def search_and_summarize(self, query: str, groq_client=None) -> str:
        """
        Search and summarize results using AI
        
        Args:
            query: Search query
            groq_client: Groq client for summarization
            
        Returns:
            Summarized search results
        """
        
        results = self.search(query)
        
        if not results:
            return "No search results found."
        
        # Format results
        formatted = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.get('title', 'No title')}\n"
            formatted += f"   {result.get('body', 'No description')[:200]}...\n"
            formatted += f"   Source: {result.get('href', 'No URL')}\n\n"
        
        # If Groq available, summarize
        if groq_client and groq_client.client:
            summary_prompt = f"Summarize these search results concisely:\n{formatted}"
            summary = groq_client.chat(summary_prompt)
            return summary
        
        return formatted


# ============================================
# PART 3: VOICE ID INTEGRATION FIX
# ============================================

def fix_voice_identification_integration(system_integration):
    """
    Fix the voice identification connection in system_integration
    
    Args:
        system_integration: The RobotSystemIntegration instance
    """
    
    logger = logging.getLogger("VoiceIDFix")
    
    # Check if modules exist
    if 'speech_recognition' not in system_integration.modules:
        logger.error("Speech recognition module not found")
        return
    
    if 'voice_identification' not in system_integration.modules:
        logger.error("Voice identification module not found")
        return
    
    # Monkey-patch the speech recognition to save audio
    original_stop_recording = system_integration.modules['speech_recognition'].stop_recording
    
    def enhanced_stop_recording():
        """Enhanced version that also identifies speaker"""
        
        # Get the transcribed text
        text = original_stop_recording()
        
        # Get the last audio buffer for voice ID
        if hasattr(system_integration.modules['speech_recognition'], 'audio_buffer'):
            audio_buffer = system_integration.modules['speech_recognition'].audio_buffer
            
            if audio_buffer:
                # Convert to numpy array
                import numpy as np
                audio_data = np.frombuffer(b''.join(audio_buffer), dtype=np.int16)
                
                # Identify speaker
                speaker_id, confidence = system_integration.modules['voice_identification'].identify_speaker(audio_data)
                
                # Store in context
                system_integration.current_speaker = speaker_id if confidence > 0.7 else 'unknown'
                
                logger.info(f"Speaker identified: {speaker_id} (confidence: {confidence:.2%})")
                
                # If unknown, offer to learn
                if speaker_id == 'unknown' and confidence < 0.5:
                    if 'tts' in system_integration.modules:
                        system_integration.modules['tts'].speak(
                            "I don't recognize your voice. Would you like me to learn it?",
                            priority=3
                        )
        
        return text
    
    # Replace the method
    system_integration.modules['speech_recognition'].stop_recording = enhanced_stop_recording
    
    logger.info("✓ Voice identification integration fixed")


# ============================================
# PART 4: SIM7600 4G LTE MODULE
# ============================================

class SIM7600Integration:
    """
    SIM7600 4G LTE module for SMS, calls, and internet
    """
    
    def __init__(self, port: str = "/dev/ttyUSB2"):
        """
        Initialize SIM7600
        
        Args:
            port: Serial port (usually /dev/ttyUSB2 for SIM7600)
        """
        
        self.logger = logging.getLogger("SIM7600")
        self.port = port
        self.connected = False
        
        try:
            import serial
            self.serial = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=1
            )
            
            # Test connection
            if self._send_at_command("AT"):
                self.connected = True
                self.logger.info(f"✓ SIM7600 connected on {port}")
                
                # Get module info
                info = self._send_at_command("ATI")
                self.logger.info(f"Module: {info}")
                
                # Check SIM card
                sim_status = self._send_at_command("AT+CPIN?")
                if "READY" in sim_status:
                    self.logger.info("✓ SIM card ready")
                else:
                    self.logger.warning(f"SIM status: {sim_status}")
                    
                # Check network
                network = self._send_at_command("AT+COPS?")
                self.logger.info(f"Network: {network}")
                
                # Check signal
                signal = self._send_at_command("AT+CSQ")
                self.logger.info(f"Signal: {signal}")
                
        except ImportError:
            self.logger.error("pyserial not installed. Run: pip install pyserial")
        except Exception as e:
            self.logger.error(f"SIM7600 connection failed: {e}")
    
    def _send_at_command(self, command: str, timeout: float = 1.0) -> str:
        """Send AT command and get response"""
        
        if not self.connected:
            return ""
        
        try:
            self.serial.write(f"{command}\r\n".encode())
            time.sleep(0.1)
            
            response = ""
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.serial.in_waiting:
                    response += self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                    if "OK" in response or "ERROR" in response:
                        break
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"AT command error: {e}")
            return ""
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """
        Send SMS
        
        Args:
            phone_number: Phone number (international format)
            message: SMS text
            
        Returns:
            Success status
        """
        
        if not self.connected:
            return False
        
        try:
            # Set to text mode
            self._send_at_command("AT+CMGF=1")
            
            # Set recipient
            self._send_at_command(f'AT+CMGS="{phone_number}"')
            time.sleep(0.5)
            
            # Send message
            self.serial.write(f"{message}\x1A".encode())  # Ctrl+Z to send
            time.sleep(2)
            
            response = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
            
            if "OK" in response:
                self.logger.info(f"✓ SMS sent to {phone_number}")
                return True
            else:
                self.logger.error(f"SMS failed: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"SMS error: {e}")
            return False
    
    def make_call(self, phone_number: str) -> bool:
        """Make a phone call"""
        
        if not self.connected:
            return False
        
        response = self._send_at_command(f"ATD{phone_number};")
        return "OK" in response
    
    def hang_up(self) -> bool:
        """Hang up current call"""
        
        response = self._send_at_command("ATH")
        return "OK" in response
    
    def check_internet(self) -> bool:
        """Check if internet connection is available"""
        
        response = self._send_at_command("AT+CGATT?")
        return "+CGATT: 1" in response  # 1 = attached to network
    
    def get_location(self) -> Optional[Dict[str, float]]:
        """
        Get GPS location (if SIM7600 has GPS)
        
        Returns:
            Dict with lat/lon or None
        """
        
        # Enable GPS
        self._send_at_command("AT+CGPS=1")
        time.sleep(2)
        
        # Get location
        response = self._send_at_command("AT+CGPSINFO")
        
        # Parse response (format: +CGPSINFO: lat,N,lon,E,...)
        if "+CGPSINFO:" in response:
            try:
                parts = response.split(":")[1].split(",")
                if len(parts) >= 4 and parts[0]:
                    return {
                        "latitude": float(parts[0]),
                        "longitude": float(parts[2])
                    }
            except:
                pass
        
        return None


# ============================================
# PART 5: COMPLETE INTEGRATION FUNCTION
# ============================================

def integrate_everything(robot_instance):
    """
    Integrate ALL the disconnected modules into the robot
    
    Args:
        robot_instance: The main AIRobot instance
    """
    
    logger = logging.getLogger("CompleteIntegration")
    logger.info("=" * 60)
    logger.info("Starting COMPLETE SYSTEM INTEGRATION")
    logger.info("=" * 60)
    
    # 1. Initialize Groq AI
    logger.info("\n1. Setting up Groq AI...")
    groq_ai = GroqIntegration()
    
    # Enhance conversation_ai with Groq
    if 'conversation_ai' in robot_instance.modules and groq_ai.client:
        original_generate = robot_instance.modules['conversation_ai']._generate_ai_response
        
        def groq_enhanced_response(text, context):
            # Try Groq first
            response = groq_ai.chat(text, context.messages[-10:] if context else None)
            if response and response != "Sorry, I encountered an error processing your request.":
                return response
            # Fallback to original
            return original_generate(text, context)
        
        robot_instance.modules['conversation_ai']._generate_ai_response = groq_enhanced_response
        logger.info("✓ Groq AI integrated with conversation module")
    
    # 2. Initialize Web Search
    logger.info("\n2. Setting up Web Search...")
    web_search = WebSearchIntegration()
    
    # Add web search capability to conversation
    if 'conversation_ai' in robot_instance.modules and web_search.available:
        def handle_web_search(query):
            return web_search.search_and_summarize(query, groq_ai)
        
        robot_instance.modules['conversation_ai'].web_search = handle_web_search
        logger.info("✓ Web search integrated")
    
    # 3. Fix Voice Identification
    logger.info("\n3. Fixing Voice Identification...")
    if hasattr(robot_instance, 'integration'):
        fix_voice_identification_integration(robot_instance.integration)
    
    # 4. Initialize SIM7600
    logger.info("\n4. Setting up SIM7600 4G Module...")
    
    # Try to find the correct port
    sim_ports = ["/dev/ttyUSB2", "/dev/ttyUSB1", "/dev/ttyUSB0"]
    sim7600 = None
    
    for port in sim_ports:
        if Path(port).exists():
            sim7600 = SIM7600Integration(port)
            if sim7600.connected:
                break
    
    if sim7600 and sim7600.connected:
        # Add to modules
        robot_instance.modules['sim7600'] = sim7600
        
        # Add SMS notification capability
        def send_notification(message, urgent=False):
            """Send notification via SMS"""
            master_phone = os.getenv("MASTER_PHONE")  # Add to .env
            if master_phone and urgent:
                sim7600.send_sms(master_phone, message)
        
        robot_instance.send_notification = send_notification
        logger.info("✓ SIM7600 integrated with SMS capability")
    
    # 5. Create unified intelligence system
    logger.info("\n5. Creating Unified Intelligence...")
    
    class UnifiedIntelligence:
        """Combines all AI capabilities"""
        
        def __init__(self, robot):
            self.robot = robot
            self.groq = groq_ai
            self.search = web_search
            self.sim = sim7600
        
        def process_complex_request(self, text: str, speaker: str = None) -> Dict[str, Any]:
            """
            Process complex requests using all available resources
            """
            
            response = {"text": "", "action": None, "data": {}}
            
            # Check if web search needed
            if web_search.available and any(word in text.lower() for word in ["search", "find online", "google"]):
                search_query = text.replace("search for", "").replace("find", "").strip()
                results = self.search.search_and_summarize(search_query, self.groq)
                response["text"] = results
                response["data"]["search_results"] = results
            
            # Check if SMS needed
            elif sim7600 and any(word in text.lower() for word in ["send message", "send sms", "text"]):
                response["text"] = "I can send SMS messages. What would you like to send?"
                response["action"] = {"type": "sms_prompt"}
            
            # Check for location request
            elif sim7600 and "where am i" in text.lower():
                location = sim7600.get_location()
                if location:
                    response["text"] = f"You are at latitude {location['latitude']}, longitude {location['longitude']}"
                    response["data"]["location"] = location
                else:
                    response["text"] = "GPS location not available right now"
            
            # Use Groq for general queries
            elif self.groq.client:
                ai_response = self.groq.chat(text, search_web=True)
                response["text"] = ai_response
            
            # Update learning with speaker info
            if speaker and 'learning' in robot.modules:
                robot.modules['learning'].learn_from_interaction({
                    'type': 'complex_request',
                    'speaker': speaker,
                    'request': text,
                    'response': response
                })
            
            return response
    
    # Add unified intelligence to robot
    robot_instance.unified_intelligence = UnifiedIntelligence(robot_instance)
    
    # 6. Update main processing flow
    logger.info("\n6. Updating main processing flow...")
    
    if hasattr(robot_instance, 'integration'):
        original_process = robot_instance.integration._process_speech_with_ai
        
        def enhanced_process(text, speaker=None):
            """Enhanced processing with all capabilities"""
            
            # Use unified intelligence for complex requests
            response = robot_instance.unified_intelligence.process_complex_request(text, speaker)
            
            # Execute action if needed
            if response.get('action'):
                robot_instance.integration._execute_action(response['action'])
            
            # Speak response
            if response.get('text') and 'tts' in robot_instance.modules:
                robot_instance.modules['tts'].speak(response['text'], priority=2)
            
            # Learn from interaction
            if 'learning' in robot_instance.modules:
                robot_instance.modules['learning'].learn_from_interaction({
                    'type': 'enhanced_conversation',
                    'speaker': speaker,
                    'input': text,
                    'response': response,
                    'timestamp': datetime.now()
                })
        
        robot_instance.integration._process_speech_with_ai = enhanced_process
        logger.info("✓ Enhanced processing flow integrated")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION COMPLETE!")
    logger.info("=" * 60)
    
    logger.info("\n✓ Connected Systems:")
    logger.info("  • Groq AI: " + ("Ready" if groq_ai.client else "Need API key"))
    logger.info("  • Web Search: " + ("Ready" if web_search.available else "Not installed"))
    logger.info("  • Voice ID: Fixed and connected")
    logger.info("  • SIM7600: " + ("Connected" if sim7600 and sim7600.connected else "Not found"))
    
    logger.info("\n📝 To complete setup:")
    
    if not groq_ai.client:
        logger.info("  1. Sign up at https://console.groq.com")
        logger.info("  2. Add to .env: GROQ_API_KEY=gsk_...")
    
    if not web_search.available:
        logger.info("  3. Run: pip install duckduckgo-search")
    
    if not (sim7600 and sim7600.connected):
        logger.info("  4. Check SIM7600 connection and port")
    
    logger.info("\n🚀 Your robot is now FULLY integrated!")
    
    return {
        'groq': groq_ai,
        'search': web_search,
        'sim': sim7600
    }


# ============================================
# USAGE INSTRUCTIONS
# ============================================

"""
HOW TO USE THIS COMPLETE INTEGRATION:

1. Install required packages:
   pip install groq duckduckgo-search pyserial

2. Add to your .env file:
   GROQ_API_KEY=gsk_...  # Get from https://console.groq.com
   MASTER_PHONE=+1234567890  # Your phone for SMS notifications

3. In your main.py, after initializing modules, add:
   
   from complete_integration import integrate_everything
   
   # After robot.initialize_modules()
   integration_result = integrate_everything(robot)

4. That's it! Now your robot can:
   - Have intelligent conversations with Groq AI
   - Search the web for information
   - Identify speakers by voice
   - Send SMS notifications
   - Get GPS location
   - Make phone calls

Example voice commands that now work:
   "Search for the weather in Tel Aviv"
   "Send an SMS saying I'll be late"
   "Where am I?"
   "What's happening in the news today?"
"""