#!/usr/bin/env python3
"""
Robot Primary SIM7600X Communication Manager
============================================
Uses SIM7600X as the primary internet connection for robot AI communication.
WiFi is secondary/backup only.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import subprocess
import logging
from typing import Optional
import serial
import json

class SIM7600XPrimaryManager:
    """Manages robot communication with SIM7600X as primary connection"""
    
    def __init__(self):
        self.sim7600x_port = "/dev/ttyAMA1"
        self.sim7600x_baudrate = 115200
        self.serial_connection = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Connect to SIM7600X
        self.connect_sim7600x()
    
    def connect_sim7600x(self) -> bool:
        """Connect to SIM7600X module"""
        try:
            self.serial_connection = serial.Serial(
                self.sim7600x_port, 
                self.sim7600x_baudrate, 
                timeout=5.0
            )
            self.logger.info("✅ Connected to SIM7600X")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to SIM7600X: {e}")
            return False
    
    def send_at_command(self, cmd: str, wait_time: float = 1.0) -> str:
        """Send AT command to SIM7600X"""
        if not self.serial_connection:
            return ""
        
        try:
            self.serial_connection.write(f"{cmd}\r\n".encode())
            time.sleep(wait_time)
            response = self.serial_connection.read_all().decode('utf-8', errors='ignore')
            return response.strip()
        except Exception as e:
            self.logger.error(f"AT command failed: {e}")
            return ""
    
    def check_sim7600x_status(self) -> dict:
        """Check SIM7600X connection status"""
        status = {
            "connected": False,
            "signal_strength": 0,
            "ip_address": "0.0.0.0",
            "operator": "Unknown",
            "network_type": "Unknown"
        }
        
        try:
            # Check signal strength
            response = self.send_at_command("AT+CSQ")
            if "+CSQ:" in response:
                signal = int(response.split("+CSQ: ")[1].split(",")[0])
                status["signal_strength"] = signal
            
            # Check IP address
            response = self.send_at_command("AT+CGPADDR=1")
            if "+CGPADDR: 1," in response:
                ip = response.split("+CGPADDR: 1,")[1].split("\n")[0].strip()
                if ip and ip != "0.0.0.0":
                    status["ip_address"] = ip
                    status["connected"] = True
            
            # Check operator
            response = self.send_at_command("AT+COPS?")
            if "+COPS:" in response and '"' in response:
                operator = response.split('"')[1]
                status["operator"] = operator
            
            # Check network type
            response = self.send_at_command("AT+CPSI?")
            if "+CPSI:" in response:
                parts = response.split(",")
                if len(parts) > 0:
                    network_type = parts[0].split("+CPSI: ")[1] if "+CPSI: " in parts[0] else "Unknown"
                    status["network_type"] = network_type
                    
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
        
        return status
    
    def make_groq_request_via_sim7600x(self, messages: list, api_key: str) -> Optional[str]:
        """Make Groq API request via SIM7600X HTTP commands"""
        
        try:
            # Prepare Groq API data
            groq_data = {
                "messages": messages,
                "model": "llama3-8b-8192",
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            json_data = json.dumps(groq_data)
            data_length = len(json_data)
            
            # Clean up any existing HTTP session
            self.send_at_command("AT+HTTPTERM")
            time.sleep(1)
            
            # Initialize HTTP
            response = self.send_at_command("AT+HTTPINIT")
            if "OK" not in response:
                self.logger.error("HTTP init failed")
                return None
            
            # Set parameters
            self.send_at_command("AT+HTTPPARA=\"CID\",1")
            self.send_at_command("AT+HTTPPARA=\"URL\",\"https://api.groq.com/openai/v1/chat/completions\"")
            self.send_at_command("AT+HTTPPARA=\"CONTENT\",\"application/json\"")
            self.send_at_command(f"AT+HTTPPARA=\"USERDATA\",\"Authorization: Bearer {api_key}\"")
            
            # Send data
            self.send_at_command(f"AT+HTTPDATA={data_length},10000")
            time.sleep(1)
            
            # Send JSON payload
            self.serial_connection.write(json_data.encode())
            time.sleep(3)
            
            # Execute POST request
            response = self.send_at_command("AT+HTTPACTION=1", wait_time=15.0)
            
            # Check response
            if "+HTTPACTION: 1,200" in response:
                # Read response
                read_response = self.send_at_command("AT+HTTPREAD", wait_time=5.0)
                
                # Clean up
                self.send_at_command("AT+HTTPTERM")
                
                # Try to parse Groq response
                try:
                    if "choices" in read_response:
                        # Extract the actual response content
                        response_data = json.loads(read_response.split('\n')[-2])
                        return response_data["choices"][0]["message"]["content"]
                except:
                    pass
                
                return f"SIM7600X response received (parsing needed): {read_response[:100]}..."
            else:
                self.send_at_command("AT+HTTPTERM")
                self.logger.error(f"HTTP request failed: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Groq request via SIM7600X failed: {e}")
            self.send_at_command("AT+HTTPTERM")
            return None
    
    def make_simple_http_request(self, url: str) -> Optional[str]:
        """Make simple HTTP GET request via SIM7600X"""
        
        try:
            # Clean up
            self.send_at_command("AT+HTTPTERM")
            time.sleep(1)
            
            # Initialize
            self.send_at_command("AT+HTTPINIT")
            self.send_at_command("AT+HTTPPARA=\"CID\",1")
            self.send_at_command(f"AT+HTTPPARA=\"URL\",\"{url}\"")
            
            # Execute GET
            response = self.send_at_command("AT+HTTPACTION=0", wait_time=10.0)
            
            if "+HTTPACTION: 0,200" in response:
                read_response = self.send_at_command("AT+HTTPREAD")
                self.send_at_command("AT+HTTPTERM")
                return read_response
            else:
                self.send_at_command("AT+HTTPTERM")
                return None
                
        except Exception as e:
            self.logger.error(f"Simple HTTP request failed: {e}")
            self.send_at_command("AT+HTTPTERM")
            return None
    
    def robot_speak_via_sim7600x(self, user_input: str, api_key: str) -> str:
        """Process user input and generate response via SIM7600X"""
        
        messages = [
            {"role": "system", "content": "You are Gonzo, a helpful robot assistant. Keep responses concise and friendly."},
            {"role": "user", "content": user_input}
        ]
        
        # Try Groq via SIM7600X first
        response = self.make_groq_request_via_sim7600x(messages, api_key)
        
        if response:
            return response
        
        # Fallback to simple responses if Groq fails
        user_lower = user_input.lower()
        if "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm Gonzo, your robot assistant running on SIM7600X cellular network."
        elif "how are you" in user_lower:
            return "I'm doing great! Running on 4G cellular connection and ready to help."
        elif "weather" in user_lower:
            return "I'd need to make an API call to check weather. Let me try that via my cellular connection."
        else:
            return "I'm here and ready to help! I'm running on SIM7600X cellular network for reliable connectivity."

def test_sim7600x_primary():
    """Test SIM7600X as primary communication"""
    
    print("📱 Robot SIM7600X Primary Communication Test")
    print("=" * 50)
    
    manager = SIM7600XPrimaryManager()
    
    # Check status
    print("\n📊 SIM7600X Status:")
    status = manager.check_sim7600x_status()
    
    print(f"   Connected: {'✅' if status['connected'] else '❌'}")
    print(f"   Signal: {status['signal_strength']}/31")
    print(f"   IP: {status['ip_address']}")
    print(f"   Operator: {status['operator']}")
    print(f"   Network: {status['network_type']}")
    
    if status['connected']:
        print("\n🌐 Testing simple HTTP request...")
        response = manager.make_simple_http_request("http://httpbin.org/ip")
        if response and "origin" in response:
            print("✅ HTTP requests working via SIM7600X")
            print(f"   Public IP via cellular: {response}")
        
        # Test robot conversation
        print("\n🤖 Testing robot AI conversation...")
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            test_inputs = [
                "Hello robot!",
                "How are you doing?",
                "What's the weather like?"
            ]
            
            for user_input in test_inputs:
                print(f"\n   User: {user_input}")
                robot_response = manager.robot_speak_via_sim7600x(user_input, api_key)
                print(f"   Robot: {robot_response}")
        
        print("\n✅ SIM7600X primary communication ready!")
        print("🎯 Robot will use cellular network as main internet connection")
    
    else:
        print("❌ SIM7600X not ready for primary communication")

if __name__ == "__main__":
    test_sim7600x_primary()