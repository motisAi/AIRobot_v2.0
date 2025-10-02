#!/usr/bin/env python3
"""
Robot Network Failover Manager
==============================
Manages network connectivity with WiFi primary and SIM7600X backup.
Automatically switches robot communication methods based on network availability.
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

class NetworkFailoverManager:
    """Manages network failover between WiFi and SIM7600X"""
    
    def __init__(self):
        self.wifi_available = False
        self.sim7600x_available = False
        self.current_connection = "none"
        self.sim7600x_port = "/dev/ttyAMA1"
        self.sim7600x_baudrate = 115200
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_wifi_connectivity(self) -> bool:
        """Check if WiFi internet is available"""
        try:
            response = requests.get("http://httpbin.org/ip", timeout=5)
            if response.status_code == 200:
                self.wifi_available = True
                return True
        except:
            pass
        
        self.wifi_available = False
        return False
    
    def check_sim7600x_connectivity(self) -> bool:
        """Check if SIM7600X has internet connectivity"""
        try:
            ser = serial.Serial(self.sim7600x_port, self.sim7600x_baudrate, timeout=3.0)
            
            # Check IP address
            ser.write(b"AT+CGPADDR=1\r\n")
            time.sleep(1)
            response = ser.read_all().decode('utf-8', errors='ignore')
            ser.close()
            
            if "+CGPADDR: 1," in response and "0.0.0.0" not in response:
                self.sim7600x_available = True
                return True
                
        except Exception as e:
            self.logger.debug(f"SIM7600X check failed: {e}")
        
        self.sim7600x_available = False
        return False
    
    def get_best_connection(self) -> str:
        """Determine the best available connection"""
        
        # Check both connections
        wifi_ok = self.check_wifi_connectivity()
        sim7600x_ok = self.check_sim7600x_connectivity()
        
        if wifi_ok:
            self.current_connection = "wifi"
            return "wifi"
        elif sim7600x_ok:
            self.current_connection = "sim7600x"
            return "sim7600x"
        else:
            self.current_connection = "none"
            return "none"
    
    def make_groq_request(self, messages: list, api_key: str) -> Optional[str]:
        """Make request to Groq API using best available connection"""
        
        connection = self.get_best_connection()
        
        if connection == "wifi":
            return self._groq_via_wifi(messages, api_key)
        elif connection == "sim7600x":
            return self._groq_via_sim7600x(messages, api_key)
        else:
            self.logger.error("No internet connection available")
            return None
    
    def _groq_via_wifi(self, messages: list, api_key: str) -> Optional[str]:
        """Make Groq request via WiFi (normal method)"""
        try:
            import groq
            client = groq.Groq(api_key=api_key)
            
            response = client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Groq via WiFi failed: {e}")
            return None
    
    def _groq_via_sim7600x(self, messages: list, api_key: str) -> Optional[str]:
        """Make Groq request via SIM7600X (backup method)"""
        
        # For now, return a fallback response
        # In production, this would use AT commands or a proxy
        self.logger.info("Using SIM7600X backup - returning offline response")
        
        # Simple fallback responses based on user input
        user_message = messages[-1].get("content", "").lower()
        
        if "hello" in user_message or "hi" in user_message:
            return "Hello! I'm running on SIM7600X backup connection."
        elif "how are you" in user_message:
            return "I'm doing well, thanks! Currently using cellular backup."
        elif "weather" in user_message:
            return "I can't check weather without full internet, but I'm still here to help!"
        else:
            return "I'm operating on cellular backup. Some features may be limited, but I'm still here to assist you."
    
    def get_connection_status(self) -> dict:
        """Get current connection status"""
        return {
            "wifi_available": self.wifi_available,
            "sim7600x_available": self.sim7600x_available,
            "current_connection": self.current_connection,
            "status": "online" if self.current_connection != "none" else "offline"
        }

def test_network_failover():
    """Test the network failover system"""
    
    print("🌐 Robot Network Failover Test")
    print("=" * 40)
    
    manager = NetworkFailoverManager()
    
    # Check current status
    print("\n📊 Checking network status...")
    status = manager.get_connection_status()
    
    print(f"   WiFi: {'✅' if status['wifi_available'] else '❌'}")
    print(f"   SIM7600X: {'✅' if status['sim7600x_available'] else '❌'}")
    print(f"   Active: {status['current_connection']}")
    
    # Test Groq communication
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print("\n🤖 Testing AI communication...")
        
        test_messages = [
            {"role": "user", "content": "Hello, are you working?"}
        ]
        
        response = manager.make_groq_request(test_messages, api_key)
        
        if response:
            print(f"✅ AI Response: {response}")
            print(f"   Connection used: {manager.current_connection}")
        else:
            print("❌ AI communication failed")
    
    return status

if __name__ == "__main__":
    test_network_failover()