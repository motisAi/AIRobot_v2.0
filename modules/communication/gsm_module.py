"""
GSM Communication Module
========================
Handles GSM/cellular communication using SIM7600X module.
Supports SMS, calls, and data connectivity.

Author: AI Robot System
Date: 2024
"""

import serial
import time
import logging
import threading
import re
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

class GSMModule:
    """
    GSM communication using SIM7600X module
    """
    
    def __init__(self, brain, port="/dev/ttyUSB2", baudrate=115200):
        """
        Initialize GSM module
        
        Args:
            brain: Robot brain instance
            port: Serial port for GSM module
            baudrate: Serial communication baudrate
        """
        self.brain = brain
        self.logger = logging.getLogger("GSM")
        self.port = port
        self.baudrate = baudrate
        
        # Serial connection
        self.serial_conn = None
        self.connected = False
        
        # Module status
        self.network_registered = False
        self.signal_strength = 0
        
        # Message handling
        self.message_callbacks: List[Callable] = []
        self.call_callbacks: List[Callable] = []
        
        # Background thread
        self.running = False
        self.monitor_thread = None
        
        self.logger.info("GSM Module initialized")
    
    def connect(self) -> bool:
        """
        Connect to GSM module
        
        Returns:
            True if connection successful
        """
        try:
            self.logger.info(f"Connecting to GSM module on {self.port}")
            
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=5,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Test connection
            if self._send_command("AT"):
                self.connected = True
                self.logger.info("✓ GSM module connected")
                
                # Initialize module
                self._initialize_module()
                
                # Start monitoring thread
                self.start_monitoring()
                
                return True
            else:
                self.logger.error("GSM module not responding")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to GSM module: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from GSM module"""
        try:
            self.stop_monitoring()
            
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.connected = False
                self.logger.info("GSM module disconnected")
                
        except Exception as e:
            self.logger.error(f"Error disconnecting GSM module: {e}")
    
    def _initialize_module(self):
        """Initialize GSM module settings"""
        try:
            # Set text mode for SMS
            self._send_command("AT+CMGF=1")
            
            # Enable network registration notifications
            self._send_command("AT+CREG=1")
            
            # Enable SMS notifications
            self._send_command("AT+CNMI=1,2,0,0,0")
            
            # Check network registration
            self._check_network_registration()
            
            # Get signal strength
            self._check_signal_strength()
            
        except Exception as e:
            self.logger.error(f"Error initializing GSM module: {e}")
    
    def _send_command(self, command: str, timeout: int = 5) -> str:
        """
        Send AT command to GSM module
        
        Args:
            command: AT command to send
            timeout: Response timeout in seconds
            
        Returns:
            Response from module
        """
        try:
            if not self.serial_conn or not self.serial_conn.is_open:
                return ""
            
            # Send command
            self.serial_conn.write((command + "\r\n").encode())
            
            # Read response
            start_time = time.time()
            response = ""
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    response += data
                    
                    if "OK" in response or "ERROR" in response:
                        break
                
                time.sleep(0.1)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error sending command {command}: {e}")
            return ""
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """
        Send SMS message
        
        Args:
            phone_number: Recipient phone number
            message: Message content
            
        Returns:
            True if SMS sent successfully
        """
        try:
            if not self.connected:
                self.logger.error("GSM module not connected")
                return False
            
            # Set SMS recipient
            response = self._send_command(f'AT+CMGS="{phone_number}"')
            
            if ">" in response:
                # Send message content
                self.serial_conn.write(message.encode())
                self.serial_conn.write(bytes([26]))  # Ctrl+Z to send
                
                # Wait for confirmation
                response = self._read_response(timeout=30)
                
                if "+CMGS:" in response:
                    self.logger.info(f"SMS sent to {phone_number}")
                    return True
                else:
                    self.logger.error(f"Failed to send SMS: {response}")
                    return False
            else:
                self.logger.error(f"Failed to set SMS recipient: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending SMS: {e}")
            return False
    
    def make_call(self, phone_number: str) -> bool:
        """
        Make voice call
        
        Args:
            phone_number: Number to call
            
        Returns:
            True if call initiated successfully
        """
        try:
            if not self.connected:
                self.logger.error("GSM module not connected")
                return False
            
            response = self._send_command(f"ATD{phone_number};")
            
            if "OK" in response:
                self.logger.info(f"Call initiated to {phone_number}")
                return True
            else:
                self.logger.error(f"Failed to make call: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error making call: {e}")
            return False
    
    def hang_up(self) -> bool:
        """
        Hang up current call
        
        Returns:
            True if hang up successful
        """
        try:
            response = self._send_command("ATH")
            return "OK" in response
        except Exception as e:
            self.logger.error(f"Error hanging up: {e}")
            return False
    
    def _check_network_registration(self):
        """Check network registration status"""
        try:
            response = self._send_command("AT+CREG?")
            
            # Parse response: +CREG: n,stat
            match = re.search(r'\+CREG: \d+,(\d+)', response)
            if match:
                status = int(match.group(1))
                self.network_registered = status in [1, 5]  # 1=home, 5=roaming
                
                if self.network_registered:
                    self.logger.info("✓ Network registered")
                else:
                    self.logger.warning("✗ Network not registered")
            
        except Exception as e:
            self.logger.error(f"Error checking network registration: {e}")
    
    def _check_signal_strength(self):
        """Check signal strength"""
        try:
            response = self._send_command("AT+CSQ")
            
            # Parse response: +CSQ: rssi,ber
            match = re.search(r'\+CSQ: (\d+),\d+', response)
            if match:
                rssi = int(match.group(1))
                if rssi != 99:  # 99 = unknown
                    self.signal_strength = rssi
                    self.logger.info(f"Signal strength: {rssi}/31")
                else:
                    self.signal_strength = 0
            
        except Exception as e:
            self.logger.error(f"Error checking signal strength: {e}")
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("GSM monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            self.logger.info("GSM monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    self._process_unsolicited_response(data)
                
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _process_unsolicited_response(self, data: str):
        """Process unsolicited responses from module"""
        try:
            # SMS received
            if "+CMT:" in data:
                self._handle_incoming_sms(data)
            
            # Incoming call
            elif "RING" in data:
                self._handle_incoming_call(data)
            
            # Network registration update
            elif "+CREG:" in data:
                self._handle_network_update(data)
            
        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
    
    def _handle_incoming_sms(self, data: str):
        """Handle incoming SMS"""
        try:
            # Parse SMS data
            lines = data.split('\n')
            for i, line in enumerate(lines):
                if "+CMT:" in line:
                    # Extract sender and timestamp
                    match = re.search(r'\+CMT: "([^"]+)","[^"]*","([^"]+)"', line)
                    if match and i + 1 < len(lines):
                        sender = match.group(1)
                        timestamp = match.group(2)
                        message = lines[i + 1].strip()
                        
                        sms_data = {
                            'sender': sender,
                            'message': message,
                            'timestamp': timestamp,
                            'received_at': time.time()
                        }
                        
                        self.logger.info(f"SMS received from {sender}: {message}")
                        
                        # Notify callbacks
                        for callback in self.message_callbacks:
                            try:
                                callback(sms_data)
                            except Exception as e:
                                self.logger.error(f"Error in SMS callback: {e}")
                        
                        # Notify brain
                        if hasattr(self.brain, 'process_event'):
                            self.brain.process_event({
                                'type': 'sms_received',
                                'data': sms_data
                            })
                        
                        break
            
        except Exception as e:
            self.logger.error(f"Error handling incoming SMS: {e}")
    
    def _handle_incoming_call(self, data: str):
        """Handle incoming call"""
        try:
            # Extract caller number if available
            caller = "unknown"
            match = re.search(r'\+CLIP: "([^"]+)"', data)
            if match:
                caller = match.group(1)
            
            call_data = {
                'caller': caller,
                'timestamp': time.time()
            }
            
            self.logger.info(f"Incoming call from {caller}")
            
            # Notify callbacks
            for callback in self.call_callbacks:
                try:
                    callback(call_data)
                except Exception as e:
                    self.logger.error(f"Error in call callback: {e}")
            
            # Notify brain
            if hasattr(self.brain, 'process_event'):
                self.brain.process_event({
                    'type': 'incoming_call',
                    'data': call_data
                })
            
        except Exception as e:
            self.logger.error(f"Error handling incoming call: {e}")
    
    def _handle_network_update(self, data: str):
        """Handle network registration update"""
        try:
            match = re.search(r'\+CREG: (\d+)', data)
            if match:
                status = int(match.group(1))
                was_registered = self.network_registered
                self.network_registered = status in [1, 5]
                
                if self.network_registered != was_registered:
                    if self.network_registered:
                        self.logger.info("Network registered")
                    else:
                        self.logger.warning("Network lost")
            
        except Exception as e:
            self.logger.error(f"Error handling network update: {e}")
    
    def _read_response(self, timeout: int = 5) -> str:
        """Read response from serial connection"""
        try:
            start_time = time.time()
            response = ""
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    response += data
                
                time.sleep(0.1)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error reading response: {e}")
            return ""
    
    def add_message_callback(self, callback: Callable):
        """Add callback for incoming messages"""
        self.message_callbacks.append(callback)
    
    def add_call_callback(self, callback: Callable):
        """Add callback for incoming calls"""
        self.call_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current GSM status"""
        return {
            'connected': self.connected,
            'network_registered': self.network_registered,
            'signal_strength': self.signal_strength,
            'port': self.port
        }