"""
SIM7600X Server Connectivity Module
==================================
Handles 4G/LTE connectivity using Waveshare SIM7600X module.
Provides server connection, HTTP requests, and network management.

Hardware Connection:
- SIM7600X connected to Raspberry Pi GPIO pins 0&1 (UART0)
- Power control via GPIO (optional)
- Status monitoring via GPIO (optional)

Author: Robot AI System
Date: 2025-09-24
"""

import serial
import time
import threading
import requests
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re

# Raspberry Pi 5 compatible GPIO library
try:
    from gpiozero import OutputDevice, InputDevice
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("gpiozero not available - GPIO control disabled for SIM7600X")

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


@dataclass
class NetworkStatus:
    """Network status information"""
    
    connected: bool = False
    signal_strength: int = 0  # 0-31, higher is better
    network_type: str = "unknown"  # 2G, 3G, 4G/LTE
    operator: str = "unknown"
    ip_address: str = "0.0.0.0"
    data_usage_mb: float = 0.0
    last_update: datetime = None


class SIM7600XController:
    """
    SIM7600X 4G module controller for server connectivity.
    Manages AT commands, network connection, and HTTP communication.
    """
    
    def __init__(self, brain=None):
        """
        Initialize SIM7600X controller.
        
        Args:
            brain: Reference to robot brain for callbacks
        """
        
        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing SIM7600X Controller")
        
        # Brain reference for callbacks
        self.brain = brain
        
        # Configuration from settings
        self.port = config.hardware.sim7600x_port
        self.alt_ports = config.hardware.sim7600x_alt_ports
        self.baudrate = config.hardware.sim7600x_baudrate
        self.timeout = config.hardware.sim7600x_timeout
        self.retry_attempts = config.hardware.sim7600x_retry_attempts
        self.apn = config.hardware.sim7600x_apn
        self.username = config.hardware.sim7600x_username
        self.password = config.hardware.sim7600x_password
        self.pin = config.hardware.sim7600x_pin
        
        # GPIO pins (if configured)
        self.power_pin = getattr(config.hardware, 'sim7600x_power_pin', None)
        self.reset_pin = getattr(config.hardware, 'sim7600x_reset_pin', None)
        self.status_pin = getattr(config.hardware, 'sim7600x_status_pin', None)
        
        # Serial connection
        self.serial = None
        self.connected = False
        self.network_connected = False
        
        # Network status
        self.network_status = NetworkStatus()
        
        # Threading
        self.running = False
        self.monitor_thread = None
        
        # Command queue for thread-safe operations
        self.command_queue = []
        self.command_lock = threading.Lock()
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        
        # Statistics
        self.commands_sent = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.connection_uptime = 0.0
        
        # HTTP session for connection reuse
        self.session = requests.Session()
        self.session.timeout = 30
        
        # GPIO control objects (Raspberry Pi 5 compatible)
        self.power_control = None
        self.reset_control = None
        self.status_input = None
        self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO pins for SIM7600X control using gpiozero (Raspberry Pi 5 compatible)"""
        
        if not GPIO_AVAILABLE:
            self.logger.info("gpiozero not available - GPIO control disabled")
            return
            
        try:
            # Power control pin (active high)
            if self.power_pin is not None:
                self.power_control = OutputDevice(self.power_pin, initial_value=False)
                self.logger.info(f"SIM7600X power control initialized on GPIO {self.power_pin}")
            
            # Reset control pin (active low)  
            if self.reset_pin is not None:
                self.reset_control = OutputDevice(self.reset_pin, initial_value=True)
                self.logger.info(f"SIM7600X reset control initialized on GPIO {self.reset_pin}")
                
            # Status monitoring pin (input)
            if self.status_pin is not None:
                self.status_input = InputDevice(self.status_pin)
                self.logger.info(f"SIM7600X status monitoring initialized on GPIO {self.status_pin}")
                
        except Exception as e:
            self.logger.warning(f"GPIO pins may be in use by system - GPIO control disabled: {e}")
            self.power_control = None
            self.reset_control = None
            self.status_input = None
    
    def power_on_module(self) -> bool:
        """
        Power on the SIM7600X module using GPIO control (Raspberry Pi 5 compatible).
        
        Returns:
            bool: True if power on sequence completed
        """
        
        if not self.power_control:
            self.logger.warning("SIM7600X power control not available")
            return True  # Assume module is powered
            
        try:
            self.logger.info("Powering on SIM7600X module...")
            
            # Power on sequence: pull power pin high for 2 seconds
            self.power_control.on()
            time.sleep(2.0)
            self.power_control.off()
            
            # Wait for module to boot
            self.logger.info("Waiting for SIM7600X to boot...")
            time.sleep(10.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to power on SIM7600X: {e}")
            return False
    
    def reset_module(self) -> bool:
        """
        Reset the SIM7600X module using GPIO control (Raspberry Pi 5 compatible).
        
        Returns:
            bool: True if reset completed
        """
        
        if not self.reset_control:
            self.logger.warning("SIM7600X reset control not available")
            return False
            
        try:
            self.logger.info("Resetting SIM7600X module...")
            
            # Reset sequence: pull reset low for 500ms
            self.reset_control.off()  # Active low reset
            time.sleep(0.5)
            self.reset_control.on()   # Release reset
            
            # Wait for module to restart
            time.sleep(15.0)
            
            self.connected = False
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset SIM7600X: {e}")
            return False
        
    def connect(self) -> bool:
        """
        Connect to SIM7600X module and establish network connection.
        
        Returns:
            bool: True if connected successfully
        """
        
        try:
            # Try to find and connect to SIM7600X
            if not self._find_and_connect_serial():
                return False
            
            # Initialize module
            if not self._initialize_module():
                return False
                
            # Configure network
            if not self._configure_network():
                return False
                
            # Start monitoring thread
            self._start_monitoring()
            
            self.connected = True
            self.logger.info("SIM7600X controller connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to SIM7600X: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from SIM7600X and cleanup resources"""
        
        try:
            self.running = False
            
            # Wait for monitoring thread to stop
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # Disconnect network if connected
            if self.network_connected:
                self._send_at_command("AT+NETCLOSE")
            
            # Close serial connection
            if self.serial and self.serial.is_open:
                self.serial.close()
            
            # Close HTTP session
            self.session.close()
            
            self.connected = False
            self.network_connected = False
            self.logger.info("SIM7600X controller disconnected")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    def _find_and_connect_serial(self) -> bool:
        """
        Find and connect to SIM7600X serial port.
        Tries primary port first, then alternatives.
        
        Returns:
            bool: True if serial connection established
        """
        
        # Try primary port first
        ports_to_try = [self.port] + self.alt_ports
        
        for port in ports_to_try:
            try:
                if not Path(port).exists():
                    self.logger.debug(f"Port {port} does not exist")
                    continue
                
                self.logger.info(f"Attempting connection to {port}")
                
                # Open serial connection
                self.serial = serial.Serial(
                    port=port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                # Test with AT command
                if self._send_at_command("AT", expected_response="OK"):
                    self.port = port  # Update to working port
                    self.logger.info(f"SIM7600X found on port {port}")
                    return True
                else:
                    self.serial.close()
                    
            except Exception as e:
                self.logger.debug(f"Failed to connect to {port}: {e}")
                if self.serial:
                    self.serial.close()
                continue
        
        self.logger.error("Could not find SIM7600X on any port")
        return False
    
    def _initialize_module(self) -> bool:
        """
        Initialize SIM7600X module with basic AT commands.
        
        Returns:
            bool: True if initialization successful
        """
        
        try:
            # Basic initialization commands
            init_commands = [
                ("AT", "OK", "Basic AT test"),
                ("ATI", "SIM7600", "Module identification"),
                ("AT+CFUN=1", "OK", "Enable full functionality"),
                ("ATE0", "OK", "Disable command echo"),
                ("AT+CMEE=2", "OK", "Enable verbose error reporting"),
            ]
            
            # Execute initialization commands
            for cmd, expected, description in init_commands:
                self.logger.debug(f"Executing: {description}")
                if not self._send_at_command(cmd, expected_response=expected):
                    self.logger.error(f"Failed: {description}")
                    return False
                time.sleep(0.5)  # Small delay between commands
            
            # Check SIM card
            if not self._check_sim_card():
                return False
            
            # Wait for network registration
            if not self._wait_for_network_registration():
                return False
            
            self.logger.info("SIM7600X module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            return False
    
    def _check_sim_card(self) -> bool:
        """
        Check SIM card status and enter PIN if required.
        
        Returns:
            bool: True if SIM card is ready
        """
        
        try:
            # Check SIM card status
            response = self._send_at_command("AT+CPIN?")
            if not response:
                return False
            
            if "READY" in response:
                self.logger.info("SIM card is ready")
                return True
            elif "SIM PIN" in response:
                if self.pin:
                    self.logger.info("Entering SIM PIN")
                    if self._send_at_command(f"AT+CPIN={self.pin}", expected_response="OK"):
                        time.sleep(2)  # Wait for PIN processing
                        return self._check_sim_card()  # Recursive check
                else:
                    self.logger.error("SIM PIN required but not configured")
                    return False
            else:
                self.logger.error(f"SIM card error: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"SIM card check failed: {e}")
            return False
    
    def _wait_for_network_registration(self, timeout: int = 60) -> bool:
        """
        Wait for network registration to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if registered to network
        """
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check network registration status
                response = self._send_at_command("AT+CREG?")
                if response and "+CREG:" in response:
                    # Parse registration status
                    # Format: +CREG: <n>,<stat>[,<lac>,<ci>]
                    # stat: 0=not registered, 1=registered home, 2=searching, 5=registered roaming
                    parts = response.split(',')
                    if len(parts) >= 2:
                        status = parts[1].strip()
                        if status in ['1', '5']:  # Registered
                            self.logger.info(f"Network registered (status: {status})")
                            return True
                
                self.logger.debug("Waiting for network registration...")
                time.sleep(2)
                
            except Exception as e:
                self.logger.debug(f"Registration check error: {e}")
                time.sleep(2)
        
        self.logger.error("Network registration timeout")
        return False
    
    def _configure_network(self) -> bool:
        """
        Configure network connection and establish data connection.
        
        Returns:
            bool: True if network configured successfully
        """
        
        try:
            # Configure APN
            apn_cmd = f'AT+CGDCONT=1,"IP","{self.apn}"'
            if not self._send_at_command(apn_cmd, expected_response="OK"):
                self.logger.error("Failed to configure APN")
                return False
                
            # Configure APN authentication if credentials provided
            if self.username and self.password:
                auth_cmd = f'AT+CGAUTH=1,1,"{self.username}","{self.password}"'
                if not self._send_at_command(auth_cmd, expected_response="OK"):
                    self.logger.warning("Failed to configure APN authentication - trying without auth")
                else:
                    self.logger.info("APN authentication configured successfully")
            
            # Activate PDP context
            if not self._send_at_command("AT+CGACT=1,1", expected_response="OK", timeout=30):
                self.logger.error("Failed to activate PDP context")
                return False
            
            # Start TCP/IP stack
            if not self._send_at_command("AT+NETOPEN", expected_response="OK", timeout=30):
                # Check if already open
                response = self._send_at_command("AT+NETOPEN")
                if "already" not in response.lower():
                    self.logger.error("Failed to start TCP/IP stack")
                    return False
            
            # Get IP address
            ip_response = self._send_at_command("AT+IPADDR")
            if ip_response and "ERROR" not in ip_response:
                # Parse IP address from response
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', ip_response)
                if ip_match:
                    self.network_status.ip_address = ip_match.group(1)
                    self.logger.info(f"Network IP: {self.network_status.ip_address}")
            
            self.network_connected = True
            self.network_status.connected = True
            self.logger.info("Network connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"Network configuration failed: {e}")
            return False
    
    def _send_at_command(self, command: str, expected_response: str = None, timeout: float = None) -> str:
        """
        Send AT command to SIM7600X and get response.
        
        Args:
            command: AT command to send
            expected_response: Expected response substring
            timeout: Command timeout (uses default if None)
            
        Returns:
            str: Response from module, empty if failed
        """
        
        if not self.serial or not self.serial.is_open:
            return ""
        
        try:
            with self.command_lock:
                # Clear input buffer
                self.serial.reset_input_buffer()
                
                # Send command
                cmd_bytes = (command + '\r\n').encode('utf-8')
                self.serial.write(cmd_bytes)
                self.serial.flush()
                
                self.commands_sent += 1
                self.bytes_sent += len(cmd_bytes)
                
                # Read response with timeout
                response_timeout = timeout or self.timeout
                start_time = time.time()
                response_lines = []
                
                while time.time() - start_time < response_timeout:
                    if self.serial.in_waiting > 0:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            response_lines.append(line)
                            self.bytes_received += len(line)
                            
                            # Check for command completion
                            if line in ['OK', 'ERROR'] or line.startswith('+CME ERROR:'):
                                break
                    else:
                        time.sleep(0.1)
                
                response = '\n'.join(response_lines)
                
                # Check expected response
                if expected_response:
                    if expected_response.upper() in response.upper():
                        return response
                    else:
                        self.logger.debug(f"Command '{command}' failed. Expected '{expected_response}', got: {response}")
                        return ""
                
                return response
                
        except Exception as e:
            self.logger.error(f"AT command '{command}' failed: {e}")
            return ""
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitoring_loop(self):
        """Background monitoring loop for network status"""
        
        while self.running:
            try:
                # Update network status
                self._update_network_status()
                
                # Notify callbacks
                for callback in self.status_callbacks:
                    try:
                        callback(self.network_status)
                    except Exception as e:
                        self.logger.error(f"Status callback error: {e}")
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _update_network_status(self):
        """Update current network status information"""
        
        try:
            # Get signal strength
            signal_response = self._send_at_command("AT+CSQ")
            if signal_response and "+CSQ:" in signal_response:
                # Parse: +CSQ: <rssi>,<ber>
                parts = signal_response.split(':')[1].split(',')
                if len(parts) >= 1:
                    rssi = int(parts[0].strip())
                    self.network_status.signal_strength = rssi
            
            # Get network operator
            cop_response = self._send_at_command("AT+COPS?")
            if cop_response and "+COPS:" in cop_response:
                # Parse operator name from response
                if '"' in cop_response:
                    operator = cop_response.split('"')[1]
                    self.network_status.operator = operator
            
            # Update timestamp
            self.network_status.last_update = datetime.now()
            
        except Exception as e:
            self.logger.debug(f"Status update error: {e}")
    
    def send_http_request(self, method: str, url: str, data: Dict[Any, Any] = None, 
                         headers: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """
        Send HTTP request via SIM7600X connection.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Request URL
            data: Request data (for POST/PUT)
            headers: HTTP headers
            
        Returns:
            dict: Response data or None if failed
        """
        
        if not self.network_connected:
            self.logger.error("Network not connected")
            return None
        
        try:
            # Prepare headers
            req_headers = {'User-Agent': 'SIM7600X-Robot/1.0'}
            if headers:
                req_headers.update(headers)
            
            # Send request
            if method.upper() == 'GET':
                response = self.session.get(url, headers=req_headers)
            elif method.upper() == 'POST':
                if data:
                    if 'Content-Type' not in req_headers:
                        req_headers['Content-Type'] = 'application/json'
                    response = self.session.post(url, json=data, headers=req_headers)
                else:
                    response = self.session.post(url, headers=req_headers)
            elif method.upper() == 'PUT':
                if data:
                    response = self.session.put(url, json=data, headers=req_headers)
                else:
                    response = self.session.put(url, headers=req_headers)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=req_headers)
            else:
                self.logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Parse response
            result = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'success': response.status_code < 400
            }
            
            # Try to parse JSON response
            try:
                result['data'] = response.json()
            except:
                result['data'] = response.text
            
            self.logger.info(f"HTTP {method} {url} -> {response.status_code}")
            return result
            
        except Exception as e:
            self.logger.error(f"HTTP request failed: {e}")
            return None
    
    def add_status_callback(self, callback: Callable):
        """
        Add callback for network status updates.
        
        Args:
            callback: Function to call with NetworkStatus object
        """
        self.status_callbacks.append(callback)
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get comprehensive network information.
        
        Returns:
            dict: Network information
        """
        
        return {
            'connected': self.network_connected,
            'signal_strength': self.network_status.signal_strength,
            'operator': self.network_status.operator,
            'ip_address': self.network_status.ip_address,
            'network_type': self.network_status.network_type,
            'uptime': self.connection_uptime,
            'commands_sent': self.commands_sent,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received
        }


def test_sim7600x_connection():
    """Test function for SIM7600X connectivity"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing SIM7600X connectivity...")
    
    # Create controller
    sim_controller = SIM7600XController()
    
    try:
        # Connect to module
        if sim_controller.connect():
            print("✅ SIM7600X connected successfully!")
            
            # Show network info
            info = sim_controller.get_network_info()
            print(f"📶 Signal: {info['signal_strength']}/31")
            print(f"🌐 IP: {info['ip_address']}")
            print(f"📡 Operator: {info['operator']}")
            
            # Test HTTP request
            print("\n🌐 Testing HTTP request...")
            response = sim_controller.send_http_request('GET', 'http://httpbin.org/ip')
            if response and response['success']:
                print(f"✅ HTTP test successful: {response['data']}")
            else:
                print("❌ HTTP test failed")
            
        else:
            print("❌ Failed to connect to SIM7600X")
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        sim_controller.disconnect()


if __name__ == "__main__":
    test_sim7600x_connection()