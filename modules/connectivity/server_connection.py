"""
Robot Server Connectivity Module
===============================
High-level interface for connecting robot to remote servers.
Supports multiple connection methods: WiFi, Ethernet, SIM7600X 4G.
Provides API communication, data synchronization, and remote control.

Connection Priority:
1. Ethernet (if available)
2. WiFi (if configured) 
3. SIM7600X 4G (fallback)

Author: Robot AI System  
Date: 2025-09-24
"""

import asyncio
import aiohttp
import json
import os
import time
import threading
import logging
import socket
import subprocess
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Import modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from hardware.sim7600x_controller import SIM7600XController
from config.settings import config


@dataclass 
class ServerConfig:
    """Server connection configuration"""
    
    # Primary server
    primary_url: str = "https://api.robotai.example.com"
    primary_api_key: str = ""
    
    # Backup servers
    backup_urls: List[str] = None
    
    # Connection settings
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Authentication
    auth_method: str = "api_key"  # api_key, oauth, jwt
    auth_token: str = ""
    
    # Data sync settings
    sync_interval: int = 300  # 5 minutes
    heartbeat_interval: int = 60  # 1 minute
    
    def __post_init__(self):
        if self.backup_urls is None:
            self.backup_urls = []


@dataclass
class ConnectionStatus:
    """Current connection status"""
    
    method: str = "none"  # ethernet, wifi, sim7600x
    connected: bool = False
    server_reachable: bool = False
    ip_address: str = "0.0.0.0"
    last_sync: Optional[datetime] = None
    uptime: float = 0.0
    data_sent: int = 0
    data_received: int = 0


class ServerConnectivity:
    """
    Main server connectivity manager.
    Handles connection priorities, failover, and API communication.
    """
    
    def __init__(self, brain=None):
        """
        Initialize server connectivity manager.
        
        Args:
            brain: Reference to robot brain
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Server Connectivity")
        
        # Brain reference
        self.brain = brain
        
        # Configuration
        self.server_config = ServerConfig()
        self._load_server_config()
        
        # Connection status
        self.status = ConnectionStatus()
        
        # Connection controllers
        self.sim7600x = None
        if config.system.enable_sim7600x:
            self.sim7600x = SIM7600XController()
        
        # HTTP session for connection reuse
        self.session = None
        
        # Threading
        self.running = False
        self.monitor_thread = None
        self.sync_thread = None
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        self.data_callbacks: List[Callable] = []
        
        # Data queues
        self.outgoing_queue = []  # Data to send to server
        self.incoming_queue = []  # Data received from server
        
        # Statistics
        self.connection_attempts = 0
        self.successful_connections = 0
        self.api_calls_made = 0
        self.api_calls_failed = 0
        
        # Connection start time for uptime calculation
        self.connection_start_time = None
    
    def _load_server_config(self):
        """Load server configuration from environment or config file"""
        
        try:
            # Load from environment variables
            if os.getenv('ROBOT_API_URL'):
                self.server_config.primary_url = os.getenv('ROBOT_API_URL')
            if os.getenv('ROBOT_API_KEY'):
                self.server_config.primary_api_key = os.getenv('ROBOT_API_KEY')
            
            # Load from config file if exists
            config_file = Path(config.system.data_dir) / 'server_config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    
                for key, value in config_data.items():
                    if hasattr(self.server_config, key):
                        setattr(self.server_config, key, value)
                        
            self.logger.info(f"Server config loaded: {self.server_config.primary_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to load server config: {e}")
    
    async def start(self) -> bool:
        """
        Start server connectivity with automatic connection management.
        
        Returns:
            bool: True if started successfully
        """
        
        try:
            self.running = True
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.server_config.timeout)
            )
            
            # Attempt connection with priority order
            if await self._establish_connection():
                # Start monitoring and sync threads
                self._start_background_tasks()
                
                self.logger.info("Server connectivity started successfully")
                return True
            else:
                self.logger.error("Failed to establish any connection")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start server connectivity: {e}")
            return False
    
    async def stop(self):
        """Stop server connectivity and cleanup resources"""
        
        try:
            self.running = False
            
            # Stop background tasks
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            if self.sync_thread:
                self.sync_thread.join(timeout=5.0)
            
            # Disconnect from SIM7600X if connected
            if self.sim7600x and self.status.method == "sim7600x":
                self.sim7600x.disconnect()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            # Update status
            self.status.connected = False
            self.status.method = "none"
            
            self.logger.info("Server connectivity stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping connectivity: {e}")
    
    async def _establish_connection(self) -> bool:
        """
        Establish connection using priority order: Ethernet -> WiFi -> SIM7600X.
        
        Returns:
            bool: True if any connection method succeeded
        """
        
        connection_methods = [
            ("ethernet", self._connect_ethernet),
            ("wifi", self._connect_wifi), 
            ("sim7600x", self._connect_sim7600x)
        ]
        
        for method_name, method_func in connection_methods:
            try:
                self.logger.info(f"Attempting {method_name} connection...")
                self.connection_attempts += 1
                
                if await method_func():
                    self.status.method = method_name
                    self.status.connected = True
                    self.successful_connections += 1
                    self.connection_start_time = time.time()
                    
                    # Test server reachability
                    if await self._test_server_connection():
                        self.status.server_reachable = True
                        self.logger.info(f"✅ Connected via {method_name}")
                        return True
                    else:
                        self.logger.warning(f"❌ {method_name} connected but server unreachable")
                        
            except Exception as e:
                self.logger.error(f"❌ {method_name} connection failed: {e}")
        
        return False
    
    async def _connect_ethernet(self) -> bool:
        """
        Test Ethernet connectivity.
        
        Returns:
            bool: True if Ethernet is connected
        """
        
        try:
            # Check if ethernet interface is up and has IP
            result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
            
            # Look for ethernet interface with IP
            for line in result.stdout.split('\n'):
                if 'eth0' in line or 'enp' in line:  # Common ethernet interface names
                    if 'UP' in line and 'inet ' in result.stdout:
                        # Get IP address
                        ip_result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                        if ip_result.returncode == 0:
                            ip = ip_result.stdout.strip().split()[0]
                            self.status.ip_address = ip
                            return True
                            
            return False
            
        except Exception as e:
            self.logger.debug(f"Ethernet check failed: {e}")
            return False
    
    async def _connect_wifi(self) -> bool:
        """
        Test WiFi connectivity.
        
        Returns:
            bool: True if WiFi is connected
        """
        
        try:
            # Check WiFi interface status
            result = subprocess.run(['iwconfig'], capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            # Look for connected WiFi with ESSID
            if 'ESSID:' in result.stdout and 'Access Point:' in result.stdout:
                # Check if we have IP
                ip_result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                if ip_result.returncode == 0:
                    ip = ip_result.stdout.strip().split()[0] 
                    if ip and ip != '127.0.0.1':
                        self.status.ip_address = ip
                        return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"WiFi check failed: {e}")
            return False
    
    async def _connect_sim7600x(self) -> bool:
        """
        Connect via SIM7600X 4G module.
        
        Returns:
            bool: True if SIM7600X connected
        """
        
        if not self.sim7600x:
            self.logger.debug("SIM7600X not available")
            return False
        
        try:
            # Connect to SIM7600X
            if self.sim7600x.connect():
                # Get IP address from SIM7600X
                network_info = self.sim7600x.get_network_info()
                if network_info['connected']:
                    self.status.ip_address = network_info['ip_address']
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"SIM7600X connection failed: {e}")
            return False
    
    async def _test_server_connection(self) -> bool:
        """
        Test connectivity to primary server.
        
        Returns:
            bool: True if server is reachable
        """
        
        try:
            # Try to reach primary server
            async with self.session.get(
                f"{self.server_config.primary_url}/health",
                headers=self._get_auth_headers()
            ) as response:
                return response.status < 400
                
        except Exception as e:
            self.logger.debug(f"Server test failed: {e}")
            
            # Try backup servers
            for backup_url in self.server_config.backup_urls:
                try:
                    async with self.session.get(f"{backup_url}/health") as response:
                        if response.status < 400:
                            # Update to use backup server
                            self.server_config.primary_url = backup_url
                            return True
                except:
                    continue
            
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            dict: Authentication headers
        """
        
        headers = {
            'User-Agent': 'RobotAI/1.0',
            'Content-Type': 'application/json'
        }
        
        if self.server_config.auth_method == "api_key" and self.server_config.primary_api_key:
            headers['X-API-Key'] = self.server_config.primary_api_key
        elif self.server_config.auth_method == "jwt" and self.server_config.auth_token:
            headers['Authorization'] = f'Bearer {self.server_config.auth_token}'
        
        return headers
    
    def _start_background_tasks(self):
        """Start background monitoring and synchronization tasks"""
        
        # Start connection monitoring
        self.monitor_thread = threading.Thread(target=self._connection_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start data synchronization
        self.sync_thread = threading.Thread(target=self._data_sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
    
    def _connection_monitor(self):
        """Background task to monitor connection health"""
        
        while self.running:
            try:
                # Update uptime
                if self.connection_start_time:
                    self.status.uptime = time.time() - self.connection_start_time
                
                # Test connectivity periodically
                asyncio.run(self._periodic_connectivity_check())
                
                # Notify status callbacks
                for callback in self.status_callbacks:
                    try:
                        callback(self.status)
                    except Exception as e:
                        self.logger.error(f"Status callback error: {e}")
                
                time.sleep(self.server_config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Connection monitor error: {e}")
                time.sleep(10)
    
    async def _periodic_connectivity_check(self):
        """Perform periodic connectivity checks"""
        
        try:
            # Quick server ping
            server_ok = await self._test_server_connection()
            
            if not server_ok and self.status.server_reachable:
                self.logger.warning("Lost server connectivity, attempting reconnection...")
                # Try to reconnect
                await self._establish_connection()
            
            self.status.server_reachable = server_ok
            
        except Exception as e:
            self.logger.debug(f"Connectivity check error: {e}")
    
    def _data_sync_loop(self):
        """Background task for data synchronization"""
        
        while self.running:
            try:
                # Process outgoing data queue
                asyncio.run(self._process_outgoing_data())
                
                # Fetch incoming data
                asyncio.run(self._fetch_incoming_data())
                
                time.sleep(self.server_config.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Data sync error: {e}")
                time.sleep(30)
    
    async def _process_outgoing_data(self):
        """Process queued outgoing data"""
        
        if not self.outgoing_queue or not self.status.server_reachable:
            return
        
        try:
            # Send up to 10 items per sync
            items_to_send = self.outgoing_queue[:10]
            
            for item in items_to_send:
                success = await self._send_data_to_server(item)
                if success:
                    self.outgoing_queue.remove(item)
                    self.status.data_sent += 1
                    
        except Exception as e:
            self.logger.error(f"Outgoing data processing error: {e}")
    
    async def _send_data_to_server(self, data: Dict[str, Any]) -> bool:
        """
        Send data item to server.
        
        Args:
            data: Data to send
            
        Returns:
            bool: True if sent successfully
        """
        
        try:
            endpoint = data.get('endpoint', '/api/robot/data')
            url = f"{self.server_config.primary_url}{endpoint}"
            
            async with self.session.post(
                url,
                json=data,
                headers=self._get_auth_headers()
            ) as response:
                
                self.api_calls_made += 1
                
                if response.status < 400:
                    self.logger.debug(f"Data sent successfully: {data.get('type', 'unknown')}")
                    return True
                else:
                    self.api_calls_failed += 1
                    self.logger.error(f"Server error {response.status} sending data")
                    return False
                    
        except Exception as e:
            self.api_calls_failed += 1
            self.logger.error(f"Failed to send data: {e}")
            return False
    
    async def _fetch_incoming_data(self):
        """Fetch data from server for robot"""
        
        if not self.status.server_reachable:
            return
            
        try:
            url = f"{self.server_config.primary_url}/api/robot/commands"
            
            async with self.session.get(
                url,
                headers=self._get_auth_headers()
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Process commands/data
                    if isinstance(data, dict) and 'commands' in data:
                        for command in data['commands']:
                            self.incoming_queue.append(command)
                            self.status.data_received += 1
                            
                            # Notify data callbacks
                            for callback in self.data_callbacks:
                                try:
                                    callback(command)
                                except Exception as e:
                                    self.logger.error(f"Data callback error: {e}")
                    
                    self.status.last_sync = datetime.now()
                    
        except Exception as e:
            self.logger.debug(f"Failed to fetch incoming data: {e}")
    
    # Public API methods
    
    def queue_data(self, data_type: str, payload: Dict[str, Any], endpoint: str = None):
        """
        Queue data to be sent to server.
        
        Args:
            data_type: Type of data (sensor, status, log, etc.)
            payload: Data payload
            endpoint: Custom API endpoint (optional)
        """
        
        data_item = {
            'type': data_type,
            'timestamp': datetime.now().isoformat(),
            'robot_id': getattr(config, 'robot_id', 'unknown'),
            'payload': payload
        }
        
        if endpoint:
            data_item['endpoint'] = endpoint
            
        self.outgoing_queue.append(data_item)
        self.logger.debug(f"Queued {data_type} data for upload")
    
    def add_status_callback(self, callback: Callable):
        """Add callback for connection status updates"""
        self.status_callbacks.append(callback)
    
    def add_data_callback(self, callback: Callable):
        """Add callback for incoming data from server"""
        self.data_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current connectivity status.
        
        Returns:
            dict: Status information
        """
        
        return asdict(self.status)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get connectivity statistics.
        
        Returns:
            dict: Statistics
        """
        
        return {
            'connection_attempts': self.connection_attempts,
            'successful_connections': self.successful_connections,
            'api_calls_made': self.api_calls_made,
            'api_calls_failed': self.api_calls_failed,
            'outgoing_queue_size': len(self.outgoing_queue),
            'incoming_queue_size': len(self.incoming_queue),
            'success_rate': (self.successful_connections / max(1, self.connection_attempts)) * 100
        }


async def test_server_connectivity():
    """Test function for server connectivity"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Server Connectivity...")
    
    # Create connectivity manager
    server = ServerConnectivity()
    
    try:
        # Start connectivity
        if await server.start():
            print("✅ Server connectivity established!")
            
            # Show status
            status = server.get_status()
            print(f"📡 Method: {status['method']}")
            print(f"🌐 IP: {status['ip_address']}")
            print(f"🖥️ Server: {'✅' if status['server_reachable'] else '❌'}")
            
            # Test data upload
            print("\n📤 Testing data upload...")
            server.queue_data('test', {'message': 'Hello from robot!'})
            
            # Wait for sync
            await asyncio.sleep(5)
            
            # Show statistics
            stats = server.get_statistics()
            print(f"📊 API calls: {stats['api_calls_made']}")
            print(f"📊 Success rate: {stats['success_rate']:.1f}%")
            
        else:
            print("❌ Failed to establish server connectivity")
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(test_server_connectivity())