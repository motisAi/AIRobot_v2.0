"""
ESP32 Controller Module
=======================
Handles serial communication with ESP32 for hardware control.
Manages motors, servos, sensors, and GPIO operations.


"""

import serial
import serial.tools.list_ports
import time
import threading
import queue
import logging
import json
import struct
from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import config, hardware_config


class ESP32Commands(IntEnum):
    """ESP32 command codes"""
    
    # Movement
    MOVE_FORWARD = 0x01
    MOVE_BACKWARD = 0x02
    TURN_LEFT = 0x03
    TURN_RIGHT = 0x04
    STOP = 0x05
    
    # Servo control
    SERVO_SET = 0x10
    SERVO_SWEEP = 0x11
    
    # Sensors
    SENSOR_READ = 0x20
    SENSOR_STREAM = 0x21
    
    # GPIO
    GPIO_SET = 0x30
    GPIO_READ = 0x31
    GPIO_PWM = 0x32
    
    # System
    PING = 0x40
    STATUS = 0x41
    RESET = 0x42
    CONFIG = 0x43
    
    # Feedback
    ACK = 0x50
    ERROR = 0x51
    DATA = 0x52


@dataclass
class SensorData:
    """Sensor reading data structure"""
    
    sensor_type: str
    value: float
    unit: str
    timestamp: float
    

class ESP32Controller:
    """
    ESP32 communication controller for hardware interface.
    Handles serial protocol, command queuing, and sensor data.
    """
    
    def __init__(self, brain=None):
        """
        Initialize ESP32 controller
        
        Args:
            brain: Reference to robot brain
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing ESP32 Controller")
        
        # Brain reference
        self.brain = brain
        
        # Serial configuration
        self.port = hardware_config.esp32_port
        self.baudrate = hardware_config.esp32_baudrate
        self.timeout = hardware_config.esp32_timeout
        
        # Serial connection
        self.serial = None
        self.connected = False
        
        # Command queue
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Sensor data
        self.sensor_data = {}
        self.sensor_callbacks = {}
        
        # Motor state
        self.motor_state = {
            'left': 0,
            'right': 0,
            'speed': 50  # Default speed percentage
        }
        
        # Servo positions
        self.servo_positions = {}
        for i, pin in enumerate(hardware_config.servo_pins):
            self.servo_positions[i] = 90  # Center position
        
        # Threading
        self.running = False
        self.serial_thread = None
        self.command_thread = None
        
        # Statistics
        self.commands_sent = 0
        self.commands_failed = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        
        # Auto-reconnect
        self.auto_reconnect = True
        self.reconnect_attempts = 0
        
        # Simulation mode (when hardware not available)
        self.simulation_mode = False
        
    def connect(self) -> bool:
        """
        Connect to ESP32 via serial
        
        Returns:
            bool: True if connected successfully
        """
        
        try:
            # Find ESP32 port if not specified
            if not self.port or self.port == 'auto':
                self.port = self._find_esp32_port()
            
            if not self.port:
                self.logger.warning("ESP32 not found - running in simulation mode")
                self.connected = False
                self.auto_reconnect = False  # Disable auto-reconnect when no port found
                # Continue to start controller in simulation mode
            else:
                # Try to open serial connection
                try:
                    self.serial = serial.Serial(
                        port=self.port,
                        baudrate=self.baudrate,
                        timeout=self.timeout,
                        write_timeout=self.timeout
                    )
                    
                    # Clear buffers
                    self.serial.reset_input_buffer()
                    self.serial.reset_output_buffer()
                    
                    # Wait for ESP32 to initialize
                    time.sleep(2.0)
                    
                    # Send ping to verify connection
                    if self._ping():
                        self.connected = True
                        self.logger.info(f"ESP32 connected on {self.port}")
                    else:
                        self.logger.warning("ESP32 not responding - continuing without hardware")
                        self.connected = False
                        
                except Exception as e:
                    self.logger.warning(f"ESP32 connection failed: {e} - continuing in simulation mode")
                    self.connected = False
        
            self.running = True
        
            # Disable auto-reconnect in simulation mode to prevent excessive reconnection attempts
            if not self.connected:
                self.auto_reconnect = False
        
            # Start serial communication thread (handles both connected and disconnected states)
            self.serial_thread = threading.Thread(target=self._serial_loop)
            self.serial_thread.daemon = True
            self.serial_thread.start()
        
            # Start command processing thread
            self.command_thread = threading.Thread(target=self._command_loop)
            self.command_thread.daemon = True
            self.command_thread.start()
        
            status = "connected" if self.connected else "simulation mode"
            self.logger.info(f"ESP32 controller started in {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start ESP32 controller: {e}")
            return False
    
    def stop(self):
        """Stop ESP32 controller"""
        
        self.running = False
        
        # Stop motors before disconnecting
        self.stop_motors()
        
        # Wait for threads only if they were started
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2.0)
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=2.0)
        
        # Disconnect
        if self.serial and hasattr(self.serial, 'is_open') and self.serial.is_open:
            self.serial.close()
        
        self.connected = False
        self.logger.info("ESP32 controller stopped")
    
    def _serial_loop(self):
        """Serial communication loop"""
        
        while self.running:
            try:
                if not self.connected:
                    if self.auto_reconnect:
                        self._attempt_reconnect()
                    time.sleep(1.0)
                    continue
                
                # Read incoming data
                if self.serial and self.serial.in_waiting > 0:
                    self._read_serial()
                
                time.sleep(0.01)  # Small delay to prevent CPU hogging
                
            except serial.SerialException as e:
                self.logger.error(f"Serial error: {e}")
                self.connected = False
            except Exception as e:
                self.logger.error(f"Serial loop error: {e}")
                time.sleep(0.1)
    
    def _find_esp32_port(self) -> Optional[str]:
        """
        Find ESP32 serial port automatically
        
        Returns:
            str: Port name or None if not found
        """
        
        import serial.tools.list_ports
        
        # Common ESP32 port patterns
        esp32_patterns = [
            'CP210',  # CP2102 USB-to-serial
            'CH340',  # CH340 USB-to-serial
            'FTDI',   # FTDI USB-to-serial
            'ESP32',  # Direct ESP32 identification
        ]
        
        try:
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                # Check if port description matches ESP32 patterns
                description = (port.description or '').upper()
                manufacturer = (port.manufacturer or '').upper()
                
                for pattern in esp32_patterns:
                    if pattern in description or pattern in manufacturer:
                        self.logger.info(f"Found potential ESP32 on {port.device}: {port.description}")
                        return port.device
                        
            # Try common ESP32 ports on Linux
            common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
            for port in common_ports:
                try:
                    # Test if port exists and is accessible
                    test_ser = serial.Serial(port, 115200, timeout=0.5)
                    test_ser.close()
                    self.logger.info(f"Found accessible port: {port}")
                    return port
                except:
                    continue
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ESP32 port: {e}")
            return None
    
    def _command_loop(self):
        """Command processing loop"""
        
        while self.running:
            try:
                # Get command from queue
                command = self.command_queue.get(timeout=0.1)
                
                if self.connected:
                    self._send_command(command)
                else:
                    # Simulation mode - log commands but don't send
                    cmd_name = command['cmd'].name if hasattr(command['cmd'], 'name') else str(command['cmd'])
                    self.logger.debug(f"[SIMULATION] Would send command: {cmd_name}")
                    self.commands_sent += 1  # Count as sent in simulation
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Command loop error: {e}")
    
    def _send_command(self, command: Dict[str, Any]):
        """
        Send command to ESP32
        
        Args:
            command: Command dictionary with 'cmd' and 'data'
        """
        
        try:
            cmd_code = command['cmd']
            cmd_data = command.get('data', b'')
            
            # Create packet
            packet = self._create_packet(cmd_code, cmd_data)
            
            # Send packet
            self.serial.write(packet)
            self.bytes_sent += len(packet)
            self.commands_sent += 1
            
            self.logger.debug(f"Sent command: {cmd_code.name if hasattr(cmd_code, 'name') else cmd_code}")
            
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            self.commands_failed += 1
    
    def _create_packet(self, cmd_code: int, data: bytes) -> bytes:
        """
        Create command packet
        
        Args:
            cmd_code: Command code
            data: Command data
            
        Returns:
            Complete packet with header and checksum
        """
        
        # Packet format: [START(0xFF)][LENGTH][CMD][DATA][CHECKSUM]
        packet = bytearray()
        packet.append(0xFF)  # Start byte
        packet.append(len(data) + 1)  # Length (cmd + data)
        packet.append(cmd_code)  # Command
        packet.extend(data)  # Data
        
        # Calculate checksum (XOR of all bytes except start)
        checksum = 0
        for b in packet[1:]:
            checksum ^= b
        packet.append(checksum)
        
        return bytes(packet)
    
    def _read_serial(self):
        """Read and process serial data"""
        
        try:
            # Read available data
            data = self.serial.read(self.serial.in_waiting)
            self.bytes_received += len(data)
            
            # Process data (simple parsing, can be improved)
            if data[0] == 0xFF:  # Start byte
                length = data[1] if len(data) > 1 else 0
                
                if len(data) >= length + 3:  # Complete packet
                    cmd = data[2]
                    payload = data[3:3+length-1]
                    checksum = data[3+length-1]
                    
                    # Verify checksum
                    calc_checksum = length ^ cmd
                    for b in payload:
                        calc_checksum ^= b
                    
                    if calc_checksum == checksum:
                        self._process_response(cmd, payload)
                    else:
                        self.logger.warning("Checksum mismatch")
                        
        except Exception as e:
            self.logger.error(f"Serial read error: {e}")
    
    def _process_response(self, cmd: int, data: bytes):
        """
        Process response from ESP32
        
        Args:
            cmd: Response command
            data: Response data
        """
        
        if cmd == ESP32Commands.ACK:
            self.logger.debug("Command acknowledged")
            
        elif cmd == ESP32Commands.ERROR:
            error_code = data[0] if data else 0
            self.logger.error(f"ESP32 error: {error_code}")
            
        elif cmd == ESP32Commands.DATA:
            # Sensor data
            self._process_sensor_data(data)
            
        elif cmd == ESP32Commands.STATUS:
            # Status update
            self._process_status(data)
    
    def _process_sensor_data(self, data: bytes):
        """Process sensor data from ESP32"""
        
        try:
            # Parse sensor data (format depends on ESP32 implementation)
            # Example: [SENSOR_ID][VALUE_HIGH][VALUE_LOW]
            if len(data) >= 3:
                sensor_id = data[0]
                value = struct.unpack('>H', data[1:3])[0]  # 16-bit value
                
                # Map sensor ID to type
                sensor_types = {
                    0: ('distance', 'cm'),
                    1: ('temperature', '°C'),
                    2: ('humidity', '%'),
                    3: ('light', 'lux'),
                    4: ('battery', 'V')
                }
                
                if sensor_id in sensor_types:
                    sensor_type, unit = sensor_types[sensor_id]
                    
                    # Store data
                    sensor_data = SensorData(
                        sensor_type=sensor_type,
                        value=value / 100.0,  # Convert to decimal
                        unit=unit,
                        timestamp=time.time()
                    )
                    
                    self.sensor_data[sensor_type] = sensor_data
                    
                    # Call callback if registered
                    if sensor_type in self.sensor_callbacks:
                        self.sensor_callbacks[sensor_type](sensor_data)
                    
                    # Emit event to brain
                    if self.brain:
                        from core.robot_brain import RobotEvent
                        self.brain.emit_event(RobotEvent(
                            type='sensor_reading',
                            source='esp32',
                            data={
                                'sensor': sensor_type,
                                'value': sensor_data.value,
                                'unit': unit
                            }
                        ))
                        
        except Exception as e:
            self.logger.error(f"Sensor data processing error: {e}")
    
    def _process_status(self, data: bytes):
        """Process status update from ESP32"""
        
        try:
            # Parse status (implementation specific)
            status = {
                'battery': data[0] if len(data) > 0 else 0,
                'motors_enabled': bool(data[1]) if len(data) > 1 else False,
                'error_flags': data[2] if len(data) > 2 else 0
            }
            
            self.logger.debug(f"ESP32 status: {status}")
            
            # Check for critical conditions
            if status['battery'] < 20:
                self.logger.warning(f"Low battery: {status['battery']}%")
                
                if self.brain:
                    from core.robot_brain import RobotEvent
                    self.brain.emit_event(RobotEvent(
                        type='battery_low',
                        source='esp32',
                        data={'level': status['battery']},
                        priority=2
                    ))
                    
        except Exception as e:
            self.logger.error(f"Status processing error: {e}")
    
    def _ping(self) -> bool:
        """
        Ping ESP32 to check connection
        
        Returns:
            bool: True if response received
        """
        
        try:
            # Send ping
            packet = self._create_packet(ESP32Commands.PING, b'')
            self.serial.write(packet)
            
            # Wait for response
            time.sleep(0.1)
            
            if self.serial.in_waiting > 0:
                response = self.serial.read(self.serial.in_waiting)
                return response[0] == 0xFF  # Check for valid response
                
            return False
            
        except:
            return False
    
    def _get_status(self):
        """Request status from ESP32"""
        
        self.command_queue.put({
            'cmd': ESP32Commands.STATUS,
            'data': b''
        })
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to ESP32"""
        
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts % 10 == 1:  # Log every 10 attempts
            self.logger.info(f"Attempting reconnection ({self.reconnect_attempts})")
        
        if self.connect():
            self.logger.info("Reconnected successfully")
            self.reconnect_attempts = 0
    
    # Motor Control Methods
    
    def move_forward(self, speed: int = 50, duration: Optional[float] = None):
        """
        Move robot forward
        
        Args:
            speed: Speed percentage (0-100)
            duration: Duration in seconds (None for continuous)
        """
        
        self.motor_state['left'] = speed
        self.motor_state['right'] = speed
        
        if not self.connected:
            self.logger.info(f"[SIMULATION] Moving forward at {speed}% speed")
            return  # Don't queue commands in simulation mode
        
        data = struct.pack('B', min(100, max(0, speed)))
        self.command_queue.put({
            'cmd': ESP32Commands.MOVE_FORWARD,
            'data': data
        })
        
        if duration:
            threading.Timer(duration, self.stop_motors).start()
    
    def move_backward(self, speed: int = 50, duration: Optional[float] = None):
        """Move robot backward"""
        
        self.motor_state['left'] = -speed
        self.motor_state['right'] = -speed
        
        if not self.connected:
            self.logger.info(f"[SIMULATION] Moving backward at {speed}% speed")
            return  # Don't queue commands in simulation mode
        
        data = struct.pack('B', min(100, max(0, speed)))
        self.command_queue.put({
            'cmd': ESP32Commands.MOVE_BACKWARD,
            'data': data
        })
        
        if duration:
            threading.Timer(duration, self.stop_motors).start()
    
    def turn_left(self, speed: int = 50, duration: Optional[float] = None):
        """Turn robot left"""
        
        self.motor_state['left'] = -speed
        self.motor_state['right'] = speed
        
        if not self.connected:
            self.logger.info(f"[SIMULATION] Turning left at {speed}% speed")
            return  # Don't queue commands in simulation mode
        
        data = struct.pack('B', min(100, max(0, speed)))
        self.command_queue.put({
            'cmd': ESP32Commands.TURN_LEFT,
            'data': data
        })
        
        if duration:
            threading.Timer(duration, self.stop_motors).start()
    
    def turn_right(self, speed: int = 50, duration: Optional[float] = None):
        """Turn robot right"""
        
        self.motor_state['left'] = speed
        self.motor_state['right'] = -speed
        
        if not self.connected:
            self.logger.info(f"[SIMULATION] Turning right at {speed}% speed")
            return  # Don't queue commands in simulation mode
        
        data = struct.pack('B', min(100, max(0, speed)))
        self.command_queue.put({
            'cmd': ESP32Commands.TURN_RIGHT,
            'data': data
        })
        
        if duration:
            threading.Timer(duration, self.stop_motors).start()
    
    def stop_motors(self):
        """Stop all motors"""
        
        self.motor_state['left'] = 0
        self.motor_state['right'] = 0
        
        if not self.connected:
            self.logger.info("[SIMULATION] Motors stopped")
            return  # Don't queue commands in simulation mode
        
        self.command_queue.put({
            'cmd': ESP32Commands.STOP,
            'data': b''
        })
    
    def stop_all_motors(self):
        """Emergency stop all motors"""
        
        # Send multiple stop commands for redundancy
        for _ in range(3):
            self.stop_motors()
            time.sleep(0.01)
    
    # Servo Control Methods
    
    def set_servo(self, servo_id: int, angle: int):
        """
        Set servo position
        
        Args:
            servo_id: Servo ID (0-5 for hand)
            angle: Angle in degrees (0-180)
        """
        
        if servo_id >= len(hardware_config.servo_pins):
            self.logger.error(f"Invalid servo ID: {servo_id}")
            return
        
        angle = min(180, max(0, angle))
        self.servo_positions[servo_id] = angle
        
        if not self.connected:
            self.logger.info(f"[SIMULATION] Setting servo {servo_id} to {angle}°")
            return  # Don't queue commands in simulation mode
        
        data = struct.pack('BB', servo_id, angle)
        self.command_queue.put({
            'cmd': ESP32Commands.SERVO_SET,
            'data': data
        })
    
    def move_hand(self, finger_positions: List[int]):
        """
        Move robotic hand
        
        Args:
            finger_positions: List of angles for each finger
        """
        
        for i, position in enumerate(finger_positions[:6]):
            self.set_servo(i, position)
            time.sleep(0.05)  # Small delay between servos
    
    # Sensor Methods
    
    def read_sensor(self, sensor_type: str) -> Optional[float]:
        """
        Read sensor value
        
        Args:
            sensor_type: Type of sensor
            
        Returns:
            Sensor value or None
        """
        
        if sensor_type in self.sensor_data:
            return self.sensor_data[sensor_type].value
        
        return None
    
    def register_sensor_callback(self, sensor_type: str, callback: Callable):
        """
        Register callback for sensor data
        
        Args:
            sensor_type: Type of sensor
            callback: Function to call with sensor data
        """
        
        self.sensor_callbacks[sensor_type] = callback
    
    # GPIO Methods
    
    def set_gpio(self, pin: int, value: bool):
        """
        Set GPIO pin state
        
        Args:
            pin: GPIO pin number
            value: High (True) or Low (False)
        """
        
        data = struct.pack('BB', pin, int(value))
        self.command_queue.put({
            'cmd': ESP32Commands.GPIO_SET,
            'data': data
        })
    
    def set_led(self, color: str):
        """
        Set status LED color
        
        Args:
            color: Color name (red, green, blue, off)
        """
        
        led_pins = {
            'red': 25,
            'green': 26,
            'blue': 27
        }
        
        # Turn off all LEDs
        for pin in led_pins.values():
            self.set_gpio(pin, False)
        
        # Turn on requested color
        if color in led_pins:
            self.set_gpio(led_pins[color], True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics"""
        
        return {
            'connected': self.connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'commands_sent': self.commands_sent,
            'commands_failed': self.commands_failed,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'motor_state': self.motor_state,
            'sensors': list(self.sensor_data.keys()),
            'reconnect_attempts': self.reconnect_attempts
        }
    
    def shutdown(self):
        """Clean shutdown"""
        
        self.stop()


if __name__ == "__main__":
    """Test ESP32 controller"""
    
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create controller
    controller = ESP32Controller()
    
    print("\nESP32 Controller Test")
    print("=" * 50)
    print("Commands:")
    print("  c        - Connect")
    print("  f        - Move forward")
    print("  b        - Move backward")
    print("  l        - Turn left")
    print("  r        - Turn right")
    print("  s        - Stop")
    print("  h <pos>  - Move hand servos")
    print("  t        - Show statistics")
    print("  q        - Quit")
    print()
    
    # Start controller
    if controller.connect():
        controller.start()
        print("Controller connected and started")
    else:
        print("Failed to connect")
    
    try:
        while True:
            cmd = input("Command: ").strip().lower()
            
            if cmd == 'c':
                if controller.connect():
                    controller.start()
                    print("Connected")
                else:
                    print("Connection failed")
            
            elif cmd == 'f':
                controller.move_forward(50, duration=2.0)
                print("Moving forward...")
            
            elif cmd == 'b':
                controller.move_backward(50, duration=2.0)
                print("Moving backward...")
            
            elif cmd == 'l':
                controller.turn_left(50, duration=1.0)
                print("Turning left...")
            
            elif cmd == 'r':
                controller.turn_right(50, duration=1.0)
                print("Turning right...")
            
            elif cmd == 's':
                controller.stop_motors()
                print("Stopped")
            
            elif cmd.startswith('h '):
                try:
                    positions = [int(x) for x in cmd[2:].split()]
                    controller.move_hand(positions)
                    print(f"Hand positions: {positions}")
                except:
                    print("Usage: h <pos1> <pos2> ...")
            
            elif cmd == 't':
                stats = controller.get_statistics()
                print("\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
            
            elif cmd == 'q':
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        controller.stop()
        print("\nController stopped")

    def connect(self) -> bool:
        """
        Connect to ESP32 via serial
        
        Returns:
            bool: True if connected successfully
        """
        
        try:
            # Find ESP32 port if not specified
            if not self.port or self.port == 'auto':
                self.port = self._find_esp32_port()
            
            if not self.port:
                self.logger.error("ESP32 port not found")
                return False
            
            # Open serial connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            # Clear buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Wait for ESP32 to initialize
            time.sleep(2.0)
            
            # Send ping to verify connection
            if self._ping():
                self.connected = True
                self.reconnect_attempts = 0
                self.logger.info(f"Connected to ESP32 on {self.port}")
                
                # Get ESP32 status
                self._get_status()
                
                return True
            else:
                self.logger.error("ESP32 not responding to ping")
                self.serial.close()
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to ESP32: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        
        if self.serial and self.serial.is_open:
            self.serial.close()
        
        self.connected = False
        self.logger.info("Disconnected from ESP32")
    
    def _find_esp32_port(self) -> Optional[str]:
        """
        Auto-detect ESP32 port
        
        Returns:
            Port name or None
        """
        
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Check for common ESP32 identifiers
            if 'CP210' in port.description or \
               'CH340' in port.description or \
               'ESP32' in port.description or \
               'USB Serial' in port.description:
                self.logger.info(f"Found potential ESP32 on {port.device}")
                return port.device
        
        # Try common port names
        common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', 'COM3', 'COM4']
        
        for port_name in common_ports:
            try:
                test_serial = serial.Serial(port_name, 9600, timeout=0.5)
                test_serial.close()
                return port_name
            except:
                continue
        
        return None
    
    def start(self):
        """Start ESP32 controller"""
        
        if not self.connected:
            if not self.connect():
                self.logger.error("Failed to connect to ESP32")
                return False
        
        return True