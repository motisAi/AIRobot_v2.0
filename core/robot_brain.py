"""
Robot Brain - Central Control System
=====================================
Main state machine and decision engine for the robot.
Coordinates all modules and manages robot behavior.

"""

import asyncio
import threading
import queue
import time
import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

# State machine library for robust state management
from transitions import Machine

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config, system_config, behavior_config


class RobotState(Enum):
    """Enumeration of all possible robot states"""
    
    # Basic States
    INITIALIZING = auto()  # System startup
    IDLE = auto()          # Waiting for interaction
    LISTENING = auto()     # Actively listening for commands
    PROCESSING = auto()    # Processing input/commands
    RESPONDING = auto()    # Providing response (speech/action)
    
    # Action States
    EXECUTING = auto()     # Executing a task
    MOVING = auto()        # Physical movement
    OBSERVING = auto()     # Analyzing environment
    LEARNING = auto()      # Learning new information
    
    # Special States
    PATROLLING = auto()    # Autonomous patrol mode
    FOLLOWING = auto()     # Following a person
    CHARGING = auto()      # Battery charging
    
    # Alert States
    ALERT = auto()         # Detected something important
    WARNING = auto()       # Warning state (low battery, etc.)
    
    # System States
    ERROR = auto()         # Error state
    EMERGENCY_STOP = auto() # Emergency halt
    MAINTENANCE = auto()   # Maintenance mode
    SHUTDOWN = auto()      # System shutdown


@dataclass
class RobotEvent:
    """Event structure for inter-module communication"""
    
    type: str                    # Event type (e.g., 'wake_word_detected')
    source: str                  # Source module
    data: Dict[str, Any] = field(default_factory=dict)  # Event data
    priority: int = 5            # Priority (1=highest, 10=lowest)
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    callback: Optional[Callable] = None


@dataclass
class Memory:
    """Memory storage structure"""
    
    type: str                    # Type of memory (short_term, long_term, working)
    content: Any                 # Memory content
    context: Dict[str, Any]      # Context information
    importance: float            # Importance score (0-1)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    expiry: Optional[datetime] = None


class RobotBrain:
    """
    Central control system for the robot.
    Manages state transitions, decision making, and module coordination.
    """
    
    def __init__(self):
        """Initialize the robot brain"""
        
        # Logging setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Robot Brain...")
        
        # State machine setup
        self.state = RobotState.INITIALIZING
        self._setup_state_machine()
        
        # Event system
        self.event_queue = queue.PriorityQueue()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_thread = None
        self.running = False
        
        # Module references (will be set by main.py)
        self.modules = {}
        
        # Memory systems
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        self.working_memory: Dict[str, Any] = {}
        self.context_stack: List[Dict] = []
        
        # Behavior control
        self.current_task = None
        self.task_queue = queue.Queue()
        self.behavior_mode = behavior_config.personality_type
        
        # System monitoring
        self.health_status = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'battery_level': 100,
            'temperature': 0,
            'errors': []
        }
        
        # User management
        self.current_user = None
        self.authenticated = False
        self.master_mode = False
        
        # Timers and counters
        self.idle_timer = 0
        self.interaction_count = 0
        self.uptime_start = time.time()
        
    def _setup_state_machine(self):
        """Configure the state machine with transitions"""
        
        # Define state transitions
        transitions = [
            # Initialization
            {'trigger': 'startup_complete', 'source': RobotState.INITIALIZING, 
             'dest': RobotState.IDLE, 'after': 'on_enter_idle'},
            
            # Wake word detection
            {'trigger': 'wake_word_heard', 'source': RobotState.IDLE, 
             'dest': RobotState.LISTENING, 'after': 'on_start_listening'},
            
            # Speech processing
            {'trigger': 'speech_received', 'source': RobotState.LISTENING, 
             'dest': RobotState.PROCESSING, 'after': 'on_process_speech'},
            
            # Response generation
            {'trigger': 'generate_response', 'source': RobotState.PROCESSING, 
             'dest': RobotState.RESPONDING, 'after': 'on_generate_response'},
            
            # Task execution
            {'trigger': 'execute_task', 'source': [RobotState.PROCESSING, RobotState.IDLE], 
             'dest': RobotState.EXECUTING, 'after': 'on_execute_task'},
            
            # Movement
            {'trigger': 'start_moving', 'source': [RobotState.IDLE, RobotState.EXECUTING], 
             'dest': RobotState.MOVING, 'after': 'on_start_moving'},
            
            # Return to idle
            {'trigger': 'return_idle', 'source': '*', 
             'dest': RobotState.IDLE, 'after': 'on_enter_idle'},
            
            # Emergency stop
            {'trigger': 'emergency_stop', 'source': '*', 
             'dest': RobotState.EMERGENCY_STOP, 'after': 'on_emergency_stop'},
            
            # Error handling
            {'trigger': 'error_occurred', 'source': '*', 
             'dest': RobotState.ERROR, 'after': 'on_error'},
            
            # Patrol mode
            {'trigger': 'start_patrol', 'source': RobotState.IDLE, 
             'dest': RobotState.PATROLLING, 'after': 'on_start_patrol'},
            
            # Learning mode
            {'trigger': 'start_learning', 'source': [RobotState.IDLE, RobotState.OBSERVING], 
             'dest': RobotState.LEARNING, 'after': 'on_start_learning'},
            
            # Observation mode
            {'trigger': 'start_observing', 'source': [RobotState.IDLE, RobotState.PATROLLING], 
             'dest': RobotState.OBSERVING, 'after': 'on_start_observing'},
            
            # Alert state
            {'trigger': 'raise_alert', 'source': '*', 
             'dest': RobotState.ALERT, 'after': 'on_raise_alert'},
            
            # Shutdown
            {'trigger': 'shutdown', 'source': '*', 
             'dest': RobotState.SHUTDOWN, 'after': 'on_shutdown'}
        ]
        
        # Create state machine
        self.machine = Machine(
            model=self,
            states=RobotState,
            transitions=transitions,
            initial=RobotState.INITIALIZING,
            auto_transitions=False,
            ignore_invalid_triggers=True
        )
        
    # State Entry Callbacks
    
    def on_enter_idle(self):
        """Called when entering IDLE state"""
        self.logger.info("Entering IDLE state")
        self.idle_timer = time.time()
        self.current_task = None
        
        # Check for queued tasks
        if not self.task_queue.empty():
            task = self.task_queue.get()
            self.execute_task(task)
    
    def on_start_listening(self):
        """Called when starting to listen"""
        self.logger.info("Starting to listen for commands")
        
        # Notify audio module to start recording
        if 'audio' in self.modules:
            self.modules['audio'].start_recording()
        
        # Visual feedback
        self.set_led_color('blue')
        
        # Set timeout for listening
        asyncio.create_task(self._listening_timeout())
    
    def on_process_speech(self, text: str):
        """Process received speech"""
        self.logger.info(f"Processing speech: {text}")
        
        # Add to working memory
        self.working_memory['last_input'] = text
        self.working_memory['input_time'] = time.time()
        
        # Analyze intent
        intent = self._analyze_intent(text)
        
        # Make decision
        action = self._make_decision(intent)
        
        # Execute action
        if action:
            self.execute_task(action)
        else:
            self.generate_response()
    
    def on_generate_response(self):
        """Generate and deliver response"""
        self.logger.info("Generating response")
        
        response = self._generate_response_text()
        
        # Speak response
        if 'audio' in self.modules and response:
            self.modules['audio'].speak(response)
        
        # Return to idle after response
        asyncio.create_task(self._response_complete())
    
    def on_execute_task(self, task: Dict[str, Any]):
        """Execute a specific task"""
        self.logger.info(f"Executing task: {task.get('name', 'unknown')}")
        
        self.current_task = task
        
        # Route task to appropriate module
        task_type = task.get('type')
        
        if task_type == 'movement':
            self.start_moving(task)
        elif task_type == 'observation':
            self.start_observing()
        elif task_type == 'learning':
            self.start_learning(task)
        else:
            # Generic task execution
            asyncio.create_task(self._execute_generic_task(task))
    
    def on_start_moving(self, movement_data: Dict):
        """Start movement action"""
        self.logger.info(f"Starting movement: {movement_data}")
        
        if 'hardware' in self.modules:
            self.modules['hardware'].move(movement_data)
        
        # Monitor movement completion
        asyncio.create_task(self._monitor_movement())
    
    def on_emergency_stop(self):
        """Handle emergency stop"""
        self.logger.critical("EMERGENCY STOP ACTIVATED!")
        
        # Stop all motors immediately
        if 'hardware' in self.modules:
            self.modules['hardware'].stop_all_motors()
        
        # Alert
        self.set_led_color('red')
        if 'audio' in self.modules:
            self.modules['audio'].play_alert_sound()
        
        # Save state for debugging
        self._save_emergency_state()
    
    def on_error(self, error: Exception):
        """Handle error state"""
        self.logger.error(f"Error occurred: {error}")
        
        # Add to health status
        self.health_status['errors'].append({
            'time': time.time(),
            'error': str(error),
            'state': self.state.name
        })
        
        # Attempt recovery
        asyncio.create_task(self._attempt_recovery(error))
    
    def on_start_patrol(self):
        """Start patrol mode"""
        self.logger.info("Starting patrol mode")
        
        # Initialize patrol parameters
        self.working_memory['patrol_start'] = time.time()
        self.working_memory['patrol_waypoints'] = self._generate_patrol_route()
        
        # Start patrol loop
        asyncio.create_task(self._patrol_loop())
    
    def on_start_learning(self, learning_data: Dict):
        """Enter learning mode"""
        self.logger.info(f"Starting learning: {learning_data.get('type', 'general')}")
        
        # Set learning context
        self.working_memory['learning_context'] = learning_data
        
        # Start learning process
        asyncio.create_task(self._learning_process(learning_data))
    
    def on_start_observing(self):
        """Start observation mode"""
        self.logger.info("Starting observation mode")
        
        # Enable all sensors
        if 'vision' in self.modules:
            self.modules['vision'].enable_continuous_capture()
        
        # Start analysis
        asyncio.create_task(self._observation_loop())
    
    def on_raise_alert(self, alert_data: Dict):
        """Handle alert state"""
        self.logger.warning(f"Alert raised: {alert_data}")
        
        # Notify user
        if 'audio' in self.modules:
            alert_message = alert_data.get('message', 'Alert detected')
            self.modules['audio'].speak(alert_message)
        
        # Log alert
        self._log_alert(alert_data)
        
        # Send notification if configured
        if 'communication' in self.modules:
            self.modules['communication'].send_alert(alert_data)
    
    def on_shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Robot Brain")
        
        # Save current state
        self._save_state()
        
        # Stop all modules
        for module in self.modules.values():
            if hasattr(module, 'shutdown'):
                module.shutdown()
        
        # Clean up
        self.running = False
    
    # Event System
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """
        Register an event handler
        
        Args:
            event_type: Type of event to handle
            handler: Callback function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for {event_type}")
    
    def emit_event(self, event: RobotEvent):
        """
        Emit an event to the event queue
        
        Args:
            event: Event to emit
        """
        # Priority queue uses tuple (priority, event)
        self.event_queue.put((event.priority, event))
    
    def _process_events(self):
        """Process events from the event queue"""
        while self.running:
            try:
                # Get event with timeout
                priority, event = self.event_queue.get(timeout=0.1)
                
                # Process event
                self._handle_event(event)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing event: {e}")
    
    def _handle_event(self, event: RobotEvent):
        """
        Handle a single event
        
        Args:
            event: Event to handle
        """
        self.logger.debug(f"Handling event: {event.type} from {event.source}")
        
        # Call registered handlers
        if event.type in self.event_handlers:
            for handler in self.event_handlers[event.type]:
                try:
                    result = handler(event)
                    
                    # Handle callback if needed
                    if event.requires_response and event.callback:
                        event.callback(result)
                        
                except Exception as e:
                    self.logger.error(f"Handler error for {event.type}: {e}")
        
        # Built-in event handling
        self._handle_builtin_event(event)
    
    def _handle_builtin_event(self, event: RobotEvent):
        """Handle built-in system events"""
        
        event_map = {
            'wake_word_detected': lambda: self.wake_word_heard(),
            'speech_recognized': lambda: self.speech_received(event.data.get('text')),
            'face_detected': lambda: self._handle_face_detection(event.data),
            'object_detected': lambda: self._handle_object_detection(event.data),
            'movement_complete': lambda: self.return_idle(),
            'battery_low': lambda: self._handle_low_battery(),
            'emergency_button': lambda: self.emergency_stop(),
            'user_authenticated': lambda: self._handle_authentication(event.data)
        }
        
        if event.type in event_map:
            event_map[event.type]()
    
    # Decision Making
    
    def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """
        Analyze user intent from speech text
        
        Args:
            text: Speech text to analyze
            
        Returns:
            Intent dictionary with type and entities
        """
        intent = {
            'raw_text': text,
            'type': None,
            'entities': {},
            'confidence': 0.0
        }
        
        # Convert to lowercase for analysis
        text_lower = text.lower()
        
        # Movement commands
        if any(word in text_lower for word in ['move', 'go', 'come', 'follow', 'stop']):
            intent['type'] = 'movement'
            
            if 'forward' in text_lower:
                intent['entities']['direction'] = 'forward'
            elif 'back' in text_lower:
                intent['entities']['direction'] = 'backward'
            elif 'left' in text_lower:
                intent['entities']['direction'] = 'left'
            elif 'right' in text_lower:
                intent['entities']['direction'] = 'right'
            elif 'follow' in text_lower:
                intent['entities']['action'] = 'follow'
            elif 'stop' in text_lower:
                intent['entities']['action'] = 'stop'
        
        # Information queries
        elif any(word in text_lower for word in ['what', 'who', 'where', 'when', 'how']):
            intent['type'] = 'query'
            
            if 'time' in text_lower:
                intent['entities']['query_type'] = 'time'
            elif 'weather' in text_lower:
                intent['entities']['query_type'] = 'weather'
            elif 'name' in text_lower:
                intent['entities']['query_type'] = 'name'
        
        # Object interaction
        elif any(word in text_lower for word in ['find', 'look', 'search', 'get', 'grab']):
            intent['type'] = 'object_interaction'
            intent['entities']['action'] = text_lower.split()[0]
        
        # Learning commands
        elif any(word in text_lower for word in ['learn', 'remember', 'teach']):
            intent['type'] = 'learning'
        
        # System commands
        elif any(word in text_lower for word in ['shutdown', 'restart', 'sleep', 'wake']):
            intent['type'] = 'system'
            intent['entities']['command'] = text_lower.split()[0]
        
        # Social interaction
        elif any(word in text_lower for word in ['hello', 'hi', 'hey', 'goodbye', 'bye']):
            intent['type'] = 'social'
            intent['entities']['greeting_type'] = 'greeting' if 'hello' in text_lower else 'farewell'
        
        # Default to conversation
        else:
            intent['type'] = 'conversation'
        
        # Set confidence based on keyword matching (simplified)
        intent['confidence'] = 0.8 if intent['type'] else 0.3
        
        self.logger.debug(f"Intent analysis: {intent}")
        return intent
    
    def _make_decision(self, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make decision based on intent
        
        Args:
            intent: Analyzed intent
            
        Returns:
            Action to execute or None
        """
        action = None
        
        # Check permissions first
        if not self._check_permissions(intent['type']):
            return {
                'type': 'response',
                'message': 'Sorry, you need to be authenticated for that action.'
            }
        
        # Make decision based on intent type
        if intent['type'] == 'movement':
            action = {
                'type': 'movement',
                'name': 'move_robot',
                'parameters': intent['entities']
            }
        
        elif intent['type'] == 'query':
            action = {
                'type': 'query',
                'name': 'answer_query',
                'query_type': intent['entities'].get('query_type')
            }
        
        elif intent['type'] == 'object_interaction':
            action = {
                'type': 'vision',
                'name': 'find_object',
                'parameters': intent['entities']
            }
        
        elif intent['type'] == 'learning':
            action = {
                'type': 'learning',
                'name': 'learn_information',
                'data': {'source': 'user_command'}
            }
        
        elif intent['type'] == 'system':
            command = intent['entities'].get('command')
            if command == 'shutdown' and self.master_mode:
                action = {
                    'type': 'system',
                    'name': 'shutdown'
                }
        
        elif intent['type'] == 'social':
            action = {
                'type': 'social',
                'name': 'social_response',
                'greeting_type': intent['entities'].get('greeting_type')
            }
        
        return action
    
    def _check_permissions(self, action_type: str) -> bool:
        """
        Check if current user has permission for action
        
        Args:
            action_type: Type of action to check
            
        Returns:
            bool: True if permitted
        """
        # Public actions (no auth required)
        public_actions = ['social', 'query', 'conversation']
        
        if action_type in public_actions:
            return True
        
        # Check authentication
        if not self.authenticated:
            return False
        
        # Master-only actions
        master_only = ['system', 'learning']
        
        if action_type in master_only and not self.master_mode:
            return False
        
        return True
    
    def _generate_response_text(self) -> str:
        """
        Generate appropriate response text
        
        Returns:
            Response text to speak
        """
        # Check context
        last_input = self.working_memory.get('last_input', '')
        current_task = self.current_task
        
        # Generate based on context
        if current_task and current_task.get('type') == 'social':
            greeting_type = current_task.get('greeting_type', 'greeting')
            if greeting_type == 'greeting':
                return behavior_config.greeting_message.format(name=behavior_config.robot_name)
            else:
                return behavior_config.goodbye_message
        
        elif current_task and current_task.get('type') == 'query':
            query_type = current_task.get('query_type')
            if query_type == 'time':
                return f"The current time is {datetime.now().strftime('%H:%M')}"
            elif query_type == 'name':
                return f"My name is {behavior_config.robot_name}"
        
        # Default response
        return "I understand. How can I help you?"
    
    # Memory Management
    
    def add_memory(self, content: Any, memory_type: str = 'short_term', 
                   importance: float = 0.5, context: Dict = None):
        """
        Add a memory to the system
        
        Args:
            content: Content to remember
            memory_type: Type of memory (short_term/long_term)
            importance: Importance score (0-1)
            context: Additional context
        """
        memory = Memory(
            type=memory_type,
            content=content,
            context=context or {},
            importance=importance
        )
        
        if memory_type == 'short_term':
            self.short_term_memory.append(memory)
            
            # Limit short-term memory size
            if len(self.short_term_memory) > 100:
                # Move important memories to long-term
                important = [m for m in self.short_term_memory if m.importance > 0.7]
                for mem in important:
                    mem.type = 'long_term'
                    self.long_term_memory.append(mem)
                
                # Keep only recent memories
                self.short_term_memory = self.short_term_memory[-50:]
        
        else:
            self.long_term_memory.append(memory)
            
            # Limit long-term memory
            if len(self.long_term_memory) > behavior_config.max_memories:
                # Remove least important/oldest
                self.long_term_memory.sort(key=lambda x: (x.importance, x.created_at))
                self.long_term_memory = self.long_term_memory[-behavior_config.max_memories:]
    
    def recall_memory(self, query: str, memory_type: str = 'all') -> List[Memory]:
        """
        Recall memories based on query
        
        Args:
            query: Search query
            memory_type: Type to search (short_term/long_term/all)
            
        Returns:
            List of relevant memories
        """
        memories = []
        
        if memory_type in ['short_term', 'all']:
            memories.extend(self.short_term_memory)
        
        if memory_type in ['long_term', 'all']:
            memories.extend(self.long_term_memory)
        
        # Simple relevance scoring (can be improved with embeddings)
        relevant = []
        for memory in memories:
            if query.lower() in str(memory.content).lower():
                memory.accessed_count += 1
                memory.last_accessed = datetime.now()
                relevant.append(memory)
        
        return sorted(relevant, key=lambda x: x.importance, reverse=True)
    
    # Async Helper Methods
    
    async def _listening_timeout(self):
        """Timeout for listening state"""
        await asyncio.sleep(5.0)  # 5 second timeout
        
        if self.state == RobotState.LISTENING:
            self.logger.info("Listening timeout, returning to idle")
            self.return_idle()
    
    async def _response_complete(self):
        """Wait for response completion"""
        await asyncio.sleep(0.5)  # Brief pause
        self.return_idle()
    
    async def _execute_generic_task(self, task: Dict):
        """Execute a generic task"""
        try:
            # Simulate task execution
            await asyncio.sleep(2.0)
            
            self.logger.info(f"Task {task.get('name')} completed")
            self.return_idle()
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self.error_occurred(e)
    
    async def _monitor_movement(self):
        """Monitor movement completion"""
        # Wait for movement complete event
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.state != RobotState.MOVING:
                break
            await asyncio.sleep(0.1)
        
        if self.state == RobotState.MOVING:
            self.logger.warning("Movement timeout")
            self.return_idle()
    
    async def _attempt_recovery(self, error: Exception):
        """Attempt to recover from error"""
        self.logger.info(f"Attempting recovery from {error}")
        
        # Wait a moment
        await asyncio.sleep(2.0)
        
        # Try to return to idle
        try:
            self.return_idle()
            self.logger.info("Recovery successful")
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            # Last resort - emergency stop
            self.emergency_stop()
    
    async def _patrol_loop(self):
        """Main patrol loop"""
        waypoints = self.working_memory.get('patrol_waypoints', [])
        waypoint_index = 0
        
        while self.state == RobotState.PATROLLING:
            if waypoints:
                # Move to next waypoint
                waypoint = waypoints[waypoint_index]
                
                # Navigate to waypoint
                self.emit_event(RobotEvent(
                    type='navigate_to',
                    source='brain',
                    data={'destination': waypoint}
                ))
                
                # Wait for arrival or timeout
                await asyncio.sleep(10.0)
                
                # Next waypoint
                waypoint_index = (waypoint_index + 1) % len(waypoints)
            
            # Check for interesting observations
            if random.random() < 0.1:  # 10% chance
                self.start_observing()
                await asyncio.sleep(5.0)
            
            await asyncio.sleep(1.0)
    
    async def _learning_process(self, learning_data: Dict):
        """Process learning task"""
        self.logger.info("Starting learning process")
        
        learning_type = learning_data.get('type', 'general')
        
        if learning_type == 'face':
            # Learn new face
            if 'vision' in self.modules:
                result = await self.modules['vision'].learn_face(learning_data)
                self.add_memory(result, 'long_term', importance=0.9)
        
        elif learning_type == 'object':
            # Learn new object
            if 'vision' in self.modules:
                result = await self.modules['vision'].learn_object(learning_data)
                self.add_memory(result, 'long_term', importance=0.7)
        
        elif learning_type == 'voice':
            # Learn voice profile
            if 'audio' in self.modules:
                result = await self.modules['audio'].learn_voice(learning_data)
                self.add_memory(result, 'long_term', importance=0.8)
        
        # Return to previous state
        await asyncio.sleep(1.0)
        self.return_idle()
    
    async def _observation_loop(self):
        """Continuous observation loop"""
        observation_duration = 10.0  # Observe for 10 seconds
        start_time = time.time()
        
        while time.time() - start_time < observation_duration:
            if self.state != RobotState.OBSERVING:
                break
            
            # Get vision data
            if 'vision' in self.modules:
                observations = self.modules['vision'].get_current_observations()
                
                # Process observations
                for obs in observations:
                    if obs['type'] == 'person':
                        self._handle_person_observation(obs)
                    elif obs['type'] == 'object':
                        self._handle_object_observation(obs)
                    elif obs['type'] == 'anomaly':
                        self.raise_alert({'message': f"Anomaly detected: {obs['description']}"})
            
            await asyncio.sleep(0.5)
        
        self.return_idle()
    
    # Event Handlers
    
    def _handle_face_detection(self, data: Dict):
        """Handle face detection event"""
        face_id = data.get('face_id')
        confidence = data.get('confidence', 0)
        
        if face_id == 'unknown' and confidence > 0.7:
            # Unknown person
            if behavior_config.learn_new_faces:
                response = behavior_config.unknown_person_response
                self.emit_event(RobotEvent(
                    type='speak',
                    source='brain',
                    data={'text': response}
                ))
        
        elif face_id == security_config.master_user_id:
            # Master detected
            self.master_mode = True
            self.authenticated = True
            self.current_user = face_id
            
            self.emit_event(RobotEvent(
                type='speak',
                source='brain',
                data={'text': f"Welcome back, Master!"}
            ))
        
        else:
            # Known person
            self.authenticated = True
            self.current_user = face_id
            
            # Recall memories about this person
            memories = self.recall_memory(face_id)
            if memories:
                last_interaction = memories[0].context.get('last_seen')
                # Personalized greeting based on memory
    
    def _handle_object_detection(self, data: Dict):
        """Handle object detection event"""
        objects = data.get('objects', [])
        
        for obj in objects:
            # Add to working memory
            self.working_memory[f"object_{obj['class']}"] = {
                'location': obj.get('location'),
                'confidence': obj.get('confidence'),
                'time': time.time()
            }
            
            # Check if this is what we're looking for
            if self.current_task and self.current_task.get('type') == 'find_object':
                target = self.current_task.get('target')
                if target and target.lower() in obj['class'].lower():
                    # Found the object!
                    self.emit_event(RobotEvent(
                        type='object_found',
                        source='brain',
                        data={'object': obj}
                    ))
    
    def _handle_person_observation(self, observation: Dict):
        """Handle person observation"""
        person_id = observation.get('person_id')
        activity = observation.get('activity')
        
        # Add to memory
        self.add_memory(
            content={'person': person_id, 'activity': activity},
            memory_type='short_term',
            importance=0.6,
            context={'location': self.working_memory.get('current_location')}
        )
        
        # Check for concerning activities
        concerning_activities = ['falling', 'distress', 'unusual_behavior']
        if activity in concerning_activities:
            self.raise_alert({
                'type': 'person_concern',
                'person': person_id,
                'activity': activity,
                'message': f"Person appears to be in {activity}"
            })
    
    def _handle_object_observation(self, observation: Dict):
        """Handle object observation"""
        obj_class = observation.get('class')
        location = observation.get('location')
        
        # Update object database
        self.working_memory[f"last_seen_{obj_class}"] = {
            'location': location,
            'time': time.time()
        }
        
        # Learn if new object
        if behavior_config.learn_new_objects:
            known_objects = self.working_memory.get('known_objects', [])
            if obj_class not in known_objects:
                self.start_learning({'type': 'object', 'class': obj_class})
    
    def _handle_low_battery(self):
        """Handle low battery event"""
        self.logger.warning("Low battery detected")
        
        # Set warning state
        self.state = RobotState.WARNING
        
        # Notify user
        self.emit_event(RobotEvent(
            type='speak',
            source='brain',
            data={'text': "My battery is running low. I need to charge soon."}
        ))
        
        # Find charging station if autonomous
        if behavior_config.auto_charge:
            self.emit_event(RobotEvent(
                type='find_charger',
                source='brain',
                priority=2
            ))
    
    def _handle_authentication(self, data: Dict):
        """Handle user authentication"""
        user_id = data.get('user_id')
        auth_method = data.get('method')  # face, voice, manual
        
        self.authenticated = True
        self.current_user = user_id
        
        # Check if master
        if user_id == security_config.master_user_id:
            self.master_mode = True
        
        # Log authentication
        self.add_memory(
            content={'user': user_id, 'method': auth_method},
            memory_type='short_term',
            importance=0.7,
            context={'time': datetime.now()}
        )
    
    # Utility Methods
    
    def set_led_color(self, color: str):
        """Set status LED color"""
        if 'hardware' in self.modules:
            self.modules['hardware'].set_led(color)
    
    def _generate_patrol_route(self) -> List[Dict]:
        """Generate patrol waypoints"""
        # This would normally use the mapped environment
        # For now, return simple waypoints
        return [
            {'x': 0, 'y': 0},
            {'x': 1, 'y': 0},
            {'x': 1, 'y': 1},
            {'x': 0, 'y': 1}
        ]
    
    def _save_state(self):
        """Save current state to disk"""
        state_data = {
            'state': self.state.name,
            'working_memory': self.working_memory,
            'current_user': self.current_user,
            'uptime': time.time() - self.uptime_start,
            'interaction_count': self.interaction_count,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            state_file = PROJECT_ROOT / "data" / "robot_state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.info("State saved successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _save_emergency_state(self):
        """Save state during emergency"""
        emergency_data = {
            'state': self.state.name,
            'working_memory': self.working_memory,
            'health_status': self.health_status,
            'current_task': self.current_task,
            'timestamp': datetime.now().isoformat(),
            'reason': 'emergency_stop'
        }
        
        try:
            emergency_file = PROJECT_ROOT / "data" / f"emergency_{int(time.time())}.json"
            with open(emergency_file, 'w') as f:
                json.dump(emergency_data, f, indent=2, default=str)
            
            self.logger.critical(f"Emergency state saved to {emergency_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to save emergency state: {e}")
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to file"""
        alert_data['timestamp'] = datetime.now().isoformat()
        
        try:
            alert_file = PROJECT_ROOT / "data" / "alerts.json"
            
            # Load existing alerts
            alerts = []
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            
            # Add new alert
            alerts.append(alert_data)
            
            # Keep only last 1000 alerts
            alerts = alerts[-1000:]
            
            # Save back
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
    
    # Main Control Loop
    
    def start(self):
        """Start the robot brain"""
        self.logger.info("Starting Robot Brain")
        
        # Set running flag
        self.running = True
        
        # Start event processing thread
        self.event_thread = threading.Thread(target=self._process_events)
        self.event_thread.daemon = True
        self.event_thread.start()
        
        # Complete initialization
        self.startup_complete()
        
        self.logger.info("Robot Brain started successfully")
    
    def stop(self):
        """Stop the robot brain"""
        self.logger.info("Stopping Robot Brain")
        
        # Trigger shutdown
        self.shutdown()
        
        # Wait for threads to complete
        if self.event_thread:
            self.event_thread.join(timeout=5.0)
        
        self.logger.info("Robot Brain stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current robot status
        
        Returns:
            Status dictionary
        """
        return {
            'state': self.state.name,
            'authenticated': self.authenticated,
            'current_user': self.current_user,
            'master_mode': self.master_mode,
            'current_task': self.current_task,
            'health': self.health_status,
            'uptime': time.time() - self.uptime_start,
            'interaction_count': self.interaction_count,
            'memory_usage': {
                'short_term': len(self.short_term_memory),
                'long_term': len(self.long_term_memory)
            }
        }


# Import for other modules
import random  # For patrol randomness

if __name__ == "__main__":
    """Test the Robot Brain"""
    
    # Create brain instance
    brain = RobotBrain()
    
    # Start brain
    brain.start()
    
    # Simulate some events
    print("Robot Brain Test")
    print("=" * 50)
    
    # Test state transitions
    print(f"Initial state: {brain.state.name}")
    
    # Simulate wake word
    brain.wake_word_heard()
    print(f"After wake word: {brain.state.name}")
    
    # Get status
    status = brain.get_status()
    print(f"\nStatus: {json.dumps(status, indent=2)}")
    
    # Let it run for a bit
    time.sleep(2)
    
    # Stop brain
    brain.stop()
    print("\nBrain stopped")