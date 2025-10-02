"""
Web Server Module
================
Provides HTTP API interface for robot control and monitoring.
Enables remote control and status monitoring via web interface.

Author: AI Robot System
Date: 2024
"""

import logging
import threading
import time
import json
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# Import web framework
try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

class WebServer:
    """
    Web server for robot control and monitoring
    """
    
    def __init__(self, brain, host='0.0.0.0', port=5000):
        """
        Initialize web server
        
        Args:
            brain: Robot brain instance
            host: Host address to bind to
            port: Port number to listen on
        """
        self.brain = brain
        self.logger = logging.getLogger("WebServer")
        self.host = host
        self.port = port
        
        # Server state
        self.running = False
        self.server_thread = None
        
        # Flask app
        self.app = None
        
        if not FLASK_AVAILABLE:
            self.logger.error("Flask not available - web server disabled")
            return
        
        # Initialize Flask app
        self._setup_flask_app()
        
        self.logger.info(f"Web server initialized on {host}:{port}")
    
    def _setup_flask_app(self):
        """Setup Flask application with routes"""
        try:
            self.app = Flask(__name__)
            CORS(self.app)  # Enable CORS for cross-origin requests
            
            # Setup routes
            self._setup_routes()
            
        except Exception as e:
            self.logger.error(f"Error setting up Flask app: {e}")
    
    def _setup_routes(self):
        """Setup web server routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return render_template_string(self._get_dashboard_html())
        
        @self.app.route('/api/status')
        def get_status():
            """Get robot status"""
            try:
                status = self._get_robot_status()
                return jsonify(status)
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/modules')
        def get_modules():
            """Get module information"""
            try:
                modules_info = self._get_modules_info()
                return jsonify(modules_info)
            except Exception as e:
                self.logger.error(f"Error getting modules: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/speak', methods=['POST'])
        def speak():
            """Make robot speak"""
            try:
                data = request.get_json()
                text = data.get('text', '')
                
                if not text:
                    return jsonify({'error': 'No text provided'}), 400
                
                # Send speak command to brain
                if hasattr(self.brain, 'process_event'):
                    self.brain.process_event({
                        'type': 'web_speak_command',
                        'data': {'text': text}
                    })
                
                return jsonify({'success': True, 'message': 'Speaking...'})
                
            except Exception as e:
                self.logger.error(f"Error in speak endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/conversation', methods=['POST'])
        def conversation():
            """Send message to conversation AI"""
            try:
                data = request.get_json()
                message = data.get('message', '')
                
                if not message:
                    return jsonify({'error': 'No message provided'}), 400
                
                # Send to conversation AI if available
                response = "I received your message, but conversation AI is not available."
                
                if hasattr(self.brain, 'modules') and 'conversation_ai' in self.brain.modules:
                    conv_ai = self.brain.modules['conversation_ai']
                    if hasattr(conv_ai, 'process_input'):
                        response = conv_ai.process_input(message)
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Error in conversation endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/frame')
        def get_camera_frame():
            """Get current camera frame info"""
            try:
                frame_info = {
                    'available': False,
                    'timestamp': None,
                    'resolution': None
                }
                
                if hasattr(self.brain, 'current_frame') and self.brain.current_frame is not None:
                    frame_info['available'] = True
                    frame_info['timestamp'] = time.time()
                    frame_info['resolution'] = self.brain.current_frame.shape[:2] if hasattr(self.brain.current_frame, 'shape') else None
                
                return jsonify(frame_info)
                
            except Exception as e:
                self.logger.error(f"Error getting camera frame: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/faces')
        def get_faces():
            """Get detected faces"""
            try:
                faces_info = {
                    'detected_faces': [],
                    'count': 0,
                    'timestamp': time.time()
                }
                
                if hasattr(self.brain, 'modules') and 'face_recognition' in self.brain.modules:
                    face_module = self.brain.modules['face_recognition']
                    if hasattr(face_module, 'get_current_faces'):
                        faces = face_module.get_current_faces()
                        faces_info['detected_faces'] = faces
                        faces_info['count'] = len(faces)
                
                return jsonify(faces_info)
                
            except Exception as e:
                self.logger.error(f"Error getting faces: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/command', methods=['POST'])
        def execute_command():
            """Execute robot command"""
            try:
                data = request.get_json()
                command = data.get('command', '')
                parameters = data.get('parameters', {})
                
                if not command:
                    return jsonify({'error': 'No command provided'}), 400
                
                # Send command to brain
                result = self._execute_robot_command(command, parameters)
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Error executing command: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/logs')
        def get_logs():
            """Get recent log entries"""
            try:
                # Read recent log entries
                logs = self._get_recent_logs()
                return jsonify({
                    'logs': logs,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Error getting logs: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _get_robot_status(self) -> Dict[str, Any]:
        """Get current robot status"""
        try:
            status = {
                'timestamp': time.time(),
                'running': True,
                'uptime': time.time() - getattr(self.brain, 'start_time', time.time()),
                'modules': {},
                'system': {
                    'memory_usage': 0,
                    'cpu_usage': 0
                }
            }
            
            # Get module status
            if hasattr(self.brain, 'modules'):
                for name, module in self.brain.modules.items():
                    status['modules'][name] = {
                        'active': hasattr(module, 'running') and getattr(module, 'running', False),
                        'type': type(module).__name__
                    }
            
            # Get system info
            try:
                import psutil
                status['system']['memory_usage'] = psutil.virtual_memory().percent
                status['system']['cpu_usage'] = psutil.cpu_percent()
            except ImportError:
                pass
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting robot status: {e}")
            return {'error': str(e)}
    
    def _get_modules_info(self) -> Dict[str, Any]:
        """Get detailed module information"""
        try:
            modules_info = {
                'total_modules': 0,
                'active_modules': 0,
                'modules': {}
            }
            
            if hasattr(self.brain, 'modules'):
                modules_info['total_modules'] = len(self.brain.modules)
                
                for name, module in self.brain.modules.items():
                    module_info = {
                        'name': name,
                        'type': type(module).__name__,
                        'active': hasattr(module, 'running') and getattr(module, 'running', False),
                        'description': getattr(module, '__doc__', '').split('\n')[0] if hasattr(module, '__doc__') else ''
                    }
                    
                    if module_info['active']:
                        modules_info['active_modules'] += 1
                    
                    modules_info['modules'][name] = module_info
            
            return modules_info
            
        except Exception as e:
            self.logger.error(f"Error getting modules info: {e}")
            return {'error': str(e)}
    
    def _execute_robot_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute robot command"""
        try:
            result = {'success': False, 'message': 'Command not recognized'}
            
            # Handle different commands
            if command == 'start_module':
                module_name = parameters.get('module')
                if module_name and hasattr(self.brain, 'modules') and module_name in self.brain.modules:
                    module = self.brain.modules[module_name]
                    if hasattr(module, 'start'):
                        module.start()
                        result = {'success': True, 'message': f'Module {module_name} started'}
                    else:
                        result = {'success': False, 'message': f'Module {module_name} cannot be started'}
                else:
                    result = {'success': False, 'message': f'Module {module_name} not found'}
            
            elif command == 'stop_module':
                module_name = parameters.get('module')
                if module_name and hasattr(self.brain, 'modules') and module_name in self.brain.modules:
                    module = self.brain.modules[module_name]
                    if hasattr(module, 'stop'):
                        module.stop()
                        result = {'success': True, 'message': f'Module {module_name} stopped'}
                    else:
                        result = {'success': False, 'message': f'Module {module_name} cannot be stopped'}
                else:
                    result = {'success': False, 'message': f'Module {module_name} not found'}
            
            elif command == 'speak':
                text = parameters.get('text', '')
                if text and hasattr(self.brain, 'process_event'):
                    self.brain.process_event({
                        'type': 'web_speak_command',
                        'data': {'text': text}
                    })
                    result = {'success': True, 'message': 'Speaking command sent'}
                else:
                    result = {'success': False, 'message': 'No text provided or TTS not available'}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {'success': False, 'message': str(e)}
    
    def _get_recent_logs(self, max_lines: int = 50) -> List[str]:
        """Get recent log entries"""
        try:
            logs = []
            
            # Try to read from log file
            log_dir = Path("data/logs")
            if log_dir.exists():
                log_files = sorted(log_dir.glob("robot_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
                
                if log_files:
                    with open(log_files[0], 'r') as f:
                        lines = f.readlines()
                        logs = [line.strip() for line in lines[-max_lines:]]
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Error getting logs: {e}")
            return [f"Error reading logs: {e}"]
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Robot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; border-radius: 4px; margin: 5px 0; }
        .status.online { background: #d4edda; color: #155724; }
        .status.offline { background: #f8d7da; color: #721c24; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        input[type="text"] { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; width: 300px; }
        .logs { background: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Robot Dashboard</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <div id="status-info">Loading...</div>
            <button onclick="refreshStatus()">Refresh Status</button>
        </div>
        
        <div class="card">
            <h2>Voice Control</h2>
            <input type="text" id="speak-text" placeholder="Enter text to speak..." />
            <button onclick="speak()">Speak</button>
        </div>
        
        <div class="card">
            <h2>Conversation</h2>
            <input type="text" id="conversation-input" placeholder="Type your message..." />
            <button onclick="sendMessage()">Send</button>
            <div id="conversation-response"></div>
        </div>
        
        <div class="card">
            <h2>Modules</h2>
            <div id="modules-info">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Recent Logs</h2>
            <div id="logs" class="logs">Loading...</div>
            <button onclick="refreshLogs()">Refresh Logs</button>
        </div>
    </div>
    
    <script>
        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                let html = `
                    <div class="status online">Robot Status: Online</div>
                    <p>Uptime: ${Math.round(data.uptime)} seconds</p>
                    <p>Memory Usage: ${data.system?.memory_usage || 'N/A'}%</p>
                    <p>CPU Usage: ${data.system?.cpu_usage || 'N/A'}%</p>
                `;
                
                document.getElementById('status-info').innerHTML = html;
            } catch (error) {
                document.getElementById('status-info').innerHTML = '<div class="status offline">Error loading status</div>';
            }
        }
        
        async function refreshModules() {
            try {
                const response = await fetch('/api/modules');
                const data = await response.json();
                
                let html = `<p>Total Modules: ${data.total_modules}, Active: ${data.active_modules}</p>`;
                
                for (const [name, module] of Object.entries(data.modules || {})) {
                    const status = module.active ? 'online' : 'offline';
                    html += `<div class="status ${status}">${name}: ${module.active ? 'Active' : 'Inactive'}</div>`;
                }
                
                document.getElementById('modules-info').innerHTML = html;
            } catch (error) {
                document.getElementById('modules-info').innerHTML = 'Error loading modules';
            }
        }
        
        async function speak() {
            const text = document.getElementById('speak-text').value;
            if (!text) return;
            
            try {
                await fetch('/api/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                
                document.getElementById('speak-text').value = '';
                alert('Speaking...');
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function sendMessage() {
            const message = document.getElementById('conversation-input').value;
            if (!message) return;
            
            try {
                const response = await fetch('/api/conversation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                document.getElementById('conversation-input').value = '';
                document.getElementById('conversation-response').innerHTML = 
                    `<p><strong>You:</strong> ${message}</p><p><strong>Robot:</strong> ${data.response}</p>`;
                    
            } catch (error) {
                document.getElementById('conversation-response').innerHTML = 'Error: ' + error.message;
            }
        }
        
        async function refreshLogs() {
            try {
                const response = await fetch('/api/logs');
                const data = await response.json();
                
                const logsHtml = data.logs.map(log => `<div>${log}</div>`).join('');
                document.getElementById('logs').innerHTML = logsHtml;
            } catch (error) {
                document.getElementById('logs').innerHTML = 'Error loading logs';
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            refreshStatus();
            refreshModules();
        }, 30000);
        
        // Initial load
        refreshStatus();
        refreshModules();
        refreshLogs();
        
        // Enter key support
        document.getElementById('speak-text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') speak();
        });
        
        document.getElementById('conversation-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
        """
    
    def start(self):
        """Start web server"""
        if not FLASK_AVAILABLE:
            self.logger.error("Cannot start web server - Flask not available")
            return False
        
        if not self.running:
            self.running = True
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.logger.info(f"Web server started on http://{self.host}:{self.port}")
            return True
        
        return False
    
    def stop(self):
        """Stop web server"""
        if self.running:
            self.running = False
            # Note: Flask doesn't have a clean shutdown method in development mode
            self.logger.info("Web server stop requested")
    
    def _run_server(self):
        """Run Flask server"""
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Error running web server: {e}")
        finally:
            self.running = False