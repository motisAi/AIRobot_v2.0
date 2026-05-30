"""Web Dashboard
===============
Lightweight Flask web UI for monitoring the robot from any browser.

Features:
 - Live MJPEG camera stream
 - Real-time status (state, detected objects, recognised faces)
 - System metrics (CPU, RAM, temperature)
 - Toggle controls (patrol, learning, etc.)

Runs in a daemon thread so it never blocks the main robot loop.
Access at  http://<pi-ip>:5000
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from modules.hardware.camera_manager import CameraManager, Frame
    from core.robot_brain import RobotBrain
    from modules.ai.learning_db import LearningDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template (self-contained, no external files needed)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ name }} — Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background:#0f1117; color:#e0e0e0; }
  header { background:#1a1d27; padding:12px 24px; display:flex; align-items:center; gap:12px; border-bottom:1px solid #2a2d37; }
  header h1 { font-size:1.3rem; color:#4fc3f7; }
  header .dot { width:10px; height:10px; border-radius:50%; background:#4caf50; animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:16px; max-width:1200px; margin:auto; }
  .card { background:#1a1d27; border-radius:10px; padding:16px; border:1px solid #2a2d37; }
  .card h2 { font-size:.95rem; color:#90caf9; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px; }
  .cam-box { grid-column:1/2; grid-row:1/3; }
  .cam-box img { width:100%; border-radius:8px; background:#000; }
  table { width:100%; border-collapse:collapse; }
  td { padding:5px 8px; border-bottom:1px solid #2a2d37; font-size:.88rem; }
  td:first-child { color:#78909c; width:40%; }
  .tag { display:inline-block; background:#263238; padding:2px 8px; border-radius:4px; margin:2px; font-size:.82rem; }
  .bar { height:6px; background:#263238; border-radius:3px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:3px; transition:width .5s; }
  .bar-cpu .bar-fill { background:#42a5f5; }
  .bar-ram .bar-fill { background:#66bb6a; }
  .bar-temp .bar-fill { background:#ff7043; }
  .log-box { grid-column:1/-1; max-height:180px; overflow-y:auto; font-family:monospace; font-size:.8rem; line-height:1.5; background:#12141b; padding:10px; border-radius:6px; }
  @media (max-width:700px) { .grid{grid-template-columns:1fr;} .cam-box{grid-column:1;grid-row:auto;} }
</style>
</head>
<body>
<header>
  <div class="dot" id="alive"></div>
  <h1>{{ name }}</h1>
</header>
<div class="grid">

  <div class="card cam-box">
    <h2>Camera</h2>
    <img id="cam" src="/video_feed" alt="camera">
  </div>

  <div class="card">
    <h2>Robot Status</h2>
    <table>
      <tr><td>State</td><td id="state">—</td></tr>
      <tr><td>Current User</td><td id="user">—</td></tr>
      <tr><td>Uptime</td><td id="uptime">—</td></tr>
      <tr><td>Objects</td><td id="objects">—</td></tr>
      <tr><td>Faces Seen</td><td id="faces">—</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>System</h2>
    <table>
      <tr><td>CPU</td><td><div class="bar bar-cpu"><div class="bar-fill" id="cpu" style="width:0%"></div></div></td></tr>
      <tr><td>RAM</td><td><div class="bar bar-ram"><div class="bar-fill" id="ram" style="width:0%"></div></div></td></tr>
      <tr><td>Temp</td><td><div class="bar bar-temp"><div class="bar-fill" id="temp" style="width:0%"></div></div></td></tr>
      <tr><td>CPU %</td><td id="cpu_val">—</td></tr>
      <tr><td>RAM %</td><td id="ram_val">—</td></tr>
      <tr><td>Temp °C</td><td id="temp_val">—</td></tr>
    </table>
  </div>

  <div class="card log-box" id="log"></div>

</div>
<script>
async function poll(){
  try{
    const r=await fetch('/api/status');
    const d=await r.json();
    document.getElementById('state').textContent=d.state||'—';
    document.getElementById('user').textContent=d.current_user||'nobody';
    document.getElementById('uptime').textContent=d.uptime||'—';
    document.getElementById('objects').innerHTML=(d.objects||[]).map(o=>'<span class="tag">'+o+'</span>').join(' ')||'—';
    document.getElementById('faces').textContent=d.faces_known||0;
    const cpu=d.cpu||0, ram=d.ram||0, tmp=d.temp||0;
    document.getElementById('cpu').style.width=cpu+'%';
    document.getElementById('ram').style.width=ram+'%';
    document.getElementById('temp').style.width=Math.min(tmp,100)+'%';
    document.getElementById('cpu_val').textContent=cpu.toFixed(1)+'%';
    document.getElementById('ram_val').textContent=ram.toFixed(1)+'%';
    document.getElementById('temp_val').textContent=tmp.toFixed(1)+'°C';
    if(d.last_log){
      const box=document.getElementById('log');
      const line=document.createElement('div');
      line.textContent=d.last_log;
      box.appendChild(line);
      box.scrollTop=box.scrollHeight;
      while(box.childElementCount>200) box.removeChild(box.firstChild);
    }
  }catch(e){}
}
setInterval(poll,2000);
poll();
</script>
</body>
</html>
"""


class WebDashboard:
    """Flask-based web dashboard running in a background thread."""

    def __init__(
        self,
        camera_manager: Optional[CameraManager] = None,
        brain: Optional[RobotBrain] = None,
        learning_db: Optional[LearningDB] = None,
        robot_name: str = "Gonzo",
        host: str = "0.0.0.0",
        port: int = 5000,
    ):
        self.camera_manager = camera_manager
        self.brain = brain
        self.learning_db = learning_db
        self.robot_name = robot_name
        self.host = host
        self.port = port

        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._start_time = time.time()
        self._log_lines: list[str] = []
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Camera subscriber
    # ------------------------------------------------------------------
    def _on_frame(self, frame: Frame):
        """Called by CameraManager for every new frame."""
        ret, jpeg = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with self._jpeg_lock:
                self._latest_jpeg = jpeg.tobytes()

    # ------------------------------------------------------------------
    # Flask app factory
    # ------------------------------------------------------------------
    def _make_app(self):
        from flask import Flask, Response, jsonify

        app = Flask(__name__)
        app.logger.setLevel(logging.WARNING)  # quiet Flask logs

        @app.route("/")
        def index():
            return HTML_PAGE.replace("{{ name }}", self.robot_name)

        @app.route("/video_feed")
        def video_feed():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/api/status")
        def api_status():
            return jsonify(self._gather_status())

        return app

    # ------------------------------------------------------------------
    # MJPEG generator
    # ------------------------------------------------------------------
    def _mjpeg_gen(self):
        """Yield JPEG frames as an MJPEG stream."""
        while True:
            with self._jpeg_lock:
                frame = self._latest_jpeg
            if frame is None:
                # Send a 1x1 black pixel if no camera frame yet
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "No camera", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
                _, buf = cv2.imencode(".jpg", blank)
                frame = buf.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.1)  # ~10 fps for browser

    # ------------------------------------------------------------------
    # Status data
    # ------------------------------------------------------------------
    def _gather_status(self) -> dict:
        data: dict = {}

        # Uptime
        secs = int(time.time() - self._start_time)
        h, m, s = secs // 3600, secs % 3600 // 60, secs % 60
        data["uptime"] = f"{h}h {m}m {s}s"

        # Brain state
        if self.brain:
            state = getattr(self.brain, "state", "unknown")
            data["state"] = state.name if hasattr(state, 'name') else str(state)
            data["current_user"] = getattr(self.brain, "current_user", None)

        # Detected objects (from object_detection module if available)
        objects = []
        if self.brain and hasattr(self.brain, "_robot") and self.brain._robot:
            od = getattr(self.brain._robot, "modules", {}).get("object_detection")
            if od and hasattr(od, "get_current_objects"):
                objects = [o.label for o in od.get_current_objects()]
        data["objects"] = objects

        # Known faces count
        faces_known = 0
        if self.learning_db:
            try:
                faces_known = len(self.learning_db.load_faces())
            except Exception:
                pass
        data["faces_known"] = faces_known

        # System metrics
        try:
            import psutil
            data["cpu"] = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            data["ram"] = mem.percent
        except ImportError:
            data["cpu"] = 0
            data["ram"] = 0

        # CPU temperature (RPi)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                data["temp"] = int(f.read().strip()) / 1000.0
        except Exception:
            data["temp"] = 0

        # Last log line
        data["last_log"] = self._log_lines[-1] if self._log_lines else None

        return data

    # ------------------------------------------------------------------
    # Logging hook
    # ------------------------------------------------------------------
    def add_log(self, message: str):
        """Append a log message (shown in the dashboard footer)."""
        ts = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{ts}] {message}")
        if len(self._log_lines) > 500:
            self._log_lines = self._log_lines[-300:]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the dashboard in a daemon thread."""
        if self._thread and self._thread.is_alive():
            return

        # Subscribe to camera
        if self.camera_manager:
            self.camera_manager.subscribe("web_dashboard", self._on_frame)

        app = self._make_app()

        def _run():
            logger.info("Web dashboard starting on http://%s:%d", self.host, self.port)
            # Use Werkzeug's quiet server
            app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

        self._thread = threading.Thread(target=_run, name="web-dashboard", daemon=True)
        self._thread.start()
        logger.info("Dashboard available at http://0.0.0.0:%d", self.port)

    def stop(self):
        """Unsubscribe from camera. (Flask thread is daemon — dies with process.)"""
        if self.camera_manager:
            self.camera_manager.unsubscribe("web_dashboard")
        logger.info("Web dashboard stopped")
