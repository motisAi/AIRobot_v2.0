"""Face bridge — serves Stella's on-demand animated face and drives it.

Additive & isolated: runs its own aiohttp server in a dedicated thread. A failure
here must NEVER take Stella down (main.py wraps startup in try/except).

Endpoints (LAN, plain HTTP on :8080):
  GET  /               -> the face web app (single self-contained page)
  GET  /ws             -> WebSocket; the browser face connects here and receives
                          emotion / look_at / state messages (M1+; M0 just blinks)
  GET  /display_state  -> {"show": bool}  the Surface controller polls this
  GET  /show , /hide   -> flip the flag (voice handler + manual/testing override)

The Surface runs a tiny poller (face_client.ps1) that launches/kills Edge kiosk
based on /display_state, so the face only appears when you ask for it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading

from aiohttp import web, WSMsgType

logger = logging.getLogger("FaceBridge")
WEBROOT = os.path.join(os.path.dirname(__file__), "webface")


class FaceBridge:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = int(port)
        self._loop = None
        self._thread = None
        self._faces = set()      # connected browser faces (ws)
        self._show = False       # desired display state (polled by the Surface)
        self._started = False

    # ---- public, thread-safe API (called from Stella's threads) ------------
    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, name="face_bridge", daemon=True)
        self._thread.start()

    @property
    def showing(self) -> bool:
        return self._show

    def show_face(self):
        self._show = True
        logger.info("face -> SHOW")
        self._broadcast({"type": "display", "mode": "show"})

    def hide_face(self):
        self._show = False
        logger.info("face -> HIDE")
        self._broadcast({"type": "display", "mode": "hide"})

    # M1+ hooks (safe to call now; no face connected = no-op)
    def set_emotion(self, value: str, intensity: float = 1.0):
        self._broadcast({"type": "emotion", "value": value, "intensity": float(intensity)})

    def look_at(self, x: float, y: float):
        self._broadcast({"type": "look_at", "x": float(x), "y": float(y)})

    def gaze_mode(self, mode: str):
        self._broadcast({"type": "gaze_mode", "value": mode})

    def set_state(self, **kw):
        self._broadcast({"type": "state", **kw})

    def speak_start(self):
        self._broadcast({"type": "speak_start"})

    def speak_end(self):
        self._broadcast({"type": "speak_end"})

    def viseme(self, level: float):
        self._broadcast({"type": "viseme", "level": float(level)})

    # ---- internals ---------------------------------------------------------
    def _broadcast(self, msg: dict):
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._do_broadcast, msg)
        except Exception:
            pass

    def _do_broadcast(self, msg: dict):
        data = json.dumps(msg)
        for ws in list(self._faces):
            try:
                asyncio.create_task(ws.send_str(data))
            except Exception:
                pass

    def _run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            app = web.Application()
            app.router.add_get("/", self._index)
            app.router.add_get("/ws", self._ws)
            app.router.add_get("/display_state", self._state)
            app.router.add_get("/show", self._show_ep)
            app.router.add_get("/hide", self._hide_ep)
            app.router.add_get("/test", self._test_ep)
            app.router.add_get("/face.png", self._asset)
            app.router.add_get("/face_meta.json", self._asset)
            runner = web.AppRunner(app)
            self._loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, self.host, self.port)
            self._loop.run_until_complete(site.start())
            logger.info("Face bridge on http://%s:%d", self.host, self.port)
            self._loop.run_forever()
        except Exception as exc:
            logger.warning("face bridge stopped: %s", exc)

    async def _index(self, request):
        path = os.path.join(WEBROOT, "index.html")
        if os.path.exists(path):
            return web.FileResponse(path)
        return web.Response(text="face app missing", status=404)

    async def _asset(self, request):
        name = os.path.basename(request.path)
        path = os.path.join(WEBROOT, name)
        if os.path.isfile(path):
            return web.FileResponse(path)
        return web.Response(status=404)

    async def _state(self, request):
        return web.json_response({"show": self._show})

    async def _show_ep(self, request):
        self.show_face()
        return web.json_response({"show": True})

    async def _hide_ep(self, request):
        self.hide_face()
        return web.json_response({"show": False})

    async def _test_ep(self, request):
        """Visual self-test: go happy + talk for 3s, then relax to neutral."""
        self.set_emotion("happy")
        self.speak_start()

        def done():
            self.speak_end()
            self.set_emotion("neutral")
        self._loop.call_later(3.0, done)
        return web.json_response({"test": "happy + talking for 3s"})

    async def _ws(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._faces.add(ws)
        logger.info("face connected (%d live)", len(self._faces))
        # tell a freshly-connected face the current display intent
        try:
            await ws.send_str(json.dumps({"type": "display",
                                          "mode": "show" if self._show else "hide"}))
        except Exception:
            pass
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
                # telemetry (battery/fps/touch) handled in M4; ignore for now
        finally:
            self._faces.discard(ws)
            logger.info("face disconnected (%d live)", len(self._faces))
        return ws


# Standalone test:  python -m face_bridge.bridge   (then open http://<pi>:8080/)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fb = FaceBridge()
    fb.start()
    fb.show_face()
    import time
    print("Face bridge running on :8080 — open it in a browser. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
