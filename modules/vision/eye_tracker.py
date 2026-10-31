"""Stella's LCD eyes — Pi-side controller for the ESP32-C3 + dual GC9D01 board.

Firmware: firmware/stella_eyes/stella_eyes.ino (serial, 115200, one command per
line):  G <x> <y>  gaze -100..100 (x: + = her right, y: + = down)
        B          blink          C  centre        E <n>  emotion 0-4
        P <n>      pupil 0-100
The ESP32 eases toward each target itself, so the Pi just streams targets.

Hardware identity: the board enumerates on its NATIVE USB port as
/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_<mac>-if00 (pinned in
config hardware.eyes_serial; a glob is fine). Plugged into its UART type-C it
would appear as a CH343 ("usb-1a86_*") just like the hand — don't.

Everything is guarded: no port -> every call is a cheap no-op, and the link is
re-tried in the background every few seconds, so the service never depends on
the eyes being there.
"""
from __future__ import annotations

import glob
import logging
import threading
import time
from typing import Optional, Tuple

try:
    import serial  # pyserial
    SERIAL_OK = True
except Exception:  # pragma: no cover
    serial = None
    SERIAL_OK = False

logger = logging.getLogger("Eyes")

EMOTIONS = {"neutral": 0, "happy": 1, "sad": 2, "surprised": 3, "sleepy": 4, "angry": 2}
DEFAULT_GLOB = "/dev/serial/by-id/usb-Espressif_*-if00"


class EyeController:
    def __init__(self, port: str = "", baud: int = 115200, enabled: bool = True,
                 invert_x: bool = False, invert_y: bool = False, retry_s: float = 5.0):
        self.port_spec = port or DEFAULT_GLOB
        self.baud = int(baud)
        self.enabled = bool(enabled) and SERIAL_OK
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)
        self.retry_s = float(retry_s)
        self._ser = None
        self._lock = threading.Lock()
        self._last_try = 0.0
        self._last_gaze: Tuple[int, int] = (0, 0)
        self._last_sent = 0.0
        self.emotion = "neutral"
        if self.enabled:
            self._connect()

    # ------------------------------------------------------------ link
    def _resolve_port(self) -> Optional[str]:
        if "*" in self.port_spec or "?" in self.port_spec:
            m = sorted(glob.glob(self.port_spec))
            return m[0] if m else None
        return self.port_spec

    def _connect(self) -> bool:
        self._last_try = time.time()
        path = self._resolve_port()
        if not path:
            return False
        try:
            self._ser = serial.Serial(path, self.baud, timeout=0.2, write_timeout=0.5)
            time.sleep(0.3)
            logger.info("Eyes connected on %s", path)
            self._send("C")
            return True
        except Exception as exc:
            logger.warning("Eyes serial connect failed (%s): %s", path, exc)
            self._ser = None
            return False

    @property
    def available(self) -> bool:
        return self.enabled and self._ser is not None

    def _ensure(self) -> bool:
        """Lazy reconnect, rate-limited, so a board plugged in later just works."""
        if not self.enabled:
            return False
        if self._ser is not None:
            return True
        if time.time() - self._last_try < self.retry_s:
            return False
        return self._connect()

    def _send(self, line: str) -> bool:
        if not self._ensure():
            return False
        with self._lock:
            try:
                self._ser.write((line.strip() + "\n").encode("ascii"))
                return True
            except Exception as exc:
                logger.warning("eyes send failed (%s) — will reconnect", exc)
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                return False

    def send_line(self, line: str) -> bool:
        """Public raw-send: used by the Hand module to share this one serial link
        when the hand and eyes are the same ESP32-S3 on one USB cable."""
        return self._send(line)

    def close(self):
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    # ------------------------------------------------------------ gaze
    def look_at_norm(self, gx: float, gy: float, min_interval: float = 0.05) -> bool:
        """gx, gy in -1..1 (x: -1 = her left, +1 = her right; y: -1 = up, +1 = down).
        Rate-limited and de-duplicated so the serial link is never flooded."""
        if self.invert_x:
            gx = -gx
        if self.invert_y:
            gy = -gy
        x = int(max(-100, min(100, round(gx * 100))))
        y = int(max(-100, min(100, round(gy * 100))))
        now = time.time()
        if (x, y) == self._last_gaze and now - self._last_sent < 1.0:
            return True
        if now - self._last_sent < min_interval:
            return True
        self._last_gaze = (x, y)
        self._last_sent = now
        return self._send(f"G {x} {y}")

    def look_at_bbox(self, box, frame_w: int, frame_h: int, mirror_x: bool = True) -> bool:
        """Aim at a face box (x, y, w, h) in camera pixels. The camera looks at the
        person, so a face on the image's LEFT is on her RIGHT -> mirror_x=True
        (same convention as the tablet face's gaze feed)."""
        x, y, w, h = box
        nx = (x + w / 2.0) / max(1, frame_w)
        ny = (y + h / 2.0) / max(1, frame_h)
        gx = ((1.0 - nx) if mirror_x else nx) * 2.0 - 1.0
        gy = (ny * 2.0 - 1.0) * 0.7
        return self.look_at_norm(gx, gy)

    def center(self) -> bool:
        self._last_gaze = (0, 0)
        return self._send("C")

    def idle(self) -> bool:
        """Nobody in view: centre; the firmware adds its own gentle drift + blinks."""
        return self.center()

    # ------------------------------------------------------------ expression
    def blink(self) -> bool:
        return self._send("B")

    def set_emotion(self, name: str) -> bool:
        code = EMOTIONS.get(str(name).lower().strip(), 0)
        self.emotion = name
        return self._send(f"E {code}")

    def set_pupil(self, percent: int) -> bool:
        return self._send(f"P {int(max(0, min(100, percent)))}")

    def status(self) -> dict:
        return {"enabled": self.enabled, "connected": self._ser is not None,
                "port": self._resolve_port(), "gaze": self._last_gaze, "emotion": self.emotion}
