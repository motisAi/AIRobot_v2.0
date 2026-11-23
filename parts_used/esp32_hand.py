"""Robotic hand control — persistent serial link to the ESP32 gesture firmware.

Opens the serial port ONCE and holds it, so the ESP32 keeps its pose between
commands (no reset-per-command). The single connect at startup resets the ESP32,
which homes it to a closed fist. Stella then just sends short gesture names
("hello", "middle", "thumbs", ...) and the ESP32 performs the smooth motion.
"""

from __future__ import annotations

import logging
import threading
import time

try:
    import serial  # pyserial
    SERIAL_OK = True
except Exception:  # pragma: no cover
    serial = None
    SERIAL_OK = False

logger = logging.getLogger("Hand")

# friendly name -> firmware command
GESTURES = {
    "wave": "hello", "hello": "hello", "hi": "hello",
    "fist": "fist", "home": "fist", "close": "fist",
    "open": "open", "openhand": "open",
    "thumbs_up": "thumbs", "thumbsup": "thumbs", "thumbs": "thumbs",
    "point": "point", "peace": "peace", "victory": "peace",
    "middle_finger": "middle", "middle": "middle",
    "middle_finger_hold": "middlehold", "raise_middle": "middlehold",
    "lower": "fist", "rest": "fist", "lower_hand": "fist",
}


class Hand:
    def __init__(self, port="/dev/ttyUSB0", baud=115200, enabled=True, link=None):
        # link: an object with send_line(str)->bool and an `available` property
        # (the EyeController). When given, the hand shares that ONE serial link
        # because the hand and eyes are the same ESP32-S3 on one USB cable; the
        # hand opens no port of its own.
        self.port = port
        self.baud = baud
        self._link = link
        self.enabled = bool(enabled) and (SERIAL_OK or link is not None)
        self._ser = None
        self._lock = threading.Lock()
        if self._link is None and self.enabled:
            self._connect()

    def _resolve_port(self) -> str:
        """Allow a /dev/serial/by-id glob (e.g. 'usb-1a86_*-if00') so the hand
        never grabs another board's /dev/ttyACMn (the eyes' ESP32-C3 took
        ttyACM0 the day it was plugged in). Returns the first match, or the
        configured string unchanged when it has no wildcard."""
        if "*" not in self.port and "?" not in self.port:
            return self.port
        import glob
        matches = sorted(glob.glob(self.port))
        if not matches:
            raise FileNotFoundError(f"no serial device matches {self.port}")
        return matches[0]

    def _connect(self):
        try:
            self._ser = serial.Serial(self._resolve_port(), self.baud, timeout=1, write_timeout=2)
            time.sleep(2.5)   # the ESP32 resets on connect; let it boot (servos stay limp until commanded)
            logger.info("Hand connected on %s (ESP32 restarted, servos limp until commanded)", self.port)
        except Exception as exc:
            logger.warning("Hand serial connect failed (%s): %s", self.port, exc)
            self._ser = None

    @property
    def available(self) -> bool:
        if self._link is not None:
            return bool(getattr(self._link, "available", False))
        return self.enabled and self._ser is not None

    def reconnect(self, port: str = None) -> bool:
        """Re-open the serial link (e.g. after the ESP32 got a new /dev port)."""
        if self._link is not None:
            return self.available   # the shared link (eyes) owns reconnection
        if not self.enabled:
            return False
        with self._lock:
            if self._ser:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        if port:
            self.port = port
        self._connect()
        return self.available

    def send(self, cmd: str) -> bool:
        if self._link is not None:
            return self._link.send_line(cmd.strip())
        if not self.available:
            return False
        with self._lock:
            try:
                self._ser.write((cmd.strip() + "\n").encode())
                return True
            except Exception as exc:
                logger.warning("hand send failed: %s", exc)
                return False

    def gesture(self, name: str) -> bool:
        cmd = GESTURES.get(str(name).lower().strip().replace(" ", "_"))
        if not cmd:
            logger.info("unknown gesture: %s", name)
            return False
        logger.info("gesture: %s", cmd)
        return self.send(cmd)

    # convenience wrappers
    def wave(self): return self.send("hello")
    def fist(self): return self.send("fist")
    def open_hand(self): return self.send("open")
    def thumbs_up(self): return self.send("thumbs")
    def middle_finger(self): return self.send("middle")
    def count(self, n: int): return self.send(f"count {int(n)}")

    def stop(self):
        with self._lock:
            if self._ser:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
