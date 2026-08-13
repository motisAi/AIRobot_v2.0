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
    def __init__(self, port="/dev/ttyUSB0", baud=115200, enabled=True):
        self.port = port
        self.baud = baud
        self.enabled = bool(enabled) and SERIAL_OK
        self._ser = None
        self._lock = threading.Lock()
        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2.5)   # ESP32 resets on connect and homes to a fist
            logger.info("Hand connected on %s (ESP32 homed to fist)", self.port)
        except Exception as exc:
            logger.warning("Hand serial connect failed (%s): %s", self.port, exc)
            self._ser = None

    @property
    def available(self) -> bool:
        return self.enabled and self._ser is not None

    def send(self, cmd: str) -> bool:
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
