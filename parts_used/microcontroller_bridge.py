"""Generic microcontroller bridge (ESP32 / Pi Zero / Arduino).

This is the single, hardware-agnostic sink for "real-world" commands the robot
decides to perform — turning on a light, driving a motor, reading a sensor.

Design goals
------------
* **One flag to go live.** ``microcontroller.connected: false`` in
  ``config/config.yaml`` keeps everything in *log-only* mode: commands are
  formed and logged but never transmitted, so the entire voice pipeline can be
  built and tested before any hardware exists. Flip it to ``true`` and pick a
  transport to start sending real commands — no code changes.
* **Pluggable transports.** Talk to the microcontroller over USB serial
  (ESP32/Arduino) *or* the network (a Pi Zero / ESP32 on Wi-Fi). Both speak a
  tiny JSON line protocol that is trivial to implement in firmware.
* **Friendly names.** ``device_map`` in the config maps what the user *says*
  ("light") to what the firmware *understands* ("relay1").

Firmware protocol (JSON, one message per line)
----------------------------------------------
Robot -> MCU :  {"action": "on", "target": "relay1"}\\n
MCU  -> Robot:  {"ok": true, "target": "relay1", "state": "on"}\\n

For the network transport the same JSON is POSTed to
``http://<host>:<port><network_path>`` and the JSON body of the reply is used.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

try:
    import serial  # pyserial
    SERIAL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    serial = None
    SERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
class BaseTransport:
    """Common transport interface. ``send`` returns the parsed reply or None."""

    name = "base"

    def open(self) -> bool:
        return True

    def close(self) -> None:
        pass

    @property
    def available(self) -> bool:
        return True

    def send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class NullTransport(BaseTransport):
    """Log-only transport used when no microcontroller is connected."""

    name = "null"

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.logger.info("[microcontroller:OFF] would send -> %s", json.dumps(message))
        return {"ok": True, "logged": True, "sent": False}


class SerialTransport(BaseTransport):
    """JSON-over-USB-UART transport (ESP32 / Arduino)."""

    name = "serial"

    def __init__(self, port: str, baudrate: int, timeout: float,
                 ack_timeout: float, logger: logging.Logger):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ack_timeout = ack_timeout
        self.logger = logger
        self._ser = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        if not SERIAL_AVAILABLE:
            self.logger.error("pyserial not installed — cannot open serial transport")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)  # allow the board to reset after opening the port
            self.logger.info("Serial transport open on %s @ %d", self.port, self.baudrate)
            return True
        except Exception as exc:
            self.logger.error("Failed to open serial port %s: %s", self.port, exc)
            self._ser = None
            return False

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    @property
    def available(self) -> bool:
        return self._ser is not None

    def send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._ser is None:
            return None
        line = (json.dumps(message) + "\n").encode("utf-8")
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write(line)
                self._ser.flush()
            except Exception as exc:
                self.logger.error("Serial write failed: %s", exc)
                return None
            # Best-effort read of a single JSON reply line within ack_timeout.
            deadline = time.time() + self.ack_timeout
            buf = b""
            while time.time() < deadline:
                try:
                    chunk = self._ser.readline()
                except Exception:
                    break
                if chunk:
                    buf = chunk
                    break
        if not buf:
            return {"ok": True, "ack": False}  # sent, but no reply
        try:
            return json.loads(buf.decode("utf-8", errors="ignore").strip())
        except Exception:
            return {"ok": True, "raw": buf.decode("utf-8", errors="ignore").strip()}


class NetworkTransport(BaseTransport):
    """JSON-over-HTTP transport (Pi Zero / ESP32 on Wi-Fi)."""

    name = "network"

    def __init__(self, host: str, port: int, path: str,
                 ack_timeout: float, logger: logging.Logger):
        self.url = f"http://{host}:{port}{path}"
        self.ack_timeout = ack_timeout
        self.logger = logger

    def send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.ack_timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
        except Exception as exc:
            self.logger.error("Network command to %s failed: %s", self.url, exc)
            return None
        if not body:
            return {"ok": True, "ack": False}
        try:
            return json.loads(body)
        except Exception:
            return {"ok": True, "raw": body}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class MicrocontrollerController:
    """High-level API the robot brain talks to for real-world actuation.

    The brain never needs to know which transport is in use, or whether a
    microcontroller is even connected. It just calls ``set_output``,
    ``send_command`` or ``move``.
    """

    def __init__(self, cfg=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if cfg is None:
            from config.settings import microcontroller_config as cfg
        self.cfg = cfg
        self.device_map = dict(getattr(cfg, "device_map", {}) or {})
        self._transport: BaseTransport = NullTransport(self.logger)
        self._connected = False
        self._stats = {"sent": 0, "ok": 0, "failed": 0}

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> bool:
        """Select and open the configured transport.

        When ``connected`` is False (or transport is ``null``) this installs the
        log-only NullTransport and still returns True, so the command path is
        always live and safe.
        """
        if not getattr(self.cfg, "connected", False):
            self._transport = NullTransport(self.logger)
            self._connected = False
            self.logger.info(
                "Microcontroller not connected — commands run in log-only mode "
                "(set microcontroller.connected: true to go live)"
            )
            return True

        transport = getattr(self.cfg, "transport", "serial").lower()
        if transport == "serial":
            self._transport = SerialTransport(
                self.cfg.serial_port, self.cfg.baudrate,
                self.cfg.serial_timeout, self.cfg.ack_timeout, self.logger,
            )
        elif transport == "network":
            self._transport = NetworkTransport(
                self.cfg.host, self.cfg.port, self.cfg.network_path,
                self.cfg.ack_timeout, self.logger,
            )
        else:
            self._transport = NullTransport(self.logger)

        ok = self._transport.open()
        if not ok:
            self.logger.warning(
                "Falling back to log-only mode (transport '%s' failed to open)",
                transport,
            )
            self._transport = NullTransport(self.logger)
            self._connected = False
            return True

        self._connected = True
        self.logger.info("Microcontroller connected via %s transport", transport)
        return True

    def start(self) -> bool:
        """Alias so the brain's start_modules() loop can start us uniformly."""
        return self.connect()

    def stop(self) -> None:
        try:
            self._transport.close()
        finally:
            self._connected = False

    def shutdown(self) -> None:
        self.stop()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- core send ---------------------------------------------------------
    def _send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._stats["sent"] += 1
        reply = self._transport.send(message)
        if reply is not None and reply.get("ok", True):
            self._stats["ok"] += 1
        else:
            self._stats["failed"] += 1
        return reply

    def resolve_target(self, name: Optional[str]) -> Optional[str]:
        """Map a spoken device name ("light") to a firmware id ("relay1")."""
        if not name:
            return None
        return self.device_map.get(str(name).lower(), str(name))

    def send_command(self, action: str, target: Optional[str] = None,
                     **params: Any) -> Optional[Dict[str, Any]]:
        """Send an arbitrary command. Returns the firmware reply (or None)."""
        message: Dict[str, Any] = {"action": action}
        resolved = self.resolve_target(target)
        if resolved is not None:
            message["target"] = resolved
        if params:
            message.update(params)
        return self._send(message)

    # -- convenience API used by the brain --------------------------------
    def set_output(self, name: str, state: bool) -> Optional[Dict[str, Any]]:
        """Turn a named output on/off — e.g. set_output('light', True)."""
        return self.send_command("on" if state else "off", target=name)

    def move(self, movement_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Drive command for future wheels. movement_data may hold direction,
        speed, duration."""
        params = movement_data if isinstance(movement_data, dict) else {}
        return self.send_command("move", **params)

    def stop_all_motors(self) -> Optional[Dict[str, Any]]:
        return self.send_command("stop")

    def set_led(self, color: str) -> Optional[Dict[str, Any]]:
        return self.send_command("led", color=color)

    def set_gpio(self, pin: int, value: int) -> Optional[Dict[str, Any]]:
        return self.send_command("gpio", pin=pin, value=value)

    def read_sensor(self, name: str) -> Optional[Any]:
        reply = self.send_command("read", target=name)
        if reply is None:
            return None
        return reply.get("value", reply)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "transport": self._transport.name,
            **self._stats,
        }
