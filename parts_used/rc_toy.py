"""RC toy chassis — Stella's future wheels (a cheap RC car/tank she drives herself).

No toy has been chosen yet, so this is the CONTRACT plus a safe no-op backend:
    rc_toy.connected: false   -> RCToy exists, is_available() is False, every drive() is a no-op
    rc_toy.connected: true    -> the transport named in rc_toy.transport is used

Transports (add a small class here when the toy arrives):
    null          — logs only (default)
    esp32_serial  — an ESP32 on the toy's remote/H-bridge, line protocol "drive <fwd> <turn>\\n"
    wifi_http     — toy or ESP32 with an HTTP endpoint, GET {address}/drive?f=..&t=..
    ble           — Bluetooth LE toy (needs `bleak`; not installed by default)

Safety: speeds are clamped to ±max_speed; a dead-man timer stops the toy if no command
arrives for `deadman_seconds`; every voice/LLM command is master-gated in the
conversation manager (the tool only exists when connected is true).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional


class _NullTransport:
    name = "null"

    def __init__(self, cfg, log):
        self.log = log

    def open(self) -> bool:
        return True

    def send(self, forward: float, turn: float) -> None:
        self.log.debug("RC(null) drive f=%.2f t=%.2f", forward, turn)

    def close(self) -> None:
        pass


class _SerialTransport:
    """ESP32/Arduino on the toy over USB serial. Protocol: 'drive <f> <t>' / 'stop'."""
    name = "esp32_serial"

    def __init__(self, cfg, log):
        self.cfg, self.log, self.ser = cfg, log, None

    def open(self) -> bool:
        try:
            import serial  # pyserial is already a project dependency
            self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=0.2)
            time.sleep(1.5)  # ESP32 resets on open
            return True
        except Exception as e:  # noqa: BLE001
            self.log.warning("RC toy serial %s not available: %s", self.cfg.port, e)
            return False

    def send(self, forward: float, turn: float) -> None:
        if self.ser:
            self.ser.write(f"drive {forward:.2f} {turn:.2f}\n".encode())

    def close(self) -> None:
        if self.ser:
            try:
                self.ser.write(b"stop\n")
                self.ser.close()
            except Exception:  # noqa: BLE001
                pass


class _HttpTransport:
    """Toy (or its ESP32) reachable over WiFi/RobotNet: GET {address}/drive?f=..&t=.."""
    name = "wifi_http"

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log

    def open(self) -> bool:
        return bool(self.cfg.address)

    def send(self, forward: float, turn: float) -> None:
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.cfg.address.rstrip('/')}/drive?f={forward:.2f}&t={turn:.2f}", timeout=1.0)
        except Exception as e:  # noqa: BLE001
            self.log.debug("RC toy http send failed: %s", e)

    def close(self) -> None:
        self.send(0.0, 0.0)


_TRANSPORTS = {"null": _NullTransport, "esp32_serial": _SerialTransport, "wifi_http": _HttpTransport}


class RCToy:
    """Drive interface: forward/turn in -1..1, clamped to config.max_speed."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger("RCToy")
        self._t = None
        self._available = False
        self._last_cmd = 0.0
        self._lock = threading.Lock()
        self._deadman: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    # -- part contract ---------------------------------------------------
    def is_available(self) -> bool:
        return self._available

    def start(self) -> bool:
        if not getattr(self.cfg, "connected", False):
            self.log.info("RC toy not connected (rc_toy.connected: false) — driving disabled")
            return False
        cls = _TRANSPORTS.get(getattr(self.cfg, "transport", "null"), _NullTransport)
        self._t = cls(self.cfg, self.log)
        self._available = bool(self._t.open())
        if self._available:
            self._stop_evt.clear()
            self._deadman = threading.Thread(target=self._deadman_loop, name="rc-deadman", daemon=True)
            self._deadman.start()
            self.log.info("✓ RC toy ready via %s", self._t.name)
        return self._available

    def stop(self) -> None:
        self._stop_evt.set()
        if self._t:
            try:
                self._t.send(0.0, 0.0)
                self._t.close()
            except Exception:  # noqa: BLE001
                pass
        self._available = False

    # -- driving ---------------------------------------------------------
    def drive(self, forward: float, turn: float = 0.0) -> bool:
        """forward: -1 (back) .. 1 (ahead); turn: -1 (left) .. 1 (right)."""
        if not self._available:
            return False
        m = float(getattr(self.cfg, "max_speed", 0.5))
        f = max(-m, min(m, float(forward)))
        t = max(-1.0, min(1.0, float(turn)))
        with self._lock:
            self._t.send(f, t)
            self._last_cmd = time.monotonic()
        return True

    def halt(self) -> None:
        if self._available:
            with self._lock:
                self._t.send(0.0, 0.0)

    def _deadman_loop(self) -> None:
        limit = float(getattr(self.cfg, "deadman_seconds", 2.0))
        while not self._stop_evt.wait(0.25):
            if self._last_cmd and time.monotonic() - self._last_cmd > limit:
                self.halt()
                self._last_cmd = 0.0
