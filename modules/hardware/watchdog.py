"""Hardware watchdog — background self-healing for Stella's peripherals.

Device indices/ports on the Pi reshuffle across reboots and re-plugs (USB mics,
the ESP32 hand's /dev/ttyACM*, HDMI ALSA cards). This thread periodically checks
each subsystem and repairs it in place, so a shifted device fixes itself instead
of needing a manual restart.

Checks (every `interval` seconds):
  * Camera   — warn if frames go stale (the CameraManager reopens on its own).
  * Hand     — if the serial port vanished / went dead, re-detect it and reconnect.
  * Audio out— if the TTS output device stops opening (HDMI card renumbered),
               re-resolve it. Conservative: only after 2 consecutive failures and
               never while she's speaking (to avoid the HDMI 'busy' false alarm).
The wake/command mics already self-heal by re-resolving their index by name.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time

logger = logging.getLogger("Watchdog")


class HardwareWatchdog:
    def __init__(self, robot, interval: float = 20.0):
        self.robot = robot
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread = None
        self._audio_fails = 0

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hw_watchdog")
        self._thread.start()
        logger.info("Hardware watchdog started (every %ds)", int(self.interval))

    def stop(self):
        self._stop.set()

    def _loop(self):
        self._stop.wait(self.interval)   # let startup settle
        while not self._stop.is_set():
            for check in (self._check_camera, self._check_hand, self._check_audio_out):
                try:
                    check()
                except Exception as exc:
                    logger.debug("%s error: %s", check.__name__, exc)
            self._stop.wait(self.interval)

    # -- camera -----------------------------------------------------------
    def _check_camera(self):
        cm = getattr(self.robot, "camera_manager", None)
        if cm is None or not getattr(cm, "is_running", False):
            return
        f = cm.get_latest_frame()
        ts = getattr(f, "timestamp", 0) if f else 0
        if ts and (time.time() - ts) > 15:
            logger.warning("camera frames stale (%.0fs) — CameraManager should reopen",
                           time.time() - ts)

    # -- robotic hand (ESP32 serial) --------------------------------------
    def _check_hand(self):
        hand = getattr(self.robot, "hand", None)
        if hand is None or not getattr(hand, "enabled", False):
            return
        port = getattr(hand, "port", "")
        if os.path.exists(port) and getattr(hand, "available", False):
            return  # healthy
        # Port vanished or link dead — find an ESP32-ish serial port and reconnect.
        cands = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        target = port if os.path.exists(port) else (cands[0] if cands else None)
        if target:
            logger.warning("hand serial unhealthy — reconnecting to %s", target)
            if hand.reconnect(target):
                logger.info("hand reconnected on %s", target)

    # -- audio output (HDMI card can renumber) ----------------------------
    def _check_audio_out(self):
        tts = self.robot.modules.get("tts") if hasattr(self.robot, "modules") else None
        if tts is None or getattr(tts, "speaking", False):
            return
        dev = getattr(tts, "alsa_device", None)
        if not (dev and hasattr(tts, "_device_opens")):
            return
        if tts._device_opens(dev):
            self._audio_fails = 0
            return
        self._audio_fails += 1
        if self._audio_fails < 2:
            return  # tolerate a transient HDMI 'busy'
        try:
            newdev = tts._resolve_output_device()
        except Exception:
            newdev = None
        if newdev and newdev != dev:
            tts.alsa_device = newdev
            self._audio_fails = 0
            logger.warning("audio output re-detected: %s -> %s", dev, newdev)
