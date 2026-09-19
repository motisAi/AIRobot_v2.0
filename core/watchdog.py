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
            for check in (self._check_camera, self._check_hand, self._check_audio_out, self._check_stuck_conversation):
                try:
                    check()
                except Exception as exc:
                    logger.debug("%s error: %s", check.__name__, exc)
            self._stop.wait(self.interval)

    # -- stuck conversation (mic hang) ------------------------------------
    def _check_stuck_conversation(self):
        """If a conversation is active but Stella has not spoken for 120s, the
        mic capture has hung (PortAudio blocking read on a stalled USB mic).
        Restart the service to recover — safe, no cross-thread audio ops."""
        r = self.robot
        if not getattr(r, "_conv_active", False):
            return
        if getattr(r, "_thinking", False):
            return  # the LLM is working (slow offline think) — not stuck
        # monotonic: the offline Pi clock jumps hours when NTP syncs; wall-clock
        # deltas would then fire a false 'stuck' restart mid-conversation.
        last = float(getattr(r, "_conv_activity", 0.0) or 0.0)
        if not (last and (time.monotonic() - last) > 75.0):
            self._soft_recover_at = 0.0   # not stuck (or recovered) — reset escalation
            return
        conv = getattr(r, "conversation", None)
        soft_at = getattr(self, "_soft_recover_at", 0.0)
        # Stage 1: end the conversation session (thread-safe) and reopen the mic,
        # instead of a ~30s full service restart that kills vision/guard/Telegram too.
        if not soft_at:
            logger.warning("Conversation stuck (no speech %.0fs) — soft-stopping the session",
                           time.monotonic() - last)
            self._soft_recover_at = time.monotonic()
            try:
                if conv is not None:
                    conv.stop()
            except Exception as exc:
                logger.error("watchdog soft stop failed: %s", exc)
            return
        # Stage 2: only if it is STILL stuck ~25s after the soft stop, restart.
        if time.monotonic() - soft_at > 25.0:
            self._soft_recover_at = 0.0
            if getattr(r, "_conv_active", False):
                logger.error("Conversation still stuck after soft stop — restarting service")
                import subprocess
                try:
                    subprocess.Popen(["sudo", "-n", "systemctl", "restart", "airobot"])
                except Exception as exc:
                    logger.error("watchdog restart failed: %s", exc)

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
        # Back off when no output exists (no HDMI at work): don't re-probe every 20s.
        if time.time() < getattr(self, "_audio_next_check", 0.0):
            return
        if tts._device_opens(dev):
            self._audio_fails = 0
            self._audio_backoff = 20.0
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
            self._audio_backoff = 20.0
            logger.warning("audio output re-detected: %s -> %s", dev, newdev)
        else:
            b = getattr(self, "_audio_backoff", 20.0)
            self._audio_next_check = time.time() + b
            self._audio_backoff = min(b * 2, 300.0)
