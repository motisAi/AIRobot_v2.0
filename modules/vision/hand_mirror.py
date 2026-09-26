"""Hand mirror — Stella copies your hand in real time.

Runs MediaPipe Hands in its OWN worker thread (NOT the camera callback thread) so
it never blocks face‑recognition, the guard, or the wake word. Uses the LITE
model on a downscaled frame at a few FPS to keep CPU sane, and AUTO‑STOPS when it
stops seeing a hand (so a forgotten "copy my hand" can't peg the CPU and starve
the microphone).
"""

from __future__ import annotations

import logging
import math
import threading
import time

try:
    import cv2
    import mediapipe as mp
    DEPS_OK = True
except Exception:  # pragma: no cover
    cv2 = None
    mp = None
    DEPS_OK = False

logger = logging.getLogger("HandMirror")

WRIST = 0
TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP, THUMB_IP, INDEX_MCP = 4, 3, 5


class HandMirror:
    def __init__(self, camera_manager, hand, fps: float = 3.0,
                 idle_stop: float = 15.0, max_duration: float = 180.0):
        self.cm = camera_manager
        self.hand = hand
        self._interval = 1.0 / max(1.0, fps)
        self.idle_stop = idle_stop          # stop after this long with no hand seen
        self.max_duration = max_duration    # hard cap, seconds
        self._active = threading.Event()
        self._thread = None
        self._hands = None
        self._last_mask = None
        self._last_hand = 0.0
        self._started = 0.0

    @property
    def available(self) -> bool:
        return DEPS_OK and self.hand is not None

    @property
    def active(self) -> bool:
        return self._active.is_set()

    def start(self):
        if not DEPS_OK:
            logger.info("HandMirror disabled (mediapipe/cv2 not available)")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hand_mirror")
        self._thread.start()
        logger.info("HandMirror ready (say 'copy my hand' to start)")

    def set_active(self, on: bool) -> bool:
        if not self.available:
            return False
        if on:
            if self._hands is None:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    model_complexity=0,               # lite model = far less CPU
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._last_mask = None
            self._last_hand = time.time()
            self._started = time.time()
            self._active.set()
        else:
            self._active.clear()
        logger.info("mirror mode %s", "ON" if on else "OFF")
        return True

    # -- worker thread ----------------------------------------------------
    def _loop(self):
        while True:
            if not self._active.is_set():
                time.sleep(0.2)
                continue
            t0 = time.time()
            if (t0 - self._last_hand > self.idle_stop) or (t0 - self._started > self.max_duration):
                self._active.clear()
                logger.info("mirror auto-stopped (no hand / max time)")
                continue
            self._process_once()
            time.sleep(max(0.0, self._interval - (time.time() - t0)))

    def _process_once(self):
        if not getattr(self.hand, "available", False):
            return
        f = self.cm.get_latest_frame() if self.cm else None
        img = getattr(f, "image", None)
        if img is None:
            return
        try:
            small = cv2.resize(img, (320, 240))        # downscale = much faster
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = self._hands.process(rgb)
        except Exception as exc:
            logger.debug("mediapipe error: %s", exc)
            return
        if not res.multi_hand_landmarks:
            return
        self._last_hand = time.time()
        mask = self._mask_from_landmarks(res.multi_hand_landmarks[0].landmark)
        if mask and mask != self._last_mask:
            self._last_mask = mask
            try:
                self.hand.send("set " + mask)
            except Exception as exc:
                logger.debug("send failed: %s", exc)

    @staticmethod
    def _mask_from_landmarks(lm) -> str:
        """5-char mask 'pinky ring middle index thumb', 1=extended."""
        def dist(a, b):
            return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

        ext = {}
        for name in ("index", "middle", "ring", "pinky"):
            ext[name] = 1 if dist(TIPS[name], WRIST) > dist(PIPS[name], WRIST) else 0
        # thumb: extended if the tip sits farther from the index knuckle than its IP.
        ext["thumb"] = 1 if dist(THUMB_TIP, INDEX_MCP) > dist(THUMB_IP, INDEX_MCP) else 0
        return f"{ext['pinky']}{ext['ring']}{ext['middle']}{ext['index']}{ext['thumb']}"

    def stop(self):
        self._active.clear()
