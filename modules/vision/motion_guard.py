"""Motion / presence guard for Stella.

When guard mode is ARMED, this watches the shared camera for movement (and,
best-effort, a human body) and fires an alert — WITHOUT needing to recognise a
face. A face only needs to be close/frontal/well-lit to be *recognised*, which
is useless for catching an intruder walking through a room. Motion is not:
anybody moving anywhere in view triggers it.

Face recognition is used only to SUPPRESS alerts: if Stella saw her master in
the last few seconds (``brain.last_master_time``), movement is treated as "you",
not an intruder.

Cheap by design: a MOG2 background subtractor on a downscaled grey frame runs in
a few milliseconds. A HOG people-detector is run ONLY after motion is found, to
label the alert as "a person" vs generic "movement" (best-effort, never blocks
the trigger).
"""

from __future__ import annotations

import logging
import time

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False

logger = logging.getLogger("MotionGuard")

# sensitivity name -> minimum moving-blob area (in the 320x240 analysis frame)
_SENSITIVITY_AREA = {"low": 6000, "medium": 3000, "high": 1200}


class MotionGuard:
    """Camera subscriber that alerts on movement while guard mode is armed."""

    def __init__(self, camera_manager, brain, alert_cb, *, enabled=True,
                 sensitivity="medium", cooldown=30.0, detect_person=True,
                 master_grace=20.0):
        self.cm = camera_manager
        self.brain = brain
        self.alert_cb = alert_cb            # alert_cb(reason: str, frame_image)
        self.enabled = bool(enabled) and CV2_AVAILABLE
        self.min_area = _SENSITIVITY_AREA.get(str(sensitivity).lower(), 3000)
        self.cooldown = float(cooldown)
        self.detect_person = bool(detect_person) and CV2_AVAILABLE
        self.master_grace = float(master_grace)

        self._last_alert = 0.0
        self._bg = None
        self._warmup = 0
        self._was_armed = False
        self._hog = None
        if self.detect_person:
            try:
                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception as exc:
                logger.info("HOG person detector unavailable: %s", exc)
                self._hog = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if not self.enabled:
            logger.info("Motion guard disabled (cv2=%s, enabled flag)", CV2_AVAILABLE)
            return
        self.cm.subscribe("motion_guard", self._on_frame)
        logger.info("Motion guard armed-when-guarding (min_area=%d, person=%s, cooldown=%ss)",
                    self.min_area, bool(self._hog), self.cooldown)

    def stop(self):
        try:
            self.cm.unsubscribe("motion_guard")
        except Exception:
            pass

    # -- per-frame ---------------------------------------------------------
    def _on_frame(self, frame):
        armed = bool(getattr(self.brain, "guard_mode", False))
        if not armed:
            if self._was_armed:
                self._was_armed = False
                self._bg = None            # forget the scene; rebuild on re-arm
            return

        if not self._was_armed:
            # Just armed: build a fresh background model and warm up so the
            # first frames (empty model = everything looks like motion) don't
            # fire a false alert.
            self._was_armed = True
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=25, detectShadows=False)
            self._warmup = 0

        img = getattr(frame, "image", None)
        if img is None or self._bg is None:
            return

        try:
            small = cv2.resize(img, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            mask = self._bg.apply(gray)
        except Exception as exc:
            logger.debug("motion process error: %s", exc)
            return

        self._warmup += 1
        if self._warmup < 12:
            return  # let the background settle after arming

        # Biggest moving blob.
        _, th = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        found = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = found[0] if len(found) == 2 else found[1]  # cv2 v4 vs v3
        area = max((cv2.contourArea(c) for c in cnts), default=0)
        if area < self.min_area:
            return

        now = time.time()
        # You're home / just seen -> not an intruder.
        if now - float(getattr(self.brain, "last_master_time", 0) or 0) < self.master_grace:
            return
        # A conversation is in progress (someone is present, talking to Stella) —
        # don't fire motion alerts about the person she's chatting with.
        if getattr(self.brain, "_suppress_greetings", False):
            return
        if now - self._last_alert < self.cooldown:
            return

        # Best-effort: is the moving thing person-shaped?
        reason = "movement"
        if self._hog is not None:
            try:
                rects, _ = self._hog.detectMultiScale(
                    small, winStride=(8, 8), padding=(8, 8), scale=1.05)
                if len(rects) > 0:
                    reason = "a person"
            except Exception:
                pass

        self._last_alert = now
        logger.warning("GUARD motion trigger (%s, area=%d)", reason, int(area))
        try:
            self.alert_cb(reason, img)
        except Exception as exc:
            logger.error("alert callback failed: %s", exc)
