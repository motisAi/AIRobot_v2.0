"""Shared Camera Manager
======================
Centralized camera access so multiple modules (face recognition, object
detection, environment mapping) share a single video stream without
conflicts.  Subscribers register a callback and receive frames at the
configured rate.

Design goals:
 - One ``cv2.VideoCapture`` instance, one capture thread.
 - Frame distribution via subscriber callbacks on a dedicated thread pool.
 - Thread-safe start / stop / subscribe / unsubscribe.
 - Graceful fallback when no camera is connected.
"""

from __future__ import annotations

import cv2
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

from config.settings import hardware_config, system_config

FrameCallback = Callable[["Frame"], None]


class Frame:
    """Immutable snapshot of a single camera frame with metadata."""

    __slots__ = ("image", "index", "timestamp", "resolution")

    def __init__(self, image, index: int, timestamp: float, resolution: Tuple[int, int]):
        self.image = image          # numpy ndarray (BGR)
        self.index = index          # monotonic frame counter
        self.timestamp = timestamp  # time.time() when captured
        self.resolution = resolution


class CameraManager:
    """Owns the physical camera and distributes frames to subscribers."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.camera_index: int = hardware_config.camera_index
        self.resolution: Tuple[int, int] = hardware_config.camera_resolution
        self.fps: int = hardware_config.camera_fps
        self.buffer_size: int = hardware_config.camera_buffer_size
        self.frame_skip: int = system_config.frame_skip

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Subscribers: name -> callback
        self._subscribers: Dict[str, FrameCallback] = OrderedDict()
        self._sub_lock = threading.Lock()

        # Latest frame (for one-shot reads)
        self._latest_frame: Optional[Frame] = None
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    # Video nodes that are NOT plain cv2-readable cameras (Pi CSI pipeline +
    # ISP + hardware codec). We must skip these when probing.
    _SKIP_NODE_HINTS = ("rp1-cfe", "pispbe", "rpivid", "codec", "isp")

    def _candidate_indices(self):
        """Order camera indices to try: real USB/UVC webcams (found by name in
        /sys) first, then the configured index, then a numeric fallback. The Pi
        CSI / ISP nodes (rp1-cfe / pispbe / ...) are EXCLUDED entirely — opening
        them spams "csi2_chN node link is not enabled" to the console and they are
        not cv2-readable anyway. Detecting by name survives node renumbering."""
        import os, re
        usb, skip = [], set()
        try:
            for d in sorted(os.listdir("/sys/class/video4linux")):
                m = re.match(r"video(\d+)$", d)
                if not m:
                    continue
                idx = int(m.group(1))
                try:
                    name = open("/sys/class/video4linux/%s/name" % d).read().strip().lower()
                except Exception:
                    name = ""
                if any(h in name for h in self._SKIP_NODE_HINTS):
                    skip.add(idx)      # CSI/ISP node -> never open it
                else:
                    usb.append(idx)    # real capture device (USB webcam)
        except Exception:
            pass
        order = usb + [self.camera_index] + list(range(0, 16))
        seen, out = set(), []
        for i in order:
            if i not in seen and i not in skip:
                seen.add(i); out.append(i)
        return out

    def _open_index(self, index: int):
        """Open one index with MJPG + config resolution, then warm-read for up to
        ~2.5s (USB cams routinely fail the first few reads). Returns an opened,
        frame-producing VideoCapture or None."""
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self._apply_camera_controls(cap)
        for _ in range(16):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.15)
        cap.release()
        return None

    def start(self) -> bool:
        """Open the camera and start the capture thread.

        Returns True if the camera opened successfully.
        """
        with self._lock:
            if self._running:
                return True

            cap = None
            for idx in self._candidate_indices():
                cap = self._open_index(idx)
                if cap is not None:
                    if idx != self.camera_index:
                        self.logger.info("Camera auto-detected at index %d "
                                         "(configured was %d)", idx, self.camera_index)
                    self.camera_index = idx
                    break
            if cap is None:
                self.logger.error("No readable camera found (tried CSI/USB nodes). "
                                  "Is the USB webcam plugged in?")
                return False

            self._cap = cap
            self._running = True

            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self.logger.info("Camera started (index=%d, res=%s, fps=%d)",
                             self.camera_index, self.resolution, self.fps)
            return True

    def _apply_camera_controls(self, cap) -> None:
        """Apply image controls (brightness/gain/exposure/etc.) from config.
        Only sets values that are configured (non-None) so we don't override
        good camera defaults."""
        try:
            hc = hardware_config
            mapping = [
                (cv2.CAP_PROP_AUTO_EXPOSURE, getattr(hc, 'camera_auto_exposure', None)),
                (cv2.CAP_PROP_BRIGHTNESS, getattr(hc, 'camera_brightness', None)),
                (cv2.CAP_PROP_CONTRAST, getattr(hc, 'camera_contrast', None)),
                (cv2.CAP_PROP_GAIN, getattr(hc, 'camera_gain', None)),
                (cv2.CAP_PROP_GAMMA, getattr(hc, 'camera_gamma', None)),
            ]
            applied = []
            for prop, val in mapping:
                if val is not None:
                    cap.set(prop, float(val))
                    applied.append((prop, val))
            # Auto white-balance ON so colours are reported accurately (helps the
            # vision model), and auto-exposure adapts to the room light.
            try:
                cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
                if getattr(hc, 'camera_auto_exposure', None) is None:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
            except Exception:
                pass
            if applied:
                self.logger.info("Applied %d camera control(s) from config", len(applied))
        except Exception as exc:
            self.logger.warning("Could not apply camera controls: %s", exc)

    def stop(self) -> None:
        """Release the camera and stop the capture thread."""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
        self.logger.info("Camera stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def subscribe(self, name: str, callback: FrameCallback) -> None:
        """Register *callback* to receive every Nth frame (governed by frame_skip)."""
        with self._sub_lock:
            self._subscribers[name] = callback
        self.logger.info("Subscriber added: %s (total=%d)", name, len(self._subscribers))

    def unsubscribe(self, name: str) -> None:
        with self._sub_lock:
            self._subscribers.pop(name, None)
        self.logger.info("Subscriber removed: %s", name)

    def get_latest_frame(self) -> Optional[Frame]:
        """Return the most recent frame without subscribing."""
        return self._latest_frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """Read frames and distribute to subscribers."""
        while self._running:
            try:
                if not self._cap or not self._cap.isOpened():
                    self.logger.warning("Camera disconnected — attempting reopen")
                    time.sleep(2.0)
                    self._reopen()
                    continue

                ok, image = self._cap.read()
                if not ok:
                    self.logger.warning("Frame read failed")
                    time.sleep(0.05)
                    continue

                self._frame_index += 1
                frame = Frame(
                    image=image,
                    index=self._frame_index,
                    timestamp=time.time(),
                    resolution=(image.shape[1], image.shape[0]),
                )
                self._latest_frame = frame

                # Distribute to subscribers (skip frames for performance)
                if self._frame_index % self.frame_skip == 0:
                    with self._sub_lock:
                        subs = list(self._subscribers.items())
                    for name, cb in subs:
                        try:
                            cb(frame)
                        except Exception as exc:
                            self.logger.error("Subscriber '%s' error: %s", name, exc)

            except Exception as exc:
                self.logger.error("Capture loop error: %s", exc)
                time.sleep(0.1)

    def _reopen(self) -> None:
        """Try to reopen the camera after a disconnect (re-probing indices, since
        a USB re-enumeration may have moved the node)."""
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
            for idx in self._candidate_indices():
                cap = self._open_index(idx)
                if cap is not None:
                    self.camera_index = idx
                    self._cap = cap
                    self.logger.info("Camera reopened successfully (index=%d)", idx)
                    return
            self.logger.warning("Camera reopen failed — will retry")
