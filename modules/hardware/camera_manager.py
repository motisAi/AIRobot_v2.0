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
    def start(self) -> bool:
        """Open the camera and start the capture thread.

        Returns True if the camera opened successfully.
        """
        with self._lock:
            if self._running:
                return True

            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.logger.error("Failed to open camera index %d", self.camera_index)
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self._apply_camera_controls(cap)

            # Verify with a test read
            ok, _ = cap.read()
            if not ok:
                self.logger.error("Camera opened but test read failed")
                cap.release()
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
        """Try to reopen the camera after a disconnect."""
        with self._lock:
            if self._cap:
                self._cap.release()
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self._cap = cap
                self.logger.info("Camera reopened successfully")
            else:
                self._cap = None
                self.logger.warning("Camera reopen failed — will retry")
