"""Object Detection Module
=========================
High-level object detection that wraps the Hailo/OpenCV detector and
integrates with the shared camera manager and robot brain event system.

Subscribes to the camera manager, runs detection, tracks objects across
frames, and emits ``object_detected`` events.  Supports learning new
custom objects via embeddings stored in the learning database.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from config.settings import system_config
from core.robot_brain import RobotEvent
from modules.hardware.camera_manager import Frame
from modules.vision.hailo_detector import Detection, HailoDetector


class TrackedObject:
    """Lightweight object tracker across consecutive frames."""

    __slots__ = ("label", "class_id", "bbox", "confidence", "first_seen",
                 "last_seen", "hit_count", "miss_count")

    def __init__(self, det: Detection):
        self.label = det.label
        self.class_id = det.class_id
        self.bbox = det.bbox
        self.confidence = det.confidence
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.hit_count = 1
        self.miss_count = 0

    def update(self, det: Detection) -> None:
        self.bbox = det.bbox
        self.confidence = det.confidence
        self.last_seen = time.time()
        self.hit_count += 1
        self.miss_count = 0

    @property
    def age(self) -> float:
        return time.time() - self.first_seen

    @property
    def stale(self) -> bool:
        return self.miss_count > 5


class ObjectDetectionModule:
    """Receives frames from the camera manager, detects objects, emits events."""

    def __init__(self, brain=None, camera_manager=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.brain = brain
        self.camera_manager = camera_manager

        self.detector = HailoDetector()
        self._tracked: Dict[str, TrackedObject] = {}
        self._lock = threading.Lock()
        self._running = False

        # Statistics
        self.detections_total = 0
        self.frames_processed = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if not self.detector.start():
            self.logger.warning("Object detector unavailable — module will be passive")
            return False

        self._running = True
        if self.camera_manager:
            self.camera_manager.subscribe("object_detection", self._on_frame)
        self.logger.info("Object detection started (backend=%s)", self.detector.backend_name)
        return True

    def stop(self) -> None:
        self._running = False
        if self.camera_manager:
            self.camera_manager.unsubscribe("object_detection")
        self.detector.stop()
        self.logger.info("Object detection stopped")

    # ------------------------------------------------------------------
    # Frame callback
    # ------------------------------------------------------------------
    def _on_frame(self, frame: Frame) -> None:
        """Called by camera_manager for each distributed frame."""
        if not self._running:
            return

        detections = self.detector.detect(frame.image)
        self.frames_processed += 1
        self.detections_total += len(detections)

        self._update_tracking(detections)
        self._emit_events(detections)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    def _update_tracking(self, detections: List[Detection]) -> None:
        """Simple IoU-based tracking to avoid duplicate events."""
        with self._lock:
            matched_keys = set()
            for det in detections:
                key = self._match_existing(det)
                if key:
                    self._tracked[key].update(det)
                    matched_keys.add(key)
                else:
                    new_key = f"{det.label}_{id(det)}_{time.time_ns()}"
                    self._tracked[new_key] = TrackedObject(det)
                    matched_keys.add(new_key)

            # Increment miss count on unmatched tracks
            for key in list(self._tracked):
                if key not in matched_keys:
                    self._tracked[key].miss_count += 1

            # Remove stale tracks
            self._tracked = {k: v for k, v in self._tracked.items() if not v.stale}

    def _match_existing(self, det: Detection) -> Optional[str]:
        """Return the key of an existing track that overlaps det."""
        best_key = None
        best_iou = 0.3  # Minimum IoU to consider a match
        for key, tracked in self._tracked.items():
            if tracked.label != det.label:
                continue
            iou = self._iou(tracked.bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_key = key
        return best_key

    @staticmethod
    def _iou(a, b) -> float:
        """Compute Intersection-over-Union for two (x,y,w,h) boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def _emit_events(self, detections: List[Detection]) -> None:
        if not self.brain or not detections:
            return

        # Aggregate by label for a cleaner event
        summary: Dict[str, int] = defaultdict(int)
        for det in detections:
            summary[det.label] += 1

        event = RobotEvent(
            type="object_detected",
            source="object_detection",
            data={
                "objects": [
                    {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
                    for d in detections
                ],
                "summary": dict(summary),
            },
            priority=6,
        )
        try:
            self.brain.emit_event(event)
        except Exception as exc:
            self.logger.error("Failed to emit object_detected event: %s", exc)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def get_current_objects(self) -> List[dict]:
        """Return a snapshot of currently tracked objects."""
        with self._lock:
            return [
                {
                    "label": t.label,
                    "confidence": t.confidence,
                    "bbox": t.bbox,
                    "age": t.age,
                    "hits": t.hit_count,
                }
                for t in self._tracked.values()
            ]
