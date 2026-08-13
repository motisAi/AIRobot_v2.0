"""Vision-Language model (Moondream cloud) — lets Stella describe what the
camera sees: "what am I holding?", "what colour is this?", "read this label".

Free-tier cloud call (key in .env: MOONDREAM_API_KEY). The Pi stays light — no
local model. Frames come from the shared CameraManager (no camera conflict).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("VLM")


class VLM:
    def __init__(self, api_key: str = "", camera_manager=None):
        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY", "")
        self.camera_manager = camera_manager
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_model(self):
        if self._model is None:
            import moondream
            self._model = moondream.vl(api_key=self.api_key)
        return self._model

    def _grab_frame(self):
        """Latest BGR frame from the shared camera (or None)."""
        cm = self.camera_manager
        if cm is None:
            return None
        try:
            for attr in ("get_latest_frame", "get_frame"):
                if hasattr(cm, attr):
                    f = getattr(cm, attr)()
                    img = getattr(f, "image", f)  # Frame wrapper or raw ndarray
                    if img is not None and getattr(img, "size", 0):
                        return img
        except Exception as exc:
            logger.warning("frame grab failed: %s", exc)
        return None

    def look(self, question: Optional[str] = None) -> Optional[str]:
        """Answer a question about what the camera sees, or caption the scene.
        Returns the answer text, or None if it couldn't see/answer."""
        if not self.available:
            return None
        frame = self._grab_frame()
        if frame is None:
            return None
        try:
            import cv2
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            model = self._ensure_model()
            if question:
                out = model.query(img, question)
                return (out.get("answer") if isinstance(out, dict) else str(out)) or None
            out = model.caption(img)
            return (out.get("caption") if isinstance(out, dict) else str(out)) or None
        except Exception as exc:
            logger.warning("vision query failed: %s", exc)
            return None
