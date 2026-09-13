"""Vision-Language model — lets Stella describe what the camera sees:
"what am I holding?", "what colour is this?", "read this label".

Primary: Moondream cloud (MOONDREAM_API_KEY) — fast, free tier, sub-second.
Fallback: NVIDIA NIM meta/llama-3.2-11b-vision-instruct (NVIDIA_API_KEY) — a free,
strong online VLM on an independent network/provider path. Whichever is available
answers; if Moondream fails, NIM is tried. The Pi stays light (no local model).
Frames come from the shared CameraManager (no camera conflict).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("VLM")

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "meta/llama-3.2-11b-vision-instruct"


class VLM:
    def __init__(self, api_key: str = "", camera_manager=None):
        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY", "")
        self.nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        self.camera_manager = camera_manager
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.api_key or self.nvidia_key)

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
        Returns the answer text, or None if it couldn't see/answer.
        Tries Moondream first, then NVIDIA NIM."""
        if not self.available:
            return None
        frame = self._grab_frame()
        if frame is None:
            return None
        # Primary: Moondream
        if self.api_key:
            try:
                import cv2
                from PIL import Image
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                model = self._ensure_model()
                if question:
                    out = model.query(img, question)
                    ans = (out.get("answer") if isinstance(out, dict) else str(out)) or None
                else:
                    out = model.caption(img)
                    ans = (out.get("caption") if isinstance(out, dict) else str(out)) or None
                if ans:
                    return ans
            except Exception as exc:
                logger.warning("Moondream vision failed, trying NIM: %s", exc)
        # Fallback: NVIDIA NIM vision
        if self.nvidia_key:
            return self._look_nim(frame, question)
        return None

    def _look_nim(self, frame, question: Optional[str]) -> Optional[str]:
        """NVIDIA NIM VLM (llama-3.2-11b-vision). Frame is a BGR ndarray."""
        try:
            import cv2, base64, json, urllib.request
            # downscale to <=1024px on the long side to cut latency + data
            h, w = frame.shape[:2]
            scale = 1024.0 / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            b64 = base64.b64encode(buf).decode()
            prompt = question or "Describe what you see in one short sentence."
            body = {
                "model": NIM_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                ]}],
                "max_tokens": 128, "temperature": 0.2,
            }
            req = urllib.request.Request(
                NIM_URL, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + self.nvidia_key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read().decode())
            txt = d["choices"][0]["message"]["content"].strip()
            return txt or None
        except Exception as exc:
            logger.warning("NIM vision failed: %s", exc)
            return None
