"""Push notifications to the master's phone via a Telegram bot (free).

Sends text and photos (e.g. a guard-mode snapshot of an unrecognized person).
Credentials come from .env: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("Notify")


class Notifier:
    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token or os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = str(chat_id or os.getenv("TELEGRAM_CHAT_ID", ""))

    @property
    def available(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, text: str) -> bool:
        if not self.available:
            logger.info("Telegram not configured (need TELEGRAM_TOKEN + CHAT_ID)")
            return False
        try:
            import requests
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data={"chat_id": self.chat_id, "text": text}, timeout=10)
            return r.ok
        except Exception as exc:
            logger.warning("telegram message failed: %s", exc)
            return False

    def send_photo(self, image_bgr, caption: str = "") -> bool:
        """Send an OpenCV BGR frame as a JPEG photo with a caption."""
        if not self.available:
            return False
        try:
            import requests, cv2
            ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return False
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": ("snapshot.jpg", buf.tobytes(), "image/jpeg")},
                timeout=20)
            return r.ok
        except Exception as exc:
            logger.warning("telegram photo failed: %s", exc)
            return False
