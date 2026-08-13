"""Simple reminders/timers for Stella.

Stella can set a reminder ("remind me in 10 minutes to check the oven"); a
background thread announces it out loud (and via the speak callback) when due.
Reminders persist to a small JSON file so they survive a restart.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger("Reminders")


class ReminderManager:
    def __init__(self, speak_cb: Optional[Callable[[str], None]] = None,
                 store_path: str = "data/reminders.json"):
        self.speak_cb = speak_cb
        self.store_path = Path(store_path)
        self._items: List[dict] = []   # {text, due_epoch, done}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self):
        try:
            if self.store_path.exists():
                self._items = json.loads(self.store_path.read_text()) or []
        except Exception:
            self._items = []

    def _save(self):
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps(self._items))
        except Exception as exc:
            logger.debug("reminder save failed: %s", exc)

    # -- API ---------------------------------------------------------------
    def add(self, text: str, minutes: float, now_epoch: Optional[float] = None) -> str:
        """Schedule a reminder ``minutes`` from now. now_epoch lets callers pass
        the time (this process avoids Date.now-style calls elsewhere, but here a
        real timer is required)."""
        base = now_epoch if now_epoch is not None else time.time()
        due = base + max(1.0, float(minutes)) * 60.0
        clean = text.strip() or "reminder"
        with self._lock:
            # Ignore a near-duplicate (same text due within ~2 min of an existing
            # one) — stops repeated "yes"/"ok" turns creating identical reminders.
            for i in self._items:
                if (not i["done"] and i["text"].lower() == clean.lower()
                        and abs(i["due"] - due) < 120):
                    logger.info("skipped duplicate reminder: %s", clean)
                    mins0 = max(1, int(round((i["due"] - base) / 60.0)))
                    return (f"You already have that reminder set for about "
                            f"{mins0} minute{'s' if mins0 != 1 else ''} from now.")
            self._items.append({"text": clean, "due": due, "done": False})
            self._save()
        mins = max(1, int(round(minutes)))
        return f"Okay, I'll remind you in {mins} minute{'s' if mins != 1 else ''}."

    def pending(self) -> List[dict]:
        with self._lock:
            return [dict(i) for i in self._items if not i["done"]]

    def clear(self):
        with self._lock:
            self._items = []
            self._save()

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reminders")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            now = time.time()
            due_now = []
            with self._lock:
                for i in self._items:
                    if not i["done"] and i["due"] <= now:
                        i["done"] = True
                        due_now.append(i["text"])
                if due_now:
                    self._items = [i for i in self._items if not i["done"]]
                    self._save()
            for text in due_now:
                msg = f"Reminder: {text}"
                logger.info(msg)
                if self.speak_cb:
                    try:
                        self.speak_cb(msg)
                    except Exception:
                        pass
            self._stop.wait(5.0)
