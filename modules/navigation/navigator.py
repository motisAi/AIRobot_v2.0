"""Navigation & environment-learning hooks (future-ready).

This module is the seam for giving Gonzo wheels, sensors and the ability to
learn its way around the house. It is intentionally hardware-agnostic: all
motion and sensing go through the :class:`MicrocontrollerController` bridge, so
the moment you attach motors + an ultrasonic/LiDAR sensor to an ESP32/Pi Zero
and flip ``navigation.enabled: true`` in the config, these methods become live.

Everything is gated by ``config/config.yaml -> navigation`` and degrades to
safe no-ops (with logging) when the hardware isn't there yet. A real SLAM /
path-planner can later replace the internals of ``explore`` and ``navigate_to``
without changing how the rest of the robot calls this class.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Navigator:
    """High-level movement + mapping façade built on the microcontroller bridge."""

    def __init__(self, microcontroller=None, cfg=None, brain=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mc = microcontroller          # MicrocontrollerController (or None)
        self.brain = brain
        if cfg is None:
            from config.settings import navigation_config as cfg
        self.cfg = cfg

        self.map_dir = Path(getattr(cfg, "map_dir", "data/maps"))
        try:
            self.map_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._mapping = False
        self._explore_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Very small occupancy log — placeholder for a real map/SLAM backend.
        self._observations: List[Dict[str, Any]] = []

    # -- capability --------------------------------------------------------
    @property
    def available(self) -> bool:
        """True only when navigation is enabled AND we can actually drive."""
        return bool(getattr(self.cfg, "enabled", False)
                    and getattr(self.cfg, "has_wheels", False)
                    and self.mc is not None and getattr(self.mc, "connected", False))

    def _require(self, what: str) -> bool:
        if not getattr(self.cfg, "enabled", False):
            self.logger.info("Navigation disabled — %s ignored "
                             "(set navigation.enabled: true)", what)
            return False
        if self.mc is None or not getattr(self.mc, "connected", False):
            self.logger.info("No microcontroller/wheels connected — %s logged only", what)
            return False
        return True

    # -- motion primitives -------------------------------------------------
    def move(self, direction: str = "forward", speed: Optional[float] = None,
             duration: Optional[float] = None) -> None:
        speed = speed if speed is not None else getattr(self.cfg, "cruise_speed", 0.3)
        if not self._require(f"move({direction})"):
            return
        self.mc.move({"direction": direction, "speed": speed, "duration": duration})

    def stop(self) -> None:
        if self.mc is not None and hasattr(self.mc, "stop_all_motors"):
            self.mc.stop_all_motors()

    # -- sensing -----------------------------------------------------------
    def read_distance(self, sensor: str = "ultrasonic") -> Optional[float]:
        """Distance to the nearest obstacle in cm (None if unavailable)."""
        if self.mc is None or not getattr(self.mc, "connected", False):
            return None
        try:
            val = self.mc.read_sensor(sensor)
            return float(val) if val is not None else None
        except Exception:
            return None

    def obstacle_ahead(self) -> bool:
        """True when something is closer than the configured stop distance."""
        d = self.read_distance()
        if d is None:
            return False
        return d <= float(getattr(self.cfg, "obstacle_stop_distance_cm", 20.0))

    # -- environment learning (stubs, ready to grow) -----------------------
    def start_mapping(self) -> None:
        if not getattr(self.cfg, "mapping_enabled", False):
            self.logger.info("Mapping disabled (navigation.mapping_enabled: false)")
            return
        self._mapping = True
        self._observations.clear()
        self.logger.info("Started building a map of the environment")

    def record_observation(self, obj: Dict[str, Any]) -> None:
        """Feed a detected landmark/object into the (placeholder) map."""
        if self._mapping:
            self._observations.append(obj)

    def save_map(self, name: str = "home") -> Optional[str]:
        if not self._observations:
            self.logger.info("No map data to save yet")
            return None
        path = self.map_dir / f"{name}.json"
        try:
            path.write_text(json.dumps({"observations": self._observations}, indent=2))
            self.logger.info("Saved map with %d observations to %s",
                             len(self._observations), path)
            return str(path)
        except Exception as exc:
            self.logger.error("Failed to save map: %s", exc)
            return None

    def explore(self, seconds: float = 30.0) -> None:
        """Autonomously wander to build a map (obstacle-avoiding), if enabled."""
        if not getattr(self.cfg, "exploration_enabled", False):
            self.logger.info("Exploration disabled (navigation.exploration_enabled: false)")
            return
        if not self._require("explore()"):
            return
        if self._explore_thread and self._explore_thread.is_alive():
            return
        self._stop.clear()
        self._explore_thread = threading.Thread(
            target=self._explore_loop, args=(seconds,), daemon=True)
        self._explore_thread.start()

    def _explore_loop(self, seconds: float) -> None:
        self.logger.info("Exploring for %.0fs", seconds)
        self.start_mapping()
        ticks = int(seconds / 0.2)
        for _ in range(max(1, ticks)):
            if self._stop.is_set():
                break
            if self.obstacle_ahead():
                self.move("right", duration=0.4)   # simple avoid-and-turn
            else:
                self.move("forward", duration=0.2)
            time.sleep(0.2)
        self.stop()
        self.save_map()
        self.logger.info("Exploration finished")

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if getattr(self.cfg, "enabled", False):
            self.logger.info(
                "Navigator ready (wheels=%s, lidar=%s, ultrasonic=%s, mapping=%s)",
                self.cfg.has_wheels, self.cfg.has_lidar,
                self.cfg.has_ultrasonic, self.cfg.mapping_enabled,
            )
        return True

    def stop_module(self) -> None:
        self._stop.set()
        self.stop()

    def shutdown(self) -> None:
        self.stop_module()
