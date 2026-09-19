"""Tuya device control (LSPA8 smart plug, etc.) via tinytuya + local keys.

Loads devices from tinytuya's devices.json (id / ip / key / version) produced by
`python -m tinytuya wizard`. Control is LOCAL (LAN) — no cloud needed at command
time once the local key is known. Stella reaches the device on whatever network
they share (home WiFi now; RobotNet if moved there later).
"""

from __future__ import annotations

import json
import logging
import os
import threading

try:
    import tinytuya
    TUYA_OK = True
except Exception:  # pragma: no cover
    tinytuya = None
    TUYA_OK = False

logger = logging.getLogger("Tuya")


class TuyaDevices:
    def __init__(self, devices_file: str = "devices.json", switch_dp: int = 1):
        self.switch_dp = switch_dp
        self.devices: dict[str, dict] = {}   # name -> {id, ip, key, version}
        self.enabled = TUYA_OK
        self._lock = threading.Lock()
        if TUYA_OK:
            self._load(devices_file)
        else:
            logger.info("tinytuya not installed — Tuya control disabled")

    def _load(self, path: str):
        try:
            if not os.path.exists(path):
                logger.info("no %s yet (run: python -m tinytuya wizard)", path)
                return
            for d in json.load(open(path)):
                if not d.get("id") or not d.get("key"):
                    continue
                name = d.get("name") or d["id"]
                self.devices[name] = {
                    "id": d["id"], "ip": d.get("ip"),
                    "key": d["key"], "version": float(d.get("version", 3.3) or 3.3),
                }
            logger.info("Tuya: loaded %d device(s): %s",
                        len(self.devices), list(self.devices.keys()))
        except Exception as exc:
            logger.warning("Tuya load failed: %s", exc)

    # -- matching / API ----------------------------------------------------
    def _match(self, name: str):
        n = (name or "").lower().strip()
        if not n:
            return None
        for k in self.devices:                       # substring either way
            kl = k.lower()
            if kl == n or n in kl or kl in n:
                return k
        nwords = set(n.split())                       # word overlap (plug/socket)
        for k in self.devices:
            if nwords & set(k.lower().replace("smart", "").split()):
                return k
        # Single device: answer to any common on/off name for it (light, lamp,
        # socket, switch, relay1, etc.) — the user's one switchable thing.
        if len(self.devices) == 1 and any(w in n for w in (
                "light", "lamp", "plug", "socket", "switch", "power",
                "outlet", "relay", "device")):
            return next(iter(self.devices))
        return None

    def known(self, name: str) -> bool:
        return self._match(name) is not None

    def list_devices(self):
        return list(self.devices.keys())

    def _dev(self, info: dict):
        o = tinytuya.OutletDevice(info["id"], info["ip"], info["key"])
        o.set_version(info["version"])
        o.set_socketTimeout(5)
        return o

    def set(self, name: str, on: bool) -> bool:
        if not self.enabled:
            return False
        key = self._match(name)
        if not key:
            return False
        info = self.devices[key]
        try:
            with self._lock:
                res = self._dev(info).set_status(bool(on), self.switch_dp)
            err = isinstance(res, dict) and res.get("Error")
            logger.info("Tuya '%s' -> %s%s", key, "ON" if on else "OFF",
                        f" ERROR: {res}" if err else "")
            return not err
        except Exception as exc:
            logger.warning("Tuya set failed for '%s': %s", key, exc)
            return False

    def get_state(self, name: str) -> str:
        key = self._match(name)
        if not key:
            return "unknown"
        try:
            with self._lock:
                s = self._dev(self.devices[key]).status()
            v = (s or {}).get("dps", {}).get(str(self.switch_dp))
            return "ON" if v else ("OFF" if v is not None else "unknown")
        except Exception:
            return "unknown"
