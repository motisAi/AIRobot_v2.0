"""Sensibo AC control via the Sensibo cloud REST API.

Needs internet + an API key (SENSIBO_API_KEY in .env, from home.sensibo.com ->
Me -> API). Controls power, target temperature, and mode (cool/heat/fan/dry/
auto). Supports multiple units, selected by room name; defaults to the first.

Cloud-based: works from anywhere WITH internet (even remotely), but not offline.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("Sensibo")
BASE = "https://home.sensibo.com/api/v2"
_SETTABLE = ("on", "mode", "targetTemperature", "fanLevel", "swing")


class Sensibo:
    def __init__(self, api_key: str | None = None, default_room: str | None = None,
                 timeout: float = 8.0):
        self.key = api_key or os.getenv("SENSIBO_API_KEY", "")
        self.timeout = timeout
        self.pods: dict[str, str] = {}      # room name (lower) -> pod id
        self.default_pod = None
        self._cache: dict[str, dict] = {}   # pod -> last acState (avoids extra calls)
        self.enabled = bool(self.key)
        if self.enabled:
            self._load_pods(default_room)

    # -- http --------------------------------------------------------------
    def _get(self, path: str, params: dict | None = None):
        p = dict(params or {}); p["apiKey"] = self.key
        url = f"{BASE}{path}?{urllib.parse.urlencode(p)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _post(self, path: str, body: dict):
        url = f"{BASE}{path}?apiKey={urllib.parse.quote(self.key)}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            raise RuntimeError(f"Sensibo HTTP {e.code}: {body}") from e

    # -- pods --------------------------------------------------------------
    def _load_pods(self, default_room):
        try:
            d = self._get("/users/me/pods", {"fields": "id,room"})
            for p in d.get("result", []):
                room = (p.get("room") or {}).get("name") or p["id"]
                self.pods[room.lower()] = p["id"]
            vals = list(self.pods.values())
            if vals:
                self.default_pod = vals[0]
                if default_room:
                    for k, v in self.pods.items():
                        if default_room.lower() in k:
                            self.default_pod = v
            logger.info("Sensibo: %d unit(s): %s", len(self.pods),
                        list(self.pods.keys()))
        except Exception as exc:
            logger.warning("Sensibo load failed: %s", exc)

    def _resolve(self, room: str | None):
        if room:
            r = room.lower()
            for k, v in self.pods.items():
                if r in k or k in r:
                    return v
        return self.default_pod

    def _state(self, pod: str) -> dict:
        d = self._get(f"/pods/{pod}/acStates", {"limit": "1"})
        res = d.get("result", [])
        return dict(res[0]["acState"]) if res else {}

    def _apply(self, pod: str, changes: dict) -> bool:
        # Fetch the AC's REAL current state (filtered to settable fields) so we never
        # echo a stale value the unit rejects (Sensibo 422 on a mode change). Cache is
        # only used for state() reads, not as the POST base — it can drift if the AC
        # was touched by its own remote.
        cur = {k: v for k, v in self._state(pod).items() if k in _SETTABLE}
        cur.update(changes)
        # targetTemperature is invalid in fan/dry mode — drop it there.
        if cur.get("mode") in ("fan", "dry"):
            cur.pop("targetTemperature", None)
        r = self._post(f"/pods/{pod}/acStates", {"acState": cur})
        ok = r.get("status") == "success"
        if ok:
            self._cache[pod] = dict(cur)   # remember for next command
        return ok

    # -- public API --------------------------------------------------------
    def set_power(self, on: bool, room: str | None = None) -> bool:
        pod = self._resolve(room)
        return bool(pod) and self._safe(pod, {"on": bool(on)})

    def set_temp(self, temp: int, room: str | None = None) -> bool:
        pod = self._resolve(room)
        return bool(pod) and self._safe(pod, {"on": True, "targetTemperature": int(temp)})

    def set_mode(self, mode: str, room: str | None = None) -> bool:
        pod = self._resolve(room)
        return bool(pod) and self._safe(pod, {"on": True, "mode": mode})

    def set(self, power=None, temperature=None, mode=None, room: str | None = None) -> bool:
        """Combined change in ONE API call (rate-limit friendly)."""
        pod = self._resolve(room)
        if not pod:
            return False
        changes: dict = {}
        if mode is not None:
            changes["mode"] = mode
            changes["on"] = True
        if temperature is not None:
            changes["targetTemperature"] = int(temperature)
            changes["on"] = True
        if power is not None:
            changes["on"] = bool(power)
        if not changes:
            return False
        return self._safe(pod, changes)

    def _safe(self, pod, changes):
        try:
            return self._apply(pod, changes)
        except Exception as exc:
            logger.warning("Sensibo set failed: %s", exc)
            return False

    def state(self, room: str | None = None) -> dict:
        pod = self._resolve(room)
        if not pod:
            return {}
        try:
            return self._state(pod)
        except Exception:
            return {}

    def list_units(self):
        return list(self.pods.keys())
