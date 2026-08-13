"""WiFi assistant helpers.

The Pi is WiFi-only and wpa_cli needs root, so the privileged bits go through a
small, sudoers-allowed helper script (see deploy/wifi-helper.sh). Connecting is
ADDITIVE (keeps the current network) to avoid locking a WiFi-only Pi out.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from typing import List

logger = logging.getLogger("WiFi")

HELPER = "/usr/local/bin/stella-wifi-helper.sh"


def is_online(timeout: float = 3.0) -> bool:
    """True if the internet is reachable (DNS to a public resolver)."""
    for host in ("1.1.1.1", "8.8.8.8"):
        try:
            socket.setdefaulttimeout(timeout)
            socket.create_connection((host, 53))
            return True
        except Exception:
            continue
    return False


def helper_installed() -> bool:
    return os.path.exists(HELPER)


def _run(args: List[str], timeout: float = 25.0) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-n", HELPER, *args],
                          capture_output=True, text=True, timeout=timeout)


def scan() -> List[str]:
    """Return nearby WiFi SSIDs (best-effort). Empty if the helper isn't set up."""
    if not helper_installed():
        logger.info("WiFi helper not installed (run deploy/install-wifi-helper.sh)")
        return []
    try:
        r = _run(["scan"])
        if r.returncode != 0:
            logger.warning("wifi scan failed: %s", (r.stderr or "").strip()[:120])
            return []
        seen, out = set(), []
        for line in r.stdout.splitlines():
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:25]
    except Exception as exc:
        logger.warning("wifi scan error: %s", exc)
        return []


def _unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            out.append(s[i + 1]); i += 2
        else:
            out.append(s[i]); i += 1
    return ''.join(out)


def parse_wifi_qr(data: str):
    """Parse a standard Wi-Fi QR payload 'WIFI:S:ssid;T:WPA;P:pass;;'.
    Returns (ssid, password) or None."""
    if not data or not data.upper().startswith('WIFI:'):
        return None
    body = data[5:]
    parts, cur, i = [], '', 0
    while i < len(body):                     # split on UNescaped ';'
        if body[i] == '\\' and i + 1 < len(body):
            cur += body[i:i + 2]; i += 2; continue
        if body[i] == ';':
            parts.append(cur); cur = ''; i += 1; continue
        cur += body[i]; i += 1
    if cur:
        parts.append(cur)
    ssid = pw = ''
    for p in parts:
        if not p:
            continue
        k, _, v = p.partition(':')
        if k.upper() == 'S':
            ssid = _unescape(v)
        elif k.upper() == 'P':
            pw = _unescape(v)
    return (ssid, pw) if ssid else None


def read_wifi_qr(camera_manager, timeout: float = 25.0):
    """Watch the camera for a Wi-Fi QR code and return (ssid, password).

    Works fully offline (no internet needed) — ideal for onboarding when Wi-Fi
    is down. Returns None if nothing is decoded within *timeout* seconds.
    """
    try:
        import cv2
        import time as _t
    except Exception:
        return None
    detector = cv2.QRCodeDetector()
    end = _t.time() + timeout
    while _t.time() < end:
        f = camera_manager.get_latest_frame() if camera_manager else None
        img = getattr(f, 'image', None)
        if img is not None:
            try:
                data, _pts, _ = detector.detectAndDecode(img)
            except Exception:
                data = ''
            if data and data.upper().startswith('WIFI:'):
                res = parse_wifi_qr(data)
                if res:
                    return res
        _t.sleep(0.15)
    return None


def connect(ssid: str, password: str = "") -> bool:
    """Add + enable a WiFi network (additive). Returns True on success."""
    if not ssid or not helper_installed():
        return False
    try:
        r = _run(["add", ssid, password or ""])
        ok = r.returncode == 0 and "OK" in (r.stdout or "")
        if not ok:
            logger.warning("wifi add failed: %s",
                           ((r.stderr or "") + (r.stdout or "")).strip()[:160])
        return ok
    except Exception as exc:
        logger.warning("wifi connect error: %s", exc)
        return False
