#!/usr/bin/env python3
"""Guardian — Stella's deterministic safety net.

Runs a fixed list of health checks (NO LLM involved), prints a one-line
verdict per check, writes evolution/reports/guardian-latest.json and exits
0 (healthy) or 1 (something is broken).  Used by deploy/deploy.sh after every
restart, by the nightly evolution timer, and by hand:

    venv/bin/python evolution/guardian.py            # quick checks
    venv/bin/python evolution/guardian.py --wait     # after a restart: wait for
                                                     # the service, then watch it
                                                     # for --stable seconds

Checks are "hard" (fail the run) or "soft" (reported only).  Hardware that the
config does not require is always soft, so Guardian stays green on a bench
with no camera/mic plugged in.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "evolution" / "reports" / "guardian-latest.json"
SERVICE = "airobot"
PY = ROOT / "venv" / "bin" / "python"

# Import roots that belong to the project (anything else is a library).
PROJECT_PKGS = ("modules", "core", "config", "parts_used", "face_bridge", "evolution")

FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last)",
    "malloc_consolidate",
    "HAILO_OUT_OF_PHYSICAL_DEVICES",
    "double free",
    "Segmentation fault",
)
MAX_ERRORS_2MIN = 15
# present in the log = something silently degraded (reported, not fatal)
SOFT_LOG_PATTERNS = ("Object detection off", "Object detector unavailable", "Wake mic not available",
                     "Answered via fallback provider: hailo", "returned EMPTY content")


@dataclass
class Check:
    name: str
    ok: bool
    hard: bool
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Report:
    started: str
    checks: list = field(default_factory=list)
    healthy: bool = True

    def add(self, c: Check) -> None:
        self.checks.append(c)
        if c.hard and not c.ok:
            self.healthy = False
        flag = "OK  " if c.ok else ("FAIL" if c.hard else "warn")
        print(f"[{flag}] {c.name:<12} {c.detail}")


def _run(cmd, timeout=30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _timed(fn):
    t = time.monotonic()
    c = fn()
    c.seconds = round(time.monotonic() - t, 2)
    return c


# --------------------------------------------------------------------- checks
def check_syntax() -> Check:
    r = _run([str(PY), "-m", "compileall", "-q",
              "-x", r"(venv|\.venv|__pycache__|data)",
              str(ROOT)], timeout=240)
    bad = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    return Check("syntax", r.returncode == 0, True,
                 "all files compile" if r.returncode == 0 else bad[-1][:200])


def _project_imports() -> list[str]:
    """Every `from X import` / `import X` in main.py that points into the repo."""
    mods: set[str] = set()
    src = (ROOT / "main.py").read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", src, re.M):
        if m.group(1).split(".")[0] in PROJECT_PKGS:
            mods.add(m.group(1))
    for m in re.finditer(r"^\s*import\s+([\w.]+)", src, re.M):
        if m.group(1).split(".")[0] in PROJECT_PKGS:
            mods.add(m.group(1))
    return sorted(mods)


def check_imports() -> Check:
    mods = _project_imports()
    code = (
        "import importlib,sys,os\n"
        "os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1])\n"
        "bad=[]\n"
        "for m in sys.argv[2:]:\n"
        "    try: importlib.import_module(m)\n"
        "    except BaseException as e: bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n"
    )
    r = _run([str(PY), "-c", code, str(ROOT), *mods], timeout=180)
    out = (r.stdout + r.stderr).strip()
    return Check("imports", r.returncode == 0, True,
                 f"{len(mods)} project modules import" if r.returncode == 0
                 else out.splitlines()[0][:200])


def check_config() -> Check:
    code = (
        "import sys,os; os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1])\n"
        "from config.settings import config as c\n"
        "print(getattr(getattr(c,'behavior',None),'robot_name','?'))\n"
    )
    r = _run([str(PY), "-c", code, str(ROOT)], timeout=60)
    ok = r.returncode == 0
    return Check("config", ok, True,
                 f"config loads (robot_name={r.stdout.strip()})" if ok
                 else (r.stderr.strip().splitlines() or ["?"])[-1][:200])


def service_state() -> str:
    return _run(["systemctl", "is-active", SERVICE]).stdout.strip()


def check_service(wait: int, stable: int) -> Check:
    deadline = time.monotonic() + wait
    st = service_state()
    while st != "active" and time.monotonic() < deadline:
        time.sleep(2)
        st = service_state()
    if st != "active":
        return Check("service", False, True, f"{SERVICE} is {st}")
    pid0 = _run(["systemctl", "show", "-p", "MainPID", "--value", SERVICE]).stdout.strip()
    if stable:
        end = time.monotonic() + stable
        while time.monotonic() < end:
            time.sleep(5)
            if service_state() != "active":
                return Check("service", False, True, f"{SERVICE} died within {stable}s")
        pid1 = _run(["systemctl", "show", "-p", "MainPID", "--value", SERVICE]).stdout.strip()
        if pid1 != pid0:
            return Check("service", False, True, f"{SERVICE} restarted (pid {pid0}->{pid1})")
    return Check("service", True, True, f"{SERVICE} active (pid {pid0})")


def check_log(minutes: int = 2) -> Check:
    r = _run(["journalctl", "-u", SERVICE, "--since", f"{minutes} min ago",
              "--no-pager", "-o", "cat"], timeout=30)
    lines = r.stdout.splitlines()
    errors = [l for l in lines if " ERROR " in l or "Error:" in l]
    fatal = [l for l in lines if any(p in l for p in FATAL_LOG_PATTERNS)]
    if fatal:
        return Check("log", False, True, f"fatal pattern: {fatal[0][:160]}")
    if len(errors) > MAX_ERRORS_2MIN:
        return Check("log", False, True, f"{len(errors)} ERROR lines in {minutes} min: {errors[-1][:120]}")
    soft = [p for p in SOFT_LOG_PATTERNS if any(p in l for l in lines)]
    detail = f"{len(lines)} lines, {len(errors)} errors in {minutes} min"
    if soft:
        detail = "warn: " + "; ".join(soft) + " | " + detail
    return Check("log", True, True, detail)


def check_hardware() -> Check:
    cams = sorted(glob.glob("/dev/video[0-9]*"))
    mics = sorted(glob.glob("/dev/snd/pcmC*D*c"))
    hailo = os.path.exists("/dev/hailo0")
    serial = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    real_cams = [c for c in cams if not re.search(r"rp1-cfe|csi|pispbe|rpivid", _v4l_name(c), re.I)]
    detail = f"cams={len(real_cams)} mics={len(mics)} hailo={'yes' if hailo else 'NO'} serial={serial or '-'}"
    # RobotNet dongle plugged in but no wlx interface => its DKMS driver is not loaded (bug_044)
    dongle = any(Path(d, "idVendor").exists() and Path(d, "idVendor").read_text().strip() == "2357"
                 for d in glob.glob("/sys/bus/usb/devices/*"))
    if dongle and not glob.glob("/sys/class/net/wlx*"):
        detail += " | warn: WiFi dongle present but no wlx interface (sudo modprobe 8821au; bug_044)"
    return Check("hardware", True, False, detail)


def _v4l_name(node: str) -> str:
    try:
        return Path(f"/sys/class/video4linux/{os.path.basename(node)}/name").read_text()
    except OSError:
        return ""


def check_brain(timeout: int = 40) -> Check:
    if not os.path.exists("/dev/hailo0"):
        return Check("brain", False, False, "/dev/hailo0 missing -> offline brain dead "
                     "(fix: sudo dkms autoinstall && sudo modprobe hailo1x_pci)")
    body = json.dumps({"model": "qwen2.5-instruct:1.5b",
                       "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                       "max_tokens": 5})
    t = time.monotonic()
    r = _run(["curl", "-s", "-m", str(timeout), "http://127.0.0.1:8000/v1/chat/completions",
              "-H", "Content-Type: application/json", "-d", body], timeout=timeout + 5)
    dt = time.monotonic() - t
    try:
        txt = json.loads(r.stdout)["choices"][0]["message"]["content"].strip()
        return Check("brain", True, False, f"hailo-ollama answered in {dt:.1f}s: {txt[:30]!r}")
    except Exception:
        return Check("brain", False, False, f"hailo-ollama no answer ({r.stdout[:80]!r})")


def check_resources() -> Check:
    du = shutil.disk_usage(str(ROOT))
    free_gb = du.free / 1e9
    mem_avail_mb = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                mem_avail_mb = int(line.split()[1]) / 1024
    except Exception:
        pass
    thr, temp = "?", "?"
    try:
        thr = _run(["vcgencmd", "get_throttled"]).stdout.strip().split("=")[-1]
        temp = _run(["vcgencmd", "measure_temp"]).stdout.strip().split("=")[-1]
    except Exception:
        pass
    problems, warnings = [], []
    if free_gb < 2:
        problems.append(f"disk {free_gb:.1f}GB")
    if mem_avail_mb and mem_avail_mb < 500:
        problems.append(f"ram {mem_avail_mb:.0f}MB")
    hard_pw, soft_pw = decode_throttled(thr)
    problems += hard_pw
    warnings += soft_pw
    detail = f"disk {free_gb:.1f}GB free, ram {mem_avail_mb:.0f}MB avail, temp {temp}, throttled={thr}"
    if warnings:
        detail = "warn: " + ", ".join(warnings) + " | " + detail
    return Check("resources", not problems, True,
                 detail if not problems else "POWER: " + ", ".join(problems) + f" | {detail}")


# vcgencmd get_throttled bits (Raspberry Pi documentation)
_THROTTLE_BITS = {
    0: ("undervoltage NOW", True),
    1: ("arm frequency capped NOW", False),
    2: ("throttled NOW", True),
    3: ("soft temperature limit NOW", False),
    16: ("undervoltage occurred since boot", True),
    17: ("arm frequency cap occurred since boot", False),
    18: ("throttling occurred since boot", False),
    19: ("soft temperature limit occurred since boot", False),
}


def decode_throttled(hexval: str):
    """Return (hard_problems, soft_warnings) for a vcgencmd get_throttled value."""
    try:
        v = int(hexval, 16)
    except (TypeError, ValueError):
        return [], []
    hard, soft = [], []
    for bit, (label, is_hard) in _THROTTLE_BITS.items():
        if v & (1 << bit):
            (hard if is_hard else soft).append(label)
    return hard, soft


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", action="store_true", help="wait for the service to come up (after a restart)")
    ap.add_argument("--wait-seconds", type=int, default=90)
    ap.add_argument("--stable", type=int, default=45, help="with --wait: seconds the service must stay up")
    ap.add_argument("--no-brain", action="store_true", help="skip the hailo-ollama probe")
    ap.add_argument("--require-hailo", action="store_true", help="make the brain check hard")
    a = ap.parse_args()

    rep = Report(started=time.strftime("%Y-%m-%d %H:%M:%S"))
    rep.add(_timed(check_syntax))
    rep.add(_timed(check_imports))
    rep.add(_timed(check_config))
    rep.add(_timed(lambda: check_service(a.wait_seconds if a.wait else 0, a.stable if a.wait else 0)))
    rep.add(_timed(check_log))
    rep.add(_timed(check_hardware))
    if not a.no_brain:
        c = _timed(check_brain)
        c.hard = a.require_hailo
        rep.add(c)
    rep.add(_timed(check_resources))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"healthy": rep.healthy, "started": rep.started,
                                  "checks": [asdict(c) for c in rep.checks]}, indent=2))
    print("GUARDIAN:", "HEALTHY" if rep.healthy else "UNHEALTHY", f"({REPORT.relative_to(ROOT)})")
    return 0 if rep.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
