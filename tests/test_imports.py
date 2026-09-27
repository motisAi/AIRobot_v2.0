"""Smoke test: every project module that main.py imports must import cleanly.

Run on the Pi:  venv/bin/python -m pytest tests -q   (or plain: venv/bin/python tests/test_imports.py)
Guardian runs the same logic in evolution/guardian.py::check_imports.
"""
import importlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT_PKGS = ("modules", "core", "config", "parts_used", "face_bridge", "evolution")


def project_imports():
    src = (ROOT / "main.py").read_text(encoding="utf-8", errors="replace")
    mods = set()
    for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", src, re.M):
        if m.group(1).split(".")[0] in PROJECT_PKGS:
            mods.add(m.group(1))
    for m in re.finditer(r"^\s*import\s+([\w.]+)", src, re.M):
        if m.group(1).split(".")[0] in PROJECT_PKGS:
            mods.add(m.group(1))
    return sorted(mods)


def test_main_imports():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    failures = []
    for mod in project_imports():
        try:
            importlib.import_module(mod)
        except BaseException as e:  # SystemExit from optional deps counts as failure
            failures.append(f"{mod}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    try:
        test_main_imports()
    except AssertionError as e:
        print("IMPORT FAILURES:\n", e)
        sys.exit(1)
    print(f"OK: {len(project_imports())} project modules import")
