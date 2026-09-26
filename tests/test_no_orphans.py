"""Hygiene test: every module under modules/, parts_used/ and core/ must be imported
by something in the repo. An orphan is dead code that Guardian and test_imports never
exercise (the 2026-09-27 review found ~5k lines of it).

Run:  venv/bin/python tests/test_no_orphans.py   (also pytest-compatible)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("modules", "parts_used", "core")
SKIP_DIRS = {"venv", ".venv", "__pycache__", "data", ".git"}


def _py_files():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def _corpus():
    """All python source in the repo, keyed by path (read once)."""
    return {p: p.read_text(encoding="utf-8", errors="replace") for p in _py_files()}


def find_orphans():
    corpus = _corpus()
    orphans = []
    for d in SCAN_DIRS:
        for mod in sorted((ROOT / d).rglob("*.py")):
            if mod.name == "__init__.py" or "__pycache__" in mod.parts:
                continue
            rel = mod.relative_to(ROOT).with_suffix("")
            dotted = ".".join(rel.parts)              # modules.vision.vlm
            pkg, base = ".".join(rel.parts[:-1]), rel.parts[-1]
            pats = [
                r"\bimport\s+" + re.escape(dotted) + r"\b",
                r"\bfrom\s+" + re.escape(dotted) + r"\s+import\b",
                r"\bfrom\s+" + re.escape(pkg) + r"\s+import\s+[^\n]*\b" + re.escape(base) + r"\b",
            ]
            used = False
            for path, src in corpus.items():
                if path == mod:
                    continue
                if any(re.search(pt, src) for pt in pats):
                    used = True
                    break
            if not used:
                orphans.append(str(rel) + ".py")
    return orphans


def test_no_orphans():
    orphans = find_orphans()
    assert not orphans, "orphan modules (nothing imports them):\n  " + "\n  ".join(orphans)


if __name__ == "__main__":
    o = find_orphans()
    if o:
        print("ORPHANS:\n  " + "\n  ".join(o))
        sys.exit(1)
    print("OK: no orphan modules")
