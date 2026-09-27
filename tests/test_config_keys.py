"""Hygiene test: every section/key in config/config.yaml must map to a RobotConfig
section and a dataclass field, so a typo or a decoy key can never be silently ignored.

Run:  venv/bin/python tests/test_config_keys.py   (also pytest-compatible)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Sections read directly from YAML by main.py (no dataclass yet). Shrink this list,
# never grow it: each entry is tracked debt (see architecture §14).
ALLOW_NO_DATACLASS = {"mqtt"}


def check():
    import yaml
    os.chdir(ROOT)
    from config.settings import config
    data = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8")) or {}
    problems = []
    for section, body in data.items():
        if section in ALLOW_NO_DATACLASS:
            continue
        if not isinstance(body, dict):
            problems.append(f"{section}: not a mapping")
            continue
        if not hasattr(config, section):
            problems.append(f"{section}: no such config section (RobotConfig.{section})")
            continue
        obj = getattr(config, section)
        for key in body:
            if not hasattr(obj, key):
                problems.append(f"{section}.{key}: not a field of {type(obj).__name__}")
    return problems


def test_config_keys_map_to_fields():
    problems = check()
    assert not problems, "config.yaml keys with no dataclass field:\n  " + "\n  ".join(problems)


if __name__ == "__main__":
    p = check()
    if p:
        print("CONFIG KEY PROBLEMS:\n  " + "\n  ".join(p))
        sys.exit(1)
    print("OK: every config.yaml key maps to a settings field")
