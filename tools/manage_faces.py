#!/usr/bin/env python3
"""Manage Stella's face database (list / remove / rename).

Usage (run on the Pi, in ~/AIRobot_v2.0):
    ./venv/bin/python manage_faces.py list
    ./venv/bin/python manage_faces.py remove "<id or name>"
    ./venv/bin/python manage_faces.py rename "<id or name>" "<new name>"

IMPORTANT: STOP her before editing, or the running service overwrites your change
on shutdown. Correct order:
    sudo -n systemctl stop airobot
    ./venv/bin/python manage_faces.py remove "Some Name"
    sudo -n systemctl start airobot
Removing the master is blocked unless you pass --force (use enroll_master.py to redo it).
"""
import pickle
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "faces" / "face_db.pkl"  # repo root (tool lives in tools/)


def _name(v):
    if isinstance(v, dict):
        return v.get("name", "?")
    return getattr(v, "name", "?")


def _master(v):
    if isinstance(v, dict):
        return bool(v.get("is_master", False))
    return bool(getattr(v, "is_master", False))


def _nemb(v):
    e = v.get("embeddings") if isinstance(v, dict) else getattr(v, "embeddings", None)
    try:
        return len(e)
    except Exception:
        return 0


def _set_name(v, n):
    if isinstance(v, dict):
        v["name"] = n
    else:
        setattr(v, "name", n)


def load():
    with open(DB, "rb") as f:
        return pickle.load(f)


def save(db):
    with open(DB, "wb") as f:
        pickle.dump(db, f)


def matches(db, needle):
    needle = needle.lower()
    out = []
    for k, v in db.items():
        if needle in str(k).lower() or needle in _name(v).lower():
            out.append(k)
    return out


def main():
    if not DB.exists():
        print("no face DB at", DB); return
    args = sys.argv[1:]
    cmd = args[0] if args else "list"
    db = load()

    if cmd == "list":
        print(f"{len(db)} face(s) in {DB.name}:")
        for k, v in db.items():
            print(f"  id={k!r:24}  name={_name(v)!r:20}  master={_master(v)}  embeddings={_nemb(v)}")
        return

    if cmd == "remove":
        force = "--force" in args
        needle = [a for a in args[1:] if a != "--force"][0]
        hits = matches(db, needle)
        if not hits:
            print("no match for", repr(needle)); return
        for k in hits:
            if _master(db[k]) and not force:
                print(f"SKIP master id={k!r} (name={_name(db[k])!r}) — use --force to remove"); continue
            print(f"removing id={k!r} name={_name(db[k])!r}")
            del db[k]
        save(db); print("saved. restart Stella to reload.")
        return

    if cmd == "rename":
        needle, newname = args[1], args[2]
        hits = matches(db, needle)
        if not hits:
            print("no match for", repr(needle)); return
        for k in hits:
            print(f"renaming id={k!r}  {_name(db[k])!r} -> {newname!r}")
            _set_name(db[k], newname)
        save(db); print("saved. restart Stella to reload.")
        return

    print(__doc__)


if __name__ == "__main__":
    main()
