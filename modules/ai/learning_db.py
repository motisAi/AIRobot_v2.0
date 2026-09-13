"""Persistent Learning Database
================================
SQLite-backed storage for all robot memories: faces, objects, conversations,
user preferences, and environment knowledge.  Survives reboots so the robot
doesn't forget what it learned yesterday.

Tables:
 - ``memories``     — general short/long-term memories with importance scores.
 - ``faces``        — face embeddings and metadata (serialized as blobs).
 - ``objects``      — learned custom objects with visual features.
 - ``conversations``— conversation history for context recall.
 - ``preferences``  — per-user preference key/value store.

Thread-safe: all public methods use an internal lock around the connection.
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import system_config

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "robot_memory.db"


class LearningDB:
    """SQLite persistence layer for robot learning and memory."""

    def __init__(self, db_path: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open (or create) the database and ensure schema exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self.logger.info("Learning DB opened (%s)", self.db_path)

    def stop(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        self.logger.info("Learning DB closed")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self) -> None:
        with self._lock:
            c = self._conn.cursor()
            c.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    type        TEXT NOT NULL DEFAULT 'short_term',
                    content     TEXT NOT NULL,
                    context     TEXT DEFAULT '{}',
                    importance  REAL DEFAULT 0.5,
                    created_at  TEXT NOT NULL,
                    accessed_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    expiry      TEXT
                );

                CREATE TABLE IF NOT EXISTS faces (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    embeddings  BLOB,
                    is_master   INTEGER DEFAULT 0,
                    permissions TEXT DEFAULT '["basic"]',
                    first_seen  TEXT NOT NULL,
                    last_seen   TEXT NOT NULL,
                    interaction_count INTEGER DEFAULT 0,
                    metadata    TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS objects (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    label       TEXT NOT NULL,
                    custom_name TEXT,
                    features    BLOB,
                    learned_at  TEXT NOT NULL,
                    seen_count  INTEGER DEFAULT 1,
                    last_seen   TEXT,
                    metadata    TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    user_id     TEXT NOT NULL,
                    key         TEXT NOT NULL,
                    value       TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                );

                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
                CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_objects_label ON objects(label);
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------
    def add_memory(self, content: str, memory_type: str = "short_term",
                   importance: float = 0.5, context: Optional[Dict] = None,
                   expiry: Optional[str] = None) -> int:
        """Store a memory and return its row id."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "INSERT INTO memories (type, content, context, importance, created_at, expiry) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_type, content, json.dumps(context or {}), importance,
                 datetime.now().isoformat(), expiry),
            )
            self._conn.commit()
            return c.lastrowid

    def recall_memories(self, query: str, memory_type: str = "all",
                        limit: int = 10) -> List[Dict]:
        """Search memories by substring match.  Returns newest-first."""
        with self._lock:
            c = self._conn.cursor()
            if memory_type == "all":
                c.execute(
                    "SELECT * FROM memories WHERE content LIKE ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (f"%{query}%", limit),
                )
            else:
                c.execute(
                    "SELECT * FROM memories WHERE content LIKE ? AND type = ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (f"%{query}%", memory_type, limit),
                )
            rows = c.fetchall()
            # Update access stats
            for row in rows:
                c.execute(
                    "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 "
                    "WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]),
                )
            self._conn.commit()
            return [dict(r) for r in rows]

    def promote_memories(self) -> int:
        """Move high-importance short-term memories to long-term.

        Returns the number of promoted memories.
        """
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "UPDATE memories SET type = 'long_term' "
                "WHERE type = 'short_term' AND importance >= 0.7"
            )
            self._conn.commit()
            return c.rowcount

    def cleanup_expired(self) -> int:
        """Delete expired memories.  Returns removed count."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "DELETE FROM memories WHERE expiry IS NOT NULL AND expiry < ?",
                (datetime.now().isoformat(),),
            )
            self._conn.commit()
            return c.rowcount

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------
    def save_face(self, face_id: str, name: str, embeddings: Any,
                  is_master: bool = False, permissions: Optional[List[str]] = None,
                  metadata: Optional[Dict] = None) -> None:
        """Upsert a face entry.  Embeddings are pickled."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO faces "
                "(id, name, embeddings, is_master, permissions, first_seen, last_seen, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (face_id, name, pickle.dumps(embeddings), int(is_master),
                 json.dumps(permissions or ["basic"]),
                 datetime.now().isoformat(), datetime.now().isoformat(),
                 json.dumps(metadata or {})),
            )
            self._conn.commit()

    def load_faces(self) -> Dict[str, Dict]:
        """Load all faces from the database."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM faces").fetchall()
            faces = {}
            for row in rows:
                faces[row["id"]] = {
                    "name": row["name"],
                    "embeddings": pickle.loads(row["embeddings"]) if row["embeddings"] else [],
                    "is_master": bool(row["is_master"]),
                    "permissions": json.loads(row["permissions"]),
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "interaction_count": row["interaction_count"],
                    "metadata": json.loads(row["metadata"]),
                }
            return faces

    def update_face_seen(self, face_id: str) -> None:
        """Update last_seen and increment interaction count."""
        with self._lock:
            self._conn.execute(
                "UPDATE faces SET last_seen = ?, interaction_count = interaction_count + 1 "
                "WHERE id = ?",
                (datetime.now().isoformat(), face_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------
    def save_object(self, label: str, custom_name: Optional[str] = None,
                    features: Optional[Any] = None,
                    metadata: Optional[Dict] = None) -> int:
        """Store a learned object.  Returns row id."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "INSERT INTO objects (label, custom_name, features, learned_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (label, custom_name, pickle.dumps(features) if features else None,
                 datetime.now().isoformat(), json.dumps(metadata or {})),
            )
            self._conn.commit()
            return c.lastrowid

    def get_objects(self, label: Optional[str] = None) -> List[Dict]:
        with self._lock:
            if label:
                rows = self._conn.execute(
                    "SELECT * FROM objects WHERE label = ?", (label,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM objects").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    def log_conversation(self, role: str, content: str,
                         user_id: Optional[str] = None) -> None:
        """Append a conversation turn."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (user_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (user_id, role, content, datetime.now().isoformat()),
            )
            self._conn.commit()

    def get_recent_conversations(self, limit: int = 20,
                                  user_id: Optional[str] = None) -> List[Dict]:
        with self._lock:
            if user_id:
                rows = self._conn.execute(
                    "SELECT * FROM conversations WHERE user_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM conversations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------
    def set_preference(self, user_id: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO preferences (user_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, key, value, datetime.now().isoformat()),
            )
            self._conn.commit()

    def get_preference(self, user_id: str, key: str,
                       default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM preferences WHERE user_id = ? AND key = ?",
                (user_id, key),
            ).fetchone()
            return row["value"] if row else default

    def get_all_preferences(self, user_id: str) -> Dict[str, str]:
        """Return all preferences for a user as a dict."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Alias for stop() — used by main.py shutdown."""
        self.stop()
