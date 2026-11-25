"""Persistent memory: SQLite vector store + a human-readable notes file.

Same storage pattern as Aaron's Hermes economy ledger (SQLite, local file,
no service). Vectors are stored at full dimension (768d from EmbeddingGemma
2) and searched at a truncated dimension (Matryoshka-style, 256d default),
using numpy when available and a pure-Python cosine otherwise.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from array import array
from pathlib import Path

from .embeddings import cosine as _cosine_py, truncate

try:  # numpy is optional; the framework must run without it
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def _pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


def _cosine(a: list[float], b: list[float]) -> float:
    if _np is not None:
        x = _np.asarray(a, dtype=_np.float32)
        y = _np.asarray(b, dtype=_np.float32)
        denom = float(_np.linalg.norm(x) * _np.linalg.norm(y))
        return float(_np.dot(x, y) / denom) if denom else 0.0
    return _cosine_py(a, b)


class Memory:
    """Long-term memory: remember(text) / recall(query) over SQLite."""

    def __init__(self, db_path: str, notes_path: str, embedder,
                 search_dim: int = 256, recall_k: int = 4):
        self.db_path = Path(db_path).expanduser()
        self.notes_path = Path(notes_path).expanduser()
        self.embedder = embedder
        self.search_dim = search_dim
        self.recall_k = recall_k
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        # The world bridge serves from worker threads, so the connection is
        # shared across threads and all access is serialized with a lock.
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   text TEXT NOT NULL,
                   kind TEXT NOT NULL DEFAULT 'note',
                   namespace TEXT NOT NULL DEFAULT '',
                   ts REAL NOT NULL,
                   dim INTEGER NOT NULL,
                   vec BLOB NOT NULL)""")
        # Migration for DBs created before namespaces existed.
        cols = [r[1] for r in self._db.execute("PRAGMA table_info(memories)")]
        if "namespace" not in cols:
            self._db.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT ''")
        self._db.commit()

    # -- write ---------------------------------------------------------------
    def remember(self, text: str, kind: str = "note", namespace: str = "") -> int:
        vec = self.embedder.embed_one(text)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO memories (text, kind, namespace, ts, dim, vec) "
                "VALUES (?,?,?,?,?,?)",
                (text, kind, namespace, time.time(), len(vec), _pack(vec)))
            self._db.commit()
            new_id = int(cur.lastrowid)
        stamp = time.strftime("%Y-%m-%d %H:%M")
        tag = f"{namespace}/{kind}" if namespace else kind
        try:
            with open(self.notes_path, "a") as fh:
                fh.write(f"- {stamp} · [{tag}] {text}\n")
        except OSError:
            pass  # the DB is the source of truth; notes are a courtesy copy
        return new_id

    # -- read ------------------------------------------------------------------
    def recall(self, query: str, k: int | None = None,
               namespace: str | None = None) -> list[dict]:
        k = k or self.recall_k
        q = truncate(self.embedder.embed_one(query), self.search_dim)
        with self._lock:
            if namespace is None:
                rows = self._db.execute(
                    "SELECT text, kind, ts, vec FROM memories").fetchall()
            else:
                rows = self._db.execute(
                    "SELECT text, kind, ts, vec FROM memories WHERE namespace = ?",
                    (namespace,)).fetchall()
        scored = []
        for text, kind, ts, blob in rows:
            vec = truncate(_unpack(blob), self.search_dim)
            scored.append({"text": text, "kind": kind, "ts": ts,
                           "score": _cosine(q, vec)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    # -- bulk namespace access (soul save/load) --------------------------------
    def list_namespace(self, namespace: str) -> list[dict]:
        """Every memory in a namespace, oldest first:
        [{text, kind, ts}, ...]. The soul export reads through this."""
        with self._lock:
            rows = self._db.execute(
                "SELECT text, kind, ts FROM memories WHERE namespace = ? "
                "ORDER BY ts, id", (namespace,)).fetchall()
        return [{"text": text, "kind": kind, "ts": ts}
                for text, kind, ts in rows]

    def replace_namespace(self, namespace: str, entries) -> int:
        """Bulk import: clear the namespace, then insert `entries`
        ({text, kind, ts?}). Saved timestamps are preserved — rows are
        written directly rather than via remember(), so the notes file
        is not spammed with one line per restored memory. Returns the
        number of memories restored."""
        rows = []
        for entry in entries:
            text = str(entry["text"])
            kind = str(entry.get("kind") or "note")
            ts = entry.get("ts")
            ts = float(ts) if isinstance(ts, (int, float)) else time.time()
            vec = self.embedder.embed_one(text)
            rows.append((text, kind, namespace, ts, len(vec), _pack(vec)))
        with self._lock:
            self._db.execute("DELETE FROM memories WHERE namespace = ?",
                             (namespace,))
            self._db.executemany(
                "INSERT INTO memories (text, kind, namespace, ts, dim, vec) "
                "VALUES (?,?,?,?,?,?)", rows)
            self._db.commit()
        return len(rows)

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


class SessionHistory:
    """In-session conversation turns with a compression hook.

    The agent's context manager calls `set_summary()` after folding older
    turns into a summary; `messages()` then replays summary + recent turns
    instead of the full transcript, which is what lets small context
    windows (8K-32K) hold long sessions.
    """

    def __init__(self):
        self.turns: list[dict] = []
        self.summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def set_summary(self, summary: str) -> None:
        self.summary = summary

    def messages(self) -> list[dict]:
        out: list[dict] = []
        if self.summary:
            out.append({"role": "system",
                        "content": f"Earlier conversation summary: {self.summary}"})
        out.extend(self.turns)
        return out
