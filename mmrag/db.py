"""SQLite = the source of truth (manifest, page text, chunks, embeddings).
The in-memory vector index is always rebuilt from here, so a crash can never
leave the index and the data out of sync."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size INTEGER NOT NULL, mtime REAL NOT NULL,
    status TEXT NOT NULL, error TEXT,
    n_pages INTEGER DEFAULT 0, n_text_chunks INTEGER DEFAULT 0, n_images INTEGER DEFAULT 0,
    n_quarantined INTEGER DEFAULT 0, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image')),
    content TEXT, ctx_start INTEGER, ctx_end INTEGER, image_path TEXT,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- meta ---------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # ---- documents / manifest -------------------------------------------------
    def manifest(self) -> dict[str, dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents").fetchall()
        return {r["doc_id"]: dict(r) for r in rows}

    def upsert_document(self, doc_id: str, sha256: str, size: int, mtime: float,
                        status: str, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO documents(doc_id, sha256, size, mtime, status, error, updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET sha256=excluded.sha256, size=excluded.size,
                   mtime=excluded.mtime, status=excluded.status, error=excluded.error,
                   updated_at=excluded.updated_at""",
                (doc_id, sha256, size, mtime, status, error, _now()))

    def touch_document(self, doc_id: str, size: int, mtime: float) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET size=?, mtime=? WHERE doc_id=?", (size, mtime, doc_id))

    def delete_content(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))  # cascades

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents")  # cascades to pages + chunks

    def replace_content(self, doc_id: str, pages: list[tuple[int, str]], chunks: list[dict],
                        stats: dict) -> None:
        """Atomically swap a document's searchable content and mark it ready."""
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
            c.executemany("INSERT INTO pages(doc_id,page,text) VALUES(?,?,?)",
                          [(doc_id, p, t) for p, t in pages])
            c.executemany(
                """INSERT INTO chunks(doc_id,page,modality,content,ctx_start,ctx_end,image_path,embedding)
                   VALUES(?,?,?,?,?,?,?,?)""",
                [(doc_id, ch["page"], ch["modality"], ch.get("content"), ch.get("ctx_start"),
                  ch.get("ctx_end"), ch.get("image_path"), ch["embedding"]) for ch in chunks])
            c.execute(
                """UPDATE documents SET status='ready', error=NULL, n_pages=?, n_text_chunks=?,
                   n_images=?, n_quarantined=?, updated_at=? WHERE doc_id=?""",
                (stats["n_pages"], stats["n_text_chunks"], stats["n_images"],
                 stats["n_quarantined"], _now(), doc_id))

    def mark_failed(self, doc_id: str, error: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET status='failed', error=?, updated_at=? WHERE doc_id=?",
                      (error[:500], _now(), doc_id))

    # ---- reads for retrieval -----------------------------------------------------
    def all_embeddings(self) -> list[tuple[int, str, bytes]]:
        with self._conn() as c:
            rows = c.execute("SELECT id, modality, embedding FROM chunks").fetchall()
        return [(r["id"], r["modality"], r["embedding"]) for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        with self._conn() as c:
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                q = ",".join("?" * len(part))
                for r in c.execute(
                        f"SELECT id,doc_id,page,modality,content,ctx_start,ctx_end,image_path "
                        f"FROM chunks WHERE id IN ({q})", part):
                    out[r["id"]] = dict(r)
        return out

    def get_page_text(self, doc_id: str, page: int) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT text FROM pages WHERE doc_id=? AND page=?",
                            (doc_id, page)).fetchone()
        return row["text"] if row else None

    def stats(self) -> dict:
        with self._conn() as c:
            docs = c.execute(
                "SELECT status, COUNT(*) n FROM documents GROUP BY status").fetchall()
            chunks = c.execute(
                "SELECT modality, COUNT(*) n FROM chunks GROUP BY modality").fetchall()
        by_status = {r["status"]: r["n"] for r in docs}
        return {
            "documents": sum(by_status.values()),
            "documents_ready": by_status.get("ready", 0),
            "documents_failed": by_status.get("failed", 0),
            "text_chunks": next((r["n"] for r in chunks if r["modality"] == "text"), 0),
            "images": next((r["n"] for r in chunks if r["modality"] == "image"), 0),
        }
