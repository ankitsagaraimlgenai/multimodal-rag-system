"""Ingestion pipeline: scan -> load -> chunk -> injection-scan -> embed -> store.

Single writer (a lock), idempotent, and failure-isolated per document: a bad PDF is
marked 'failed' and never blocks the others. SQLite commits are atomic per document.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from mmrag import injection
from mmrag.db import Database
from mmrag.ingestion.chunking import TokenChunker, context_windows
from mmrag.ingestion.scanner import FileInfo, scan_inbox
from mmrag.loaders.pdf import load_document
from mmrag.store import VectorStore

if TYPE_CHECKING:  # pragma: no cover
    from mmrag.config import Settings
    from mmrag.embeddings import ClipEmbedder

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    ingested: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    unchanged: int = 0
    duration_s: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.ingested or self.failed or self.removed)

    def summary(self) -> str:
        return (f"{len(self.ingested)} indexed, {len(self.failed)} failed, {len(self.removed)} removed, "
                f"{self.unchanged} unchanged, {len(self.skipped)} skipped ({self.duration_s:.1f}s)")


def _doc_key(doc_id: str) -> str:
    return hashlib.sha1(doc_id.encode()).hexdigest()[:10]


class IngestionPipeline:
    def __init__(self, cfg: "Settings", db: Database, embedder: "ClipEmbedder", store: VectorStore,
                 loader: Callable = load_document):
        self.cfg, self.db, self.embedder, self.store, self.loader = cfg, db, embedder, store, loader
        self.chunker = TokenChunker(lambda t: self.embedder.count_tokens(t),
                                    cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
        self._lock = threading.Lock()

    # ---- compatibility ----------------------------------------------------------
    def fingerprint(self) -> str:
        c = self.cfg
        return "|".join(map(str, (c.clip_model_name, c.chunk_size_tokens, c.chunk_overlap_tokens,
                                  c.context_window_chunks, c.min_image_px, c.max_image_dim)))

    def ensure_compatible(self) -> None:
        """If embedding/chunking settings changed, derived data is stale: wipe it and re-ingest from the inbox."""
        fp, old = self.fingerprint(), self.db.get_meta("fingerprint")
        if old is not None and old != fp:
            log.warning("embedding/chunking config changed; rebuilding index from inbox")
            self.db.clear()
            shutil.rmtree(self.cfg.images_dir, ignore_errors=True)
        self.db.set_meta("fingerprint", fp)

    # ---- main entry ---------------------------------------------------------------
    def run(self, retry_failed: bool = False) -> IngestReport:
        with self._lock:
            started = time.perf_counter()
            report = IngestReport()
            scan = scan_inbox(self.cfg.inbox_dir, self.db.manifest(),
                              max_file_bytes=self.cfg.max_file_bytes,
                              settle_seconds=self.cfg.scan_settle_seconds, retry_failed=retry_failed)
            report.unchanged, report.skipped = len(scan.unchanged), scan.skipped
            for info in scan.touched:
                self.db.touch_document(info.doc_id, info.size, info.mtime)
            for doc_id in scan.deleted:
                self._remove(doc_id)
                report.removed.append(doc_id)
            for info in scan.pending:
                try:
                    self._ingest_one(info)
                    report.ingested.append(info.doc_id)
                except Exception as exc:
                    log.exception("ingestion failed", extra={"doc_id": info.doc_id})
                    self._fail(info, f"{type(exc).__name__}: {exc}")
                    report.failed.append((info.doc_id, str(exc)))
            if report.changed:
                self.store.reload()
            report.duration_s = time.perf_counter() - started
            if report.changed:
                log.info("ingestion run complete", extra={"summary": report.summary()})
            return report

    # ---- one document ---------------------------------------------------------------
    def _ingest_one(self, info: FileInfo) -> None:
        cfg = self.cfg
        log.info("ingesting", extra={"doc_id": info.doc_id})
        self.db.upsert_document(info.doc_id, info.sha256, info.size, info.mtime, "processing")
        doc_dir = cfg.images_dir / _doc_key(info.doc_id)
        version_dir = doc_dir / info.sha256[:10]
        shutil.rmtree(version_dir, ignore_errors=True)
        version_dir.mkdir(parents=True, exist_ok=True)

        pages: list[tuple[int, str]] = []
        rows: list[dict] = []
        n_text = n_img = n_quarantined = n_pages = 0

        for page in self.loader(info.path, cfg):
            n_pages = page.number
            # ---- text: chunk -> scan for injected instructions -> embed ----
            chunks = self.chunker.chunk(page.text) if page.text.strip() else []
            hit = next((v for v in (injection.scan(c.text, cfg.injection_threshold) for c in chunks)
                        if v.flagged), None)
            if hit:  # quarantine the whole page's text: neighbours would leak it via context windows
                n_quarantined += 1
                log.warning("page quarantined: possible prompt injection",
                            extra={"doc_id": info.doc_id, "page": page.number, "matches": hit.matches})
            elif chunks:
                pages.append((page.number, page.text))
                vecs = self.embedder.embed_texts([c.text for c in chunks])
                windows = context_windows(chunks, cfg.context_window_chunks)
                for c, v, (cs, ce) in zip(chunks, vecs, windows):
                    rows.append({"page": page.number, "modality": "text", "content": c.text,
                                 "ctx_start": cs, "ctx_end": ce, "embedding": v.tobytes()})
                n_text += len(chunks)

            # ---- images: save to disk, embed with CLIP ----
            if page.images:
                vecs = self.embedder.embed_images([a.image for a in page.images])
                for a, v in zip(page.images, vecs):
                    path = version_dir / f"p{page.number}_i{a.index}.png"
                    a.image.save(path, format="PNG")
                    rows.append({"page": page.number, "modality": "image",
                                 "content": f"[image on page {page.number}]",
                                 "image_path": str(path), "embedding": v.tobytes()})
                n_img += len(page.images)

        self.db.replace_content(info.doc_id, pages, rows, {
            "n_pages": n_pages,
            "n_text_chunks": n_text, "n_images": n_img, "n_quarantined": n_quarantined})
        for old in doc_dir.iterdir():  # drop image folders from previous versions
            if old.is_dir() and old != version_dir:
                shutil.rmtree(old, ignore_errors=True)

    def _fail(self, info: FileInfo, error: str) -> None:
        """Failed re-ingest removes stale content so answers never come from an outdated version."""
        self.db.delete_content(info.doc_id)
        self.db.mark_failed(info.doc_id, error)
        shutil.rmtree(self.cfg.images_dir / _doc_key(info.doc_id), ignore_errors=True)

    def _remove(self, doc_id: str) -> None:
        log.info("removing deleted document", extra={"doc_id": doc_id})
        self.db.delete_document(doc_id)
        shutil.rmtree(self.cfg.images_dir / _doc_key(doc_id), ignore_errors=True)
