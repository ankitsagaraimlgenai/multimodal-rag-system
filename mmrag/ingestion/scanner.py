"""Ingestion scanner: diff the inbox folder against the manifest in SQLite.

Classifies every file as new / changed / unchanged / deleted without re-hashing
files whose (size, mtime) are unchanged. Skips hidden/temp files, files that are
still being written, and files over the size limit.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf"}


@dataclass(frozen=True)
class FileInfo:
    doc_id: str      # path relative to the inbox, posix style
    path: Path
    size: int
    mtime: float
    sha256: str


@dataclass
class ScanResult:
    new: list[FileInfo] = field(default_factory=list)
    changed: list[FileInfo] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    touched: list[FileInfo] = field(default_factory=list)   # same content, new mtime -> refresh manifest
    deleted: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def pending(self) -> list[FileInfo]:
        return [*self.new, *self.changed]


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def scan_inbox(inbox: Path, manifest: dict[str, dict], *, max_file_bytes: int,
               settle_seconds: float = 2.0, retry_failed: bool = False,
               now: float | None = None) -> ScanResult:
    now = time.time() if now is None else now
    result = ScanResult()
    seen: set[str] = set()
    inbox.mkdir(parents=True, exist_ok=True)

    for path in sorted(inbox.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(inbox)
        if any(p.startswith((".", "~$")) for p in rel.parts):
            continue
        doc_id = rel.as_posix()
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        seen.add(doc_id)  # present on disk, so never treat as deleted

        if st.st_size > max_file_bytes:
            result.skipped.append((doc_id, "file too large"))
            continue
        # Only skip genuinely fresh files; a future mtime (clock skew) must not be skipped forever.
        if 0 <= now - st.st_mtime < settle_seconds:
            result.skipped.append((doc_id, "still being written"))
            continue

        rec = manifest.get(doc_id)
        status = rec["status"] if rec else None
        same_stat = bool(rec) and rec["size"] == st.st_size and abs(rec["mtime"] - st.st_mtime) < 1e-6

        if same_stat and status == "ready":
            result.unchanged.append(doc_id)
            continue
        if same_stat and status == "failed" and not retry_failed:
            result.unchanged.append(doc_id)
            continue

        info = FileInfo(doc_id, path, st.st_size, st.st_mtime, sha256_file(path))
        if rec is None:
            result.new.append(info)
        elif rec["sha256"] == info.sha256 and status == "ready":
            result.touched.append(info)
            result.unchanged.append(doc_id)
        elif rec["sha256"] == info.sha256 and status == "failed" and not retry_failed:
            result.touched.append(info)
            result.unchanged.append(doc_id)
        else:  # content differs, a previous run crashed ("processing"), or a failed doc is being retried
            result.changed.append(info)

    result.deleted = sorted(d for d in manifest if d not in seen)
    return result
