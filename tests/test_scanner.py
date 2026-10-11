import os
import tempfile
from pathlib import Path

from mmrag.ingestion.scanner import scan_inbox


def _scan(inbox, manifest, **kw):
    return scan_inbox(inbox, manifest, max_file_bytes=1000, settle_seconds=0, **kw)


def _rec(info, status="ready"):
    return {"sha256": info.sha256, "size": info.size, "mtime": info.mtime, "status": status}


def test_scanner_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        inbox = Path(d)
        (inbox / "a.pdf").write_bytes(b"%PDF-1 a")
        (inbox / "sub").mkdir()
        (inbox / "sub" / "b.pdf").write_bytes(b"%PDF-1 b")
        (inbox / ".hidden.pdf").write_bytes(b"x")
        (inbox / "notes.txt").write_text("x")
        (inbox / "big.pdf").write_bytes(b"x" * 2000)

        r = _scan(inbox, {})
        assert sorted(f.doc_id for f in r.new) == ["a.pdf", "sub/b.pdf"]
        assert r.skipped == [("big.pdf", "file too large")]

        manifest = {f.doc_id: _rec(f) for f in r.new}
        r = _scan(inbox, manifest)
        assert not r.pending and sorted(r.unchanged) == ["a.pdf", "sub/b.pdf"]

        (inbox / "a.pdf").write_bytes(b"%PDF-1 a changed")           # modified
        (inbox / "sub" / "b.pdf").unlink()                           # deleted
        r = _scan(inbox, manifest)
        assert [f.doc_id for f in r.changed] == ["a.pdf"] and r.deleted == ["sub/b.pdf"]


def test_touch_keeps_content_and_failed_docs_need_retry_flag():
    with tempfile.TemporaryDirectory() as d:
        inbox = Path(d)
        p = inbox / "a.pdf"; p.write_bytes(b"%PDF-1 a")
        info = _scan(inbox, {}).new[0]
        os.utime(p, (info.mtime + 50, info.mtime + 50))              # same bytes, new mtime
        r = _scan(inbox, {"a.pdf": _rec(info)})
        assert not r.pending and [t.doc_id for t in r.touched] == ["a.pdf"]

        failed = {"a.pdf": _rec(_scan(inbox, {}).new[0], "failed")}
        assert not _scan(inbox, failed).pending                       # don't retry corrupt files forever
        assert [f.doc_id for f in _scan(inbox, failed, retry_failed=True).changed] == ["a.pdf"]
        crashed = {"a.pdf": _rec(_scan(inbox, {}).new[0], "processing")}
        assert [f.doc_id for f in _scan(inbox, crashed).changed] == ["a.pdf"]   # crash recovery


def test_files_still_being_written_are_skipped():
    with tempfile.TemporaryDirectory() as d:
        inbox = Path(d); (inbox / "a.pdf").write_bytes(b"%PDF-1")
        r = scan_inbox(inbox, {}, max_file_bytes=1000, settle_seconds=60)
        assert not r.new and r.skipped == [("a.pdf", "still being written")]
