"""Ingest from the command line.

    python scripts/ingest.py                 # scan the inbox once
    python scripts/ingest.py --watch         # keep scanning (no server needed)
    python scripts/ingest.py --retry-failed  # retry documents that failed earlier
    python scripts/ingest.py --status        # show what is indexed
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmrag.config import get_settings  # noqa: E402
from mmrag.container import Container  # noqa: E402
from mmrag.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    cfg = get_settings()
    setup_logging(cfg.log_level)
    c = Container(cfg)
    if args.status:
        for doc_id, r in sorted(c.db.manifest().items()):
            print(f"{r['status']:<10} {doc_id}  pages={r['n_pages']} chunks={r['n_text_chunks']} "
                  f"images={r['n_images']} quarantined={r['n_quarantined']} {r['error'] or ''}")
        print(c.db.stats())
        return 0

    c.prepare()
    while True:
        report = c.pipeline.run(retry_failed=args.retry_failed)
        print(report.summary())
        for doc, err in report.failed:
            print(f"  FAILED {doc}: {err}")
        if not args.watch:
            return 1 if report.failed else 0
        time.sleep(max(cfg.scan_interval_seconds, 5))


if __name__ == "__main__":
    sys.exit(main())
