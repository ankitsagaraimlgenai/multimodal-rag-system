"""Retrieval: embed the query once, search text and image indexes separately, merge windows.

Separate quotas per modality matter: CLIP text->image similarity is systematically lower than
text->text, so a single unified top-k would almost never surface images.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from mmrag import injection
from mmrag.db import Database
from mmrag.models import Passage, RetrievalResult
from mmrag.store import VectorStore

if TYPE_CHECKING:  # pragma: no cover
    from mmrag.config import Settings
    from mmrag.embeddings import ClipEmbedder

log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, cfg: "Settings", db: Database, store: VectorStore, embedder: "ClipEmbedder"):
        self.cfg, self.db, self.store, self.embedder = cfg, db, store, embedder

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.cfg
        qvec = self.embedder.embed_texts([query])[0]
        text_hits = self.store.search("text", qvec, cfg.top_k_text, cfg.min_score_text)
        image_hits = self.store.search("image", qvec, cfg.top_k_images, cfg.min_score_image)
        rows = self.db.get_chunks([cid for cid, _ in text_hits + image_hits])

        texts = self._merge_text(text_hits, rows)
        images = [
            Passage("", "image", r["doc_id"], r["page"], "", r["image_path"], score, cid)
            for cid, score in image_hits if (r := rows.get(cid))
        ]
        # Labels (S1, S2, ...) are what the LLM cites and the UI links back to.
        texts = [_relabel(p, f"S{i}") for i, p in enumerate(texts, 1)]
        images = [_relabel(p, f"S{i}") for i, p in enumerate(images, len(texts) + 1)]
        log.info("retrieved", extra={"text": len(texts), "images": len(images)})
        return RetrievalResult(texts, images)

    def _merge_text(self, hits: list[tuple[int, float]], rows: dict[int, dict]) -> list[Passage]:
        """Merge overlapping context windows per page so the LLM never sees the same sentence twice."""
        by_page: dict[tuple[str, int], list[tuple[int, int, float, int]]] = defaultdict(list)
        for cid, score in hits:
            r = rows.get(cid)
            if r:
                by_page[(r["doc_id"], r["page"])].append((r["ctx_start"], r["ctx_end"], score, cid))

        passages: list[Passage] = []
        for (doc_id, page), spans in by_page.items():
            page_text = self.db.get_page_text(doc_id, page)
            if not page_text:
                continue
            spans.sort()
            merged: list[list] = []
            for s, e, score, cid in spans:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                    if score > merged[-1][2]:
                        merged[-1][2], merged[-1][3] = score, cid
                else:
                    merged.append([s, e, score, cid])
            for s, e, score, cid in merged:
                text = page_text[s:e].strip()
                verdict = injection.scan(text, self.cfg.injection_threshold)
                if verdict.flagged:  # second line of defence, in case config/rules changed since ingestion
                    log.warning("retrieved passage dropped: possible prompt injection",
                                extra={"doc_id": doc_id, "page": page, "matches": verdict.matches})
                    continue
                passages.append(Passage("", "text", doc_id, page, text, None, score, cid))
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages


def _relabel(p: Passage, label: str) -> Passage:
    return Passage(label, p.modality, p.doc_id, p.page, p.text, p.image_path, p.score, p.chunk_id)
