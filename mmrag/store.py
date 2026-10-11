"""In-memory exact (brute-force cosine) vector search, one matrix per modality.

Embeddings are L2-normalised, so cosine similarity == dot product. This is
identical to FAISS IndexFlatIP, with no native dependency. A flat scan is
milliseconds up to a few hundred thousand chunks; beyond that, swap this class
for FAISS-HNSW / pgvector / Qdrant (the interface is just reload() + search()).
"""
from __future__ import annotations

import logging

import numpy as np

from mmrag.db import Database

log = logging.getLogger(__name__)
_EMPTY = (np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32))


class VectorStore:
    def __init__(self, db: Database):
        self._db = db
        self._index: dict[str, tuple[np.ndarray, np.ndarray]] = {"text": _EMPTY, "image": _EMPTY}

    def reload(self) -> None:
        """Rebuild from SQLite, then swap atomically so in-flight searches never see a half-built index."""
        buckets: dict[str, tuple[list[int], list[np.ndarray]]] = {"text": ([], []), "image": ([], [])}
        for cid, modality, blob in self._db.all_embeddings():
            buckets[modality][0].append(cid)
            buckets[modality][1].append(np.frombuffer(blob, dtype=np.float32))
        new = {}
        for modality, (ids, vecs) in buckets.items():
            new[modality] = (np.asarray(ids, dtype=np.int64), np.vstack(vecs)) if ids else _EMPTY
        self._index = new
        log.info("vector index loaded", extra={m: len(v[0]) for m, v in new.items()})

    def count(self, modality: str) -> int:
        return len(self._index[modality][0])

    def search(self, modality: str, query: np.ndarray, k: int, min_score: float) -> list[tuple[int, float]]:
        ids, mat = self._index[modality]
        if k <= 0 or len(ids) == 0:
            return []
        scores = mat @ query.astype(np.float32)
        k = min(k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top if scores[i] >= min_score]
