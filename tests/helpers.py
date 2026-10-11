"""Test doubles: a deterministic bag-of-words 'embedder' and a fake PDF loader (no torch / fitz needed)."""
from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mmrag.loaders.pdf import ImageAsset, PageContent

DIM = 64
IMAGE_WORD = "zzimage"


def _word_vec(word: str) -> np.ndarray:
    seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


class FakeEmbedder:
    loaded = True

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\w+", text))

    def embed_texts(self, texts):
        out = []
        for t in texts:
            v = sum((_word_vec(w) for w in re.findall(r"\w+", t.lower())), np.zeros(DIM, np.float32))
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.asarray(out, dtype=np.float32).reshape(-1, DIM)

    def embed_images(self, images):  # every image "looks like" the word zzimage
        return np.vstack([self.embed_texts([IMAGE_WORD])[0] for _ in images]) if images else np.zeros((0, DIM), np.float32)


def make_cfg(tmp, **over):
    base = dict(
        data_dir=tmp, inbox_dir=tmp / "inbox", images_dir=tmp / "images", db_path=tmp / "rag.db",
        chunk_size_tokens=12, chunk_overlap_tokens=3, context_window_chunks=1,
        min_image_px=16, max_image_dim=256, max_pages=100, max_images_per_doc=10,
        max_file_bytes=10_000_000, scan_settle_seconds=0.0, injection_threshold=0.6,
        clip_model_name="fake", min_score_text=0.2, min_score_image=0.2,
        top_k_text=3, top_k_images=1)
    base.update(over)
    return SimpleNamespace(**base)


def fake_loader_factory(pages_by_name: dict):
    """pages_by_name: {filename: [(text, n_images), ...]}"""
    def loader(path, cfg):
        for i, (text, n_img) in enumerate(pages_by_name[path.name], start=1):
            imgs = [ImageAsset(j, Image.new("RGB", (64, 64), (200, 30 * j, 30))) for j in range(n_img)]
            yield PageContent(i, text, imgs)
    return loader
