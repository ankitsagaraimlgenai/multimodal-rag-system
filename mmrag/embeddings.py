"""CLIP embeddings: text and images share one vector space (as in the original notebook)."""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class ClipEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        self.model_name = model_name
        self.device_pref = device
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._model = self._processor = self._torch = None
        self._device = "cpu"

    # -- lifecycle -----------------------------------------------------------------
    def warmup(self) -> None:
        self._load()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = self.device_pref
            if device == "auto":
                device = ("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                          else "cpu")
            log.info("loading CLIP", extra={"model": self.model_name, "device": device})
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(device).eval()
            self._torch, self._device = torch, device

    # -- API -------------------------------------------------------------------------
    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.config.projection_dim)

    def count_tokens(self, text: str) -> int:
        self._load()
        return len(self._processor.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self._load()
        out = []
        for i in range(0, len(texts), self.batch_size):
            inputs = self._processor(text=texts[i:i + self.batch_size], return_tensors="pt",
                                     padding=True, truncation=True, max_length=77).to(self._device)
            with self._torch.inference_mode():
                feats = self._model.get_text_features(**inputs)
            out.append(self._normalise(feats))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_images(self, images: list) -> np.ndarray:
        self._load()
        out = []
        for i in range(0, len(images), self.batch_size):
            inputs = self._processor(images=images[i:i + self.batch_size],
                                     return_tensors="pt").to(self._device)
            with self._torch.inference_mode():
                feats = self._model.get_image_features(**inputs)
            out.append(self._normalise(feats))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def _normalise(self, feats) -> np.ndarray:
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().astype(np.float32)
