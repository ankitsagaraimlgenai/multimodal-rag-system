"""Document loader: PDF -> per-page text + extracted images (PyMuPDF)."""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from PIL import Image

log = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """The file cannot be read (corrupt, encrypted, too large, unsupported)."""


@dataclass
class ImageAsset:
    index: int
    image: Image.Image


@dataclass
class PageContent:
    number: int                      # 1-based
    text: str
    images: list[ImageAsset] = field(default_factory=list)


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_WRAP = re.compile(r"(?<![.!?:;])\n(?!\n)")
_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Undo PDF hard-wrapping so sentences stay intact for chunking."""
    text = text.replace("\r", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_WRAP.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))  # transparent -> white, not black
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def _extract_image(doc, xref: int, min_px: int, max_dim: int) -> Image.Image | None:
    try:
        base = doc.extract_image(xref)
        if not base:
            return None
        img = Image.open(io.BytesIO(base["image"]))
        img.load()
        if min(img.size) < min_px:
            return None
        img = _to_rgb(img)
        img.thumbnail((max_dim, max_dim))
        return img
    except Exception as exc:  # one bad image must not sink the document
        log.warning("skipping unreadable image", extra={"xref": xref, "error": str(exc)})
        return None


def load_pdf(path: Path, cfg) -> Iterator[PageContent]:
    import fitz  # PyMuPDF; imported lazily so tests / tooling don't need it

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise DocumentLoadError(f"cannot open PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise DocumentLoadError("PDF is password protected")
        if doc.page_count > cfg.max_pages:
            raise DocumentLoadError(f"PDF has {doc.page_count} pages (limit {cfg.max_pages})")
        seen_xrefs: set[int] = set()   # a logo repeated on every page is embedded once
        total_images = 0
        for number, page in enumerate(doc, start=1):
            images: list[ImageAsset] = []
            for idx, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                if xref in seen_xrefs or total_images >= cfg.max_images_per_doc:
                    continue
                seen_xrefs.add(xref)
                img = _extract_image(doc, xref, cfg.min_image_px, cfg.max_image_dim)
                if img is not None:
                    images.append(ImageAsset(idx, img))
                    total_images += 1
            yield PageContent(number, clean_text(page.get_text("text")), images)
    finally:
        doc.close()


def load_document(path: Path, cfg) -> Iterator[PageContent]:
    """Dispatch on extension. Add loaders for .docx/.pptx/.html here."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, cfg)
    raise DocumentLoadError(f"unsupported file type: {suffix}")
