"""Plain data objects shared across the app."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    """One retrieved piece of evidence (a merged text window, or one image)."""
    label: str            # S1, S2... the LLM cites these
    modality: str         # "text" | "image"
    doc_id: str           # path relative to the inbox
    page: int             # 1-based
    text: str             # empty for images
    image_path: str | None
    score: float
    chunk_id: int

    @property
    def title(self) -> str:
        return f"{self.label} · {self.doc_id} p.{self.page}"


@dataclass
class RetrievalResult:
    texts: list[Passage] = field(default_factory=list)
    images: list[Passage] = field(default_factory=list)

    @property
    def all(self) -> list[Passage]:
        return [*self.texts, *self.images]

    @property
    def empty(self) -> bool:
        return not (self.texts or self.images)


# ---- events streamed by RagService.ask() -------------------------------------
@dataclass
class Retrieved:
    result: RetrievalResult


@dataclass
class Token:
    text: str


@dataclass
class Final:
    answer: str
    sources: list[Passage]
    usage: dict
    flags: list[str] = field(default_factory=list)


@dataclass
class Blocked:
    code: str
    message: str


@dataclass
class RagAnswer:
    """Non-streaming result, used by evaluation."""
    answer: str
    contexts: list[str]
    passages: list[Passage]
    blocked: str | None = None
