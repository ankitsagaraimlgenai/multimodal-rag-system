"""Token-aware chunking with overlap.

Why not a character splitter? CLIP silently truncates anything past 77 tokens, so
a 500-character chunk loses its tail and the vector no longer represents the chunk.
Here chunk size and overlap are measured in the embedder's own tokens.

Chunks keep their (start, end) offsets into the page text, which enables
"small-to-big" retrieval: embed small chunks, hand the LLM the surrounding window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int
    tokens: int


class TokenChunker:
    def __init__(self, count_tokens: Callable[[str], int], chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.count = count_tokens
        self.size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        units = self._units(text)
        chunks: list[Chunk] = []
        i, n = 0, len(units)
        while i < n:
            j, total = i, 0
            while j < n and (j == i or total + units[j][2] <= self.size):
                total += units[j][2]
                j += 1
            start, end = units[i][0], units[j - 1][1]
            chunks.append(Chunk(text[start:end], start, end, total))
            if j >= n:
                break
            # Walk back over trailing units to build the overlap; always advance at least one unit.
            k, carried = j, 0
            while k > i + 1 and carried + units[k - 1][2] <= self.overlap:
                carried += units[k - 1][2]
                k -= 1
            i = k
        return chunks

    # -- helpers -----------------------------------------------------------------
    def _units(self, text: str) -> list[tuple[int, int, int]]:
        """Sentence/line-sized spans (start, end, tokens); oversize spans are split on words."""
        spans, pos = [], 0
        for m in _BOUNDARY.finditer(text):
            if m.start() > pos:
                spans.append((pos, m.start()))
            pos = m.end()
        if pos < len(text):
            spans.append((pos, len(text)))
        units: list[tuple[int, int, int]] = []
        for s, e in spans:
            if not text[s:e].strip():
                continue
            t = self.count(text[s:e])
            if t <= self.size:
                units.append((s, e, t))
            else:
                units.extend(self._split_long(text, s, e))
        return units

    def _split_long(self, text: str, s: int, e: int) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        cur_s = cur_e = None
        cur_t = 0
        for m in _WORD.finditer(text, s, e):
            t = self.count(m.group())
            if cur_s is not None and cur_t + t > self.size:
                out.append((cur_s, cur_e, cur_t))
                cur_s, cur_t = None, 0
            if cur_s is None:
                cur_s = m.start()
            cur_e = m.end()
            cur_t += t
        if cur_s is not None:
            out.append((cur_s, cur_e, cur_t))
        return out


def context_windows(chunks: list[Chunk], neighbours: int) -> list[tuple[int, int]]:
    """For each chunk, the page-text span covering it plus `neighbours` chunks either side."""
    n = len(chunks)
    return [(chunks[max(0, i - neighbours)].start, chunks[min(n - 1, i + neighbours)].end)
            for i in range(n)]
