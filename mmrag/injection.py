"""Prompt-injection defences.

Three layers (none is sufficient alone):
 1. Detect  - heuristic scoring of user queries AND document text (this module).
            Documents are scanned at ingestion (page quarantined) and again at retrieval.
 2. Contain - untrusted text is HTML-escaped and wrapped in <context> tags that the
            system prompt declares to be data, never instructions (mmrag/prompts.py).
 3. Verify  - a per-deployment canary in the system prompt flags leaks; markdown images
            (a classic exfiltration channel) are stripped from answers (guardrails.py).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_FLAGS = re.IGNORECASE | re.MULTILINE

# (name, regex, weight). Combined score = 1 - prod(1 - weight) over distinct matches.
_RULES: list[tuple[str, re.Pattern, float]] = [(n, re.compile(p, _FLAGS), w) for n, p, w in [
    ("ignore_previous",
     r"\b(?:ignore|disregard|forget|override|bypass)\b\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+)?"
     r"(?:previous|prior|above|earlier|preceding|original|system)\s+(?:\w+\s+)?"
     r"(?:instructions?|prompts?|rules?|guidelines?|directions?|context|polic(?:y|ies)|guardrails?)\b", 0.85),
    ("ignore_your_rules",
     r"\b(?:ignore|disregard|forget)\b\s+(?:all\s+|any\s+)?(?:of\s+)?your\s+"
     r"(?:instructions?|rules?|guidelines?|system prompt|programming|training)\b", 0.85),
    ("forget_everything",
     r"\bforget\s+(?:everything|all)\b[^.\n]{0,30}\b(?:above|before|told|said|know)\b", 0.7),
    ("reveal_prompt",
     r"\b(?:reveal|show|print|repeat|display|leak|output|disclose)\b[^.\n]{0,30}\b"
     r"(?:system|hidden|initial|original|secret|internal)\s+(?:prompt|instructions?|message)s?\b", 0.8),
    ("role_override",
     r"\b(?:you are now|from now on,? you|pretend (?:to be|you are)|roleplay as|simulate being)\b", 0.5),
    ("jailbreak_terms",
     r"\b(?:jailbreak|dan mode|developer mode|do anything now|god mode)\b", 0.8),
    ("new_instructions",
     r"\b(?:new|updated|revised|real|actual)\s+(?:instructions?|system prompt|rules?)\s*[:\-]", 0.6),
    ("control_tokens",
     r"(?:<\|?(?:im_start|im_end|system|assistant|endoftext)\|?>|\[/?inst\]|<<\/?sys>>"
     r"|^\s*#{1,3}\s*(?:system|instructions?)\b|^\s*(?:system|assistant)\s*:)", 0.7),
    ("conceal_from_user",
     r"\b(?:do not|don't|never|without)\b[^.\n]{0,20}\b(?:tell|inform|mention|reveal|notify|alert)\b"
     r"[^.\n]{0,20}\b(?:user|human|reader)\b", 0.6),
    ("exfiltration",
     r"(?:!\[[^\]]*\]\(https?://[^)]*[?&=][^)]*\)|\b(?:send|post|upload|forward|exfiltrate|email)\b"
     r"[^.\n]{0,60}\b(?:to|at)\b[^.\n]{0,20}(?:https?://|[\w.+-]+@[\w-]+\.))", 0.7),
    ("safety_override",
     r"\b(?:disable|turn off|ignore|bypass)\b[^.\n]{0,20}\b(?:safety|guardrails?|filters?|restrictions?|moderation)\b", 0.7),
]]

_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_HSPACE = re.compile(r"[ \t\r\f\v]+")


@dataclass
class Verdict:
    score: float = 0.0
    matches: list[str] = field(default_factory=list)
    flagged: bool = False


def normalize(text: str) -> str:
    """Defeat trivial obfuscation: full-width chars, zero-width chars, odd spacing."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    return _HSPACE.sub(" ", text)


def scan(text: str, threshold: float = 0.6) -> Verdict:
    norm = normalize(text)
    names, remaining = [], 1.0
    for name, rx, weight in _RULES:
        if rx.search(norm):
            names.append(name)
            remaining *= 1.0 - weight
    score = round(1.0 - remaining, 3)
    return Verdict(score=score, matches=names, flagged=score >= threshold)
