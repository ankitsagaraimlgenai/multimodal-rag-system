"""Input / output guardrails: sanitising, length, moderation, injection, rate limiting, PII, leak detection."""
from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from mmrag import injection

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI

log = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    def __init__(self, code: str, user_message: str):
        super().__init__(code)
        self.code = code
        self.user_message = user_message


# ---------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding-window limiter, in-process. Use Redis if you run more than one replica."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                raise GuardrailViolation(
                    "rate_limited", "You're sending messages too quickly. Please wait a moment and try again.")
            dq.append(now)
            if len(self._hits) > 10_000:  # opportunistic cleanup
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    del self._hits[k]


# ---------------------------------------------------------------------- input
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


class InputGuard:
    def __init__(self, *, max_chars: int, injection_threshold: float,
                 moderation_enabled: bool = False, client: "AsyncOpenAI | None" = None):
        self.max_chars = max_chars
        self.threshold = injection_threshold
        self.moderation_enabled = moderation_enabled and client is not None
        self.client = client

    def sanitize(self, text: str) -> str:
        text = _CONTROL.sub("", unicodedata.normalize("NFKC", text or "")).strip()
        if not text:
            raise GuardrailViolation("empty", "Please type a question.")
        if len(text) > self.max_chars:
            raise GuardrailViolation(
                "too_long", f"That message is too long (max {self.max_chars} characters). Please shorten it.")
        verdict = injection.scan(text, self.threshold)
        if verdict.flagged:
            log.warning("prompt injection blocked in query",
                        extra={"matches": verdict.matches, "score": verdict.score})
            raise GuardrailViolation(
                "prompt_injection",
                "That message looks like an attempt to change my instructions, so I can't act on it. "
                "Please rephrase your question about the documents.")
        return text

    async def moderate(self, text: str) -> None:
        """OpenAI moderation. Fails open on API errors so an outage doesn't take chat down; change if you need fail-closed."""
        if not self.moderation_enabled:
            return
        try:
            resp = await self.client.moderations.create(model="omni-moderation-latest", input=text)
            flagged = resp.results[0].flagged
        except Exception:
            log.warning("moderation call failed; allowing request", exc_info=True)
            return
        if flagged:
            raise GuardrailViolation(
                "moderation", "I can't help with that request. Please ask something about the documents.")


# --------------------------------------------------------------------- output
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)")

LEAK_MESSAGE = "I can't share that. Is there something in the documents I can help you with?"


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str, contact: bool = False) -> tuple[str, int]:
    count = 0

    def card(m: re.Match) -> str:
        nonlocal count
        if _luhn_ok(m.group()):
            count += 1
            return "[REDACTED-CARD]"
        return m.group()

    text = _CARD.sub(card, text)
    text, n = _SSN.subn("[REDACTED-SSN]", text)
    count += n
    if contact:
        text, n1 = _EMAIL.subn("[REDACTED-EMAIL]", text)
        text, n2 = _PHONE.subn("[REDACTED-PHONE]", text)
        count += n1 + n2
    return text, count


class OutputGuard:
    def __init__(self, canary: str, redact_contact: bool = False):
        self.canary = canary
        self.redact_contact = redact_contact

    def leaked(self, text: str) -> bool:
        return self.canary in text

    def check(self, answer: str) -> tuple[str, list[str]]:
        flags: list[str] = []
        if self.leaked(answer):
            log.error("system prompt canary leaked in model output")
            return LEAK_MESSAGE, ["canary_leak"]
        answer, n = _MD_IMAGE.subn("[image removed]", answer)  # auto-fetched URLs = exfiltration channel
        if n:
            flags.append("markdown_image_removed")
        answer, n = redact_pii(answer, self.redact_contact)
        if n:
            flags.append("pii_redacted")
        return answer, flags
