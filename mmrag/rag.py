"""RagService: the single entry point for answering a question.

   rate limit -> sanitize -> injection check -> moderation -> retrieve -> (no context? refuse)
   -> build cache-friendly prompt -> stream LLM (watch for leaks) -> output guard -> sources
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import AsyncIterator, Sequence

from mmrag.guardrails import LEAK_MESSAGE, GuardrailViolation, InputGuard, OutputGuard, RateLimiter
from mmrag.llm import LLM
from mmrag.models import Blocked, Final, RagAnswer, Retrieved, Token
from mmrag.prompts import NO_CONTEXT_MESSAGE, build_messages, cited_labels
from mmrag.retrieval import Retriever

log = logging.getLogger(__name__)
Event = Retrieved | Token | Final | Blocked


class RagService:
    def __init__(self, *, cfg, retriever: Retriever, llm: LLM, input_guard: InputGuard,
                 output_guard: OutputGuard, rate_limiter: RateLimiter, system_prompt: str):
        self.cfg, self.retriever, self.llm = cfg, retriever, llm
        self.input_guard, self.output_guard = input_guard, output_guard
        self.rate_limiter, self.system_prompt = rate_limiter, system_prompt

    async def ask(self, query: str, *, session_id: str, history: Sequence[tuple[str, str]] = (),
                  enforce_rate_limit: bool = True) -> AsyncIterator[Event]:
        t0 = time.perf_counter()
        qid = hashlib.sha256(query.encode()).hexdigest()[:10]  # log a hash, never the raw question

        # 1) guardrails on the way in
        try:
            if enforce_rate_limit:
                self.rate_limiter.check(session_id)
            query = self.input_guard.sanitize(query)
            await self.input_guard.moderate(query)
        except GuardrailViolation as v:
            log.warning("request blocked", extra={"code": v.code, "query_id": qid})
            yield Blocked(v.code, v.user_message)
            return

        # 2) retrieve (CPU-bound embedding -> worker thread)
        try:
            result = await asyncio.to_thread(self.retriever.retrieve, query)
        except Exception:
            log.exception("retrieval failed", extra={"query_id": qid})
            yield Blocked("retrieval_error", "Search is temporarily unavailable. Please try again.")
            return
        yield Retrieved(result)

        # 3) grounding guard: nothing relevant -> refuse without spending an LLM call
        if result.empty:
            yield Final(NO_CONTEXT_MESSAGE, [], {}, ["no_context"])
            return

        # 4) generate
        history = list(history)[-self.cfg.history_turns:] if self.cfg.history_turns else []
        messages = build_messages(self.system_prompt, history, query, result,
                                  image_detail=(self.cfg.image_detail
                                                if getattr(self.cfg, "llm_provider", "openai") == "openai"
                                                else None))
        parts: list[str] = []
        tail, usage, leaked = "", {}, False
        keep = len(self.output_guard.canary) + 64
        try:
            async for kind, value in self.llm.stream(messages):
                if kind == "token":
                    parts.append(value)
                    tail = (tail + value)[-keep:]
                    if self.output_guard.leaked(tail):
                        leaked = True
                        break
                    yield Token(value)
                else:
                    usage = value
        except Exception:
            log.exception("llm call failed", extra={"query_id": qid})
            yield Blocked("llm_error", "The answer service is temporarily unavailable. Please try again.")
            return

        # 5) guardrails on the way out
        if leaked:
            answer, flags = LEAK_MESSAGE, ["canary_leak"]
        else:
            answer, flags = self.output_guard.check("".join(parts))
        labels = cited_labels(answer)
        sources = [p for p in result.all if p.label in labels]

        log.info("answered", extra={
            "query_id": qid, "latency_s": round(time.perf_counter() - t0, 2),
            "text_passages": len(result.texts), "images": len(result.images),
            "flags": flags, **usage})
        yield Final(answer, sources, usage, flags)

    async def answer_once(self, query: str, *, session_id: str = "eval",
                          enforce_rate_limit: bool = False) -> RagAnswer:
        """Non-streaming convenience used by evaluation scripts and tests."""
        passages, contexts, answer, blocked = [], [], "", None
        async for ev in self.ask(query, session_id=session_id, enforce_rate_limit=enforce_rate_limit):
            if isinstance(ev, Retrieved):
                passages = ev.result.all
                contexts = [p.text if p.modality == "text" else f"[Image from {p.doc_id}, page {p.page}]"
                            for p in passages]
            elif isinstance(ev, Final):
                answer = ev.answer
            elif isinstance(ev, Blocked):
                blocked, answer = ev.code, ev.message
        return RagAnswer(answer, contexts, passages, blocked)
