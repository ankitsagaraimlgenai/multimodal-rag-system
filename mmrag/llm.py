"""Streaming chat-completion wrapper with prompt-cache hints and usage reporting."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI
    from mmrag.config import Settings

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, client: "AsyncOpenAI", cfg: "Settings"):
        self.client, self.cfg = client, cfg

    async def stream(self, messages: list[dict]) -> AsyncIterator[tuple[str, object]]:
        """Yields ("token", str) ... then ("usage", dict)."""
        cfg = self.cfg
        request = {
            "model": cfg.effective_llm_model,
            "messages": messages,
            "temperature": cfg.llm_temperature,
            "max_completion_tokens": cfg.llm_max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if cfg.llm_provider == "openai":
            # Same key for every request sharing the static prefix -> same cache shard -> more hits.
            request["prompt_cache_key"] = cfg.prompt_cache_key
        stream = await self.client.chat.completions.create(**request)
        usage: dict = {}
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield "token", chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):
                u = chunk.usage
                details = getattr(u, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) or 0
                usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                         "cached_tokens": cached,
                         "cache_hit_ratio": round(cached / u.prompt_tokens, 3) if u.prompt_tokens else 0.0}
        yield "usage", usage
