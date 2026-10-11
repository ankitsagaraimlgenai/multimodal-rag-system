"""Wires everything together once per process (shared by FastAPI, Chainlit and the CLI scripts)."""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from openai import AsyncOpenAI

from mmrag.config import Settings, get_settings
from mmrag.db import Database
from mmrag.embeddings import ClipEmbedder
from mmrag.guardrails import InputGuard, OutputGuard, RateLimiter
from mmrag.ingestion.pipeline import IngestionPipeline
from mmrag.llm import LLM
from mmrag.prompts import render_system_prompt
from mmrag.rag import RagService
from mmrag.retrieval import Retriever
from mmrag.store import VectorStore

log = logging.getLogger(__name__)


class Container:
    def __init__(self, cfg: Settings | None = None):
        cfg = self.settings = cfg or get_settings()
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.inbox_dir.mkdir(parents=True, exist_ok=True)

        self.db = Database(cfg.db_path)
        self.embedder = ClipEmbedder(cfg.clip_model_name, cfg.embedding_device, cfg.embedding_batch_size)
        self.store = VectorStore(self.db)
        self.pipeline = IngestionPipeline(cfg, self.db, self.embedder, self.store)
        self.retriever = Retriever(cfg, self.db, self.store, self.embedder)

        api_key = cfg.active_api_key.get_secret_value()
        if cfg.llm_provider == "groq":
            client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1",
                                 timeout=cfg.llm_timeout_s, max_retries=cfg.llm_max_retries)
            if cfg.moderation_enabled:
                log.warning("OpenAI moderation is unavailable with Groq; moderation is disabled")
        else:
            client = AsyncOpenAI(api_key=api_key, timeout=cfg.llm_timeout_s,
                                 max_retries=cfg.llm_max_retries)
        self.rag = RagService(
            cfg=cfg, retriever=self.retriever, llm=LLM(client, cfg),
            input_guard=InputGuard(max_chars=cfg.max_query_chars, injection_threshold=cfg.injection_threshold,
                                   moderation_enabled=cfg.moderation_enabled and cfg.llm_provider == "openai",
                                   client=client),
            output_guard=OutputGuard(cfg.canary_token, cfg.redact_contact_info),
            rate_limiter=RateLimiter(cfg.rate_limit_per_minute),
            system_prompt=render_system_prompt(cfg.canary_token))
        self.ready = False

    def prepare(self) -> None:
        """Blocking startup: load CLIP, check index compatibility, load the vector index."""
        self.embedder.warmup()
        self.pipeline.ensure_compatible()
        self.store.reload()
        self.ready = True
        log.info("ready", extra=self.db.stats())

    async def scanner_loop(self) -> None:
        """Background folder watcher: runs immediately, then every SCAN_INTERVAL_SECONDS."""
        interval = self.settings.scan_interval_seconds
        while True:
            try:
                report = await asyncio.to_thread(self.pipeline.run)
                if report.changed:
                    log.info("scan finished", extra={"summary": report.summary()})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scan failed")
            if interval <= 0:
                return
            await asyncio.sleep(interval)


@lru_cache
def get_container() -> Container:
    return Container()
