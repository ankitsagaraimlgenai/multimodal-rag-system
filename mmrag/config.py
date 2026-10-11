"""Central configuration. Every knob is an environment variable (or a line in .env)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["openai", "groq"] = "openai"
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 800
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    image_detail: Literal["low", "high", "auto"] = "auto"
    history_turns: int = Field(4, ge=0, le=20)
    # Shared by every request so OpenAI routes them to the same cache (see mmrag/llm.py).
    prompt_cache_key: str = "mmrag-v1"

    # --- Embeddings (CLIP embeds text AND images into one space) --------------
    clip_model_name: str = "openai/clip-vit-base-patch32"
    embedding_device: str = "auto"  # auto | cpu | cuda | mps
    embedding_batch_size: int = 32

    # --- Chunking (measured in CLIP tokens; CLIP's hard limit is 77 incl. BOS/EOS)
    chunk_size_tokens: int = Field(60, ge=16, le=75)
    chunk_overlap_tokens: int = Field(12, ge=0)
    # Small-to-big: embed small chunks, but give the LLM +/- N neighbouring chunks.
    context_window_chunks: int = Field(2, ge=0, le=6)

    # --- Document loading -------------------------------------------------------
    min_image_px: int = 128      # skip icons / bullets / logos smaller than this
    max_image_dim: int = 1024    # downscale before storing + sending to the LLM
    max_pages: int = 500
    max_images_per_doc: int = 200
    max_file_mb: int = 50

    # --- Storage / ingestion ----------------------------------------------------
    data_dir: Path = Path("data")
    scan_interval_seconds: int = 60   # 0 = scan once at startup only
    scan_settle_seconds: float = 2.0  # ignore files modified in the last N seconds
    allow_uploads: bool = True

    # --- Retrieval ----------------------------------------------------------------
    top_k_text: int = 5
    top_k_images: int = 2
    # CLIP text<->text cosine runs higher than text<->image, so thresholds differ.
    min_score_text: float = 0.25
    min_score_image: float = 0.18

    # --- Guardrails ---------------------------------------------------------------
    max_query_chars: int = 1000
    moderation_enabled: bool = True
    rate_limit_per_minute: int = 20
    redact_contact_info: bool = False  # emails / phones in answers (cards + SSNs are always redacted)
    injection_threshold: float = 0.6

    # --- UI auth (optional). Set both + CHAINLIT_AUTH_SECRET to enable. -----------
    app_username: str | None = None
    app_password: SecretStr | None = None

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.llm_provider == "openai" and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "groq" and self.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        return self

    @property
    def active_api_key(self) -> SecretStr:
        key = self.groq_api_key if self.llm_provider == "groq" else self.openai_api_key
        if key is None:
            variable = "GROQ_API_KEY" if self.llm_provider == "groq" else "OPENAI_API_KEY"
            raise ValueError(f"{variable} is required when LLM_PROVIDER={self.llm_provider}")
        return key

    @property
    def effective_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return "qwen/qwen3.8-27b" if self.llm_provider == "groq" else "gpt-4.1"

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def canary_token(self) -> str:
        """Stable per-deployment secret planted in the system prompt to detect prompt leaks."""
        digest = hashlib.sha256(("canary:" + self.active_api_key.get_secret_value()).encode())
        return "cnry-" + digest.hexdigest()[:12]

    @property
    def auth_enabled(self) -> bool:
        return bool(self.app_username and self.app_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
