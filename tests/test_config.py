import pytest
from pydantic import SecretStr, ValidationError

from mmrag.config import Settings


def test_groq_settings_select_groq_model_and_key():
    cfg = Settings(_env_file=None, llm_provider="groq", groq_api_key="groq-test-key")

    assert cfg.active_api_key == SecretStr("groq-test-key")
    assert cfg.effective_llm_model == "qwen/qwen3.8-27b"
    assert cfg.canary_token.startswith("cnry-")


def test_groq_settings_allow_model_override():
    cfg = Settings(_env_file=None, llm_provider="groq", groq_api_key="groq-test-key",
                   llm_model="custom/vision-model")

    assert cfg.effective_llm_model == "custom/vision-model"


def test_groq_settings_require_groq_key():
    with pytest.raises(ValidationError, match="GROQ_API_KEY is required"):
        Settings(_env_file=None, llm_provider="groq")


def test_openai_settings_still_require_openai_key():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is required"):
        Settings(_env_file=None)
