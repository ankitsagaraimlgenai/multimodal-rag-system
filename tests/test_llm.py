import asyncio
from types import SimpleNamespace

from mmrag.llm import LLM


class FakeStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class FakeCompletions:
    def __init__(self):
        self.request = None

    async def create(self, **request):
        self.request = request
        return FakeStream()


def test_groq_request_uses_selected_model_without_openai_cache_hint():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    cfg = SimpleNamespace(
        effective_llm_model="qwen/qwen3.8-27b",
        llm_provider="groq",
        llm_temperature=0.1,
        llm_max_output_tokens=800,
    )

    async def consume():
        return [event async for event in LLM(client, cfg).stream([])]

    assert asyncio.run(consume()) == [("usage", {})]
    assert completions.request["model"] == "qwen/qwen3.8-27b"
    assert "prompt_cache_key" not in completions.request
