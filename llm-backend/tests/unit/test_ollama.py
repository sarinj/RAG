from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from rag_app.config import Settings
from rag_app.errors import ProviderProtocolError
from rag_app.providers.ollama import (
    EmbeddingInput,
    GenerationInput,
    OllamaProvider,
    find_context_window,
    find_model,
)


@pytest.mark.asyncio
async def test_embedding_adapter_returns_finite_vectors_and_dimension() -> None:
    with respx.mock(base_url="http://localhost:11434") as router:
        route = router.post("/api/embed").mock(
            return_value=Response(
                200,
                json={
                    "model": "embedding-test:latest",
                    "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                },
            )
        )
        provider = OllamaProvider(
            Settings(
                _env_file=None,
                ollama_base_url="http://localhost:11434",
                embedding_model="embedding-test:latest",
            )
        )
        try:
            result = await provider.embed(EmbeddingInput(texts=["first text", "second text"]))
        finally:
            await provider.close()

    request_body = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "embedding-test:latest",
        "input": ["first text", "second text"],
    }
    assert result.dimension == 3
    assert len(result.vectors) == 2


@pytest.mark.asyncio
async def test_embedding_adapter_rejects_non_finite_values() -> None:
    with respx.mock(base_url="http://localhost:11434") as router:
        router.post("/api/embed").mock(
            return_value=Response(
                200,
                content=(b'{"model":"embedding-test:latest","embeddings":[[1e999]]}'),
            )
        )
        provider = OllamaProvider(
            Settings(
                _env_file=None,
                ollama_base_url="http://localhost:11434",
                embedding_model="embedding-test:latest",
            )
        )
        try:
            with pytest.raises(ProviderProtocolError):
                await provider.embed(EmbeddingInput(texts=["text"]))
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_generation_adapter_sends_bounded_non_streaming_request() -> None:
    with respx.mock(base_url="http://localhost:11434") as router:
        route = router.post("/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "model": "generation-test:latest",
                    "response": "Hello.",
                    "done": True,
                    "total_duration": 123,
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                },
            )
        )
        provider = OllamaProvider(
            Settings(
                _env_file=None,
                ollama_base_url="http://localhost:11434",
                generation_model="generation-test:latest",
            )
        )
        try:
            result = await provider.generate(
                GenerationInput(
                    prompt="Say hello.",
                    context_window_tokens=2048,
                    max_output_tokens=32,
                    temperature=0,
                    think=False,
                    response_format="json",
                )
            )
        finally:
            await provider.close()

    request_body = json.loads(route.calls[0].request.content)
    assert request_body["stream"] is False
    assert request_body["think"] is False
    assert request_body["format"] == "json"
    assert request_body["options"] == {
        "num_ctx": 2048,
        "num_predict": 32,
        "temperature": 0,
    }
    assert result.response == "Hello."


def test_model_lookup_handles_default_tags_and_finds_context_length() -> None:
    from rag_app.providers.ollama import OllamaModelTag

    model = OllamaModelTag(name="embedding-test:latest", digest="a" * 64)

    assert find_model([model], "embedding-test") == model
    assert find_context_window({"model": {"architecture.context_length": 8192}}) == 8192
