from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from rag_app.config import IndexConfiguration
from rag_app.providers.models import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)

TextSink = Callable[[str], Awaitable[None]]


class StreamingGenerationProvider(Protocol):
    async def generate_stream(
        self, request: GenerationInput, on_text: TextSink
    ) -> GenerationOutput: ...


class LLMProvider(Protocol):
    async def generate(self, request: GenerationInput) -> GenerationOutput: ...

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput: ...


class GenerationProvider(Protocol):
    async def generate(self, request: GenerationInput) -> GenerationOutput: ...

    async def check_generation_readiness(self) -> None: ...

    async def close(self) -> None: ...


class EmbeddingProvider(Protocol):
    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput: ...

    async def check_embedding_readiness(self, config: IndexConfiguration) -> None: ...

    async def close(self) -> None: ...


class ProviderAdapter(LLMProvider, Protocol):
    """Composed application port used by chat, ingestion, and readiness checks."""

    async def check_readiness(
        self,
        config: IndexConfiguration,
        *,
        require_generation: bool = True,
    ) -> None: ...

    async def close(self) -> None: ...
