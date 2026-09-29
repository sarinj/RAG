from __future__ import annotations

from typing import Protocol

from rag_app.providers.ollama import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)


class LLMProvider(Protocol):
    async def generate(self, request: GenerationInput) -> GenerationOutput: ...

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput: ...
