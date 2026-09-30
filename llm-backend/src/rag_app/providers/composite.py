from __future__ import annotations

from rag_app.config import IndexConfiguration
from rag_app.providers.models import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)
from rag_app.providers.protocols import EmbeddingProvider, GenerationProvider


class CompositeProvider:
    """Keep generation and embeddings behind one application-facing provider port."""

    def __init__(
        self,
        generator: GenerationProvider,
        embedder: EmbeddingProvider,
    ) -> None:
        self._generator = generator
        self._embedder = embedder

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        return await self._generator.generate(request)

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput:
        return await self._embedder.embed(request)

    async def check_readiness(
        self,
        config: IndexConfiguration,
        *,
        require_generation: bool = True,
    ) -> None:
        if require_generation:
            await self._generator.check_generation_readiness()
        await self._embedder.check_embedding_readiness(config)

    async def close(self) -> None:
        if self._generator is self._embedder:
            await self._generator.close()
            return
        try:
            await self._generator.close()
        finally:
            await self._embedder.close()
