from __future__ import annotations

from typing import cast

from rag_app.config import IndexConfiguration
from rag_app.providers.models import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)
from rag_app.providers.protocols import (
    EmbeddingProvider,
    GenerationProvider,
    StreamingGenerationProvider,
    TextSink,
)


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

    async def generate_stream(
        self, request: GenerationInput, on_text: TextSink
    ) -> GenerationOutput:
        if callable(getattr(self._generator, "generate_stream", None)):
            return await cast(StreamingGenerationProvider, self._generator).generate_stream(
                request, on_text
            )
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
        if self._generator is cast(object, self._embedder):
            await self._generator.close()
            return
        try:
            await self._generator.close()
        finally:
            await self._embedder.close()
