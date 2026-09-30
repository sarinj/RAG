from __future__ import annotations

from rag_app.config import ProviderConnection, Settings
from rag_app.providers.claude import ClaudeProvider
from rag_app.providers.composite import CompositeProvider
from rag_app.providers.gemini import GeminiProvider
from rag_app.providers.huggingface import HuggingFaceProvider
from rag_app.providers.ollama import OllamaProvider
from rag_app.providers.openai import OpenAIProvider
from rag_app.providers.protocols import EmbeddingProvider, GenerationProvider, ProviderAdapter


def create_provider(
    settings: Settings,
    *,
    embedding_dimension: int | None = None,
) -> ProviderAdapter:
    generation_connection = settings.generation_connection
    embedding_connection = settings.embedding_connection
    generator = _create_generation_provider(settings, generation_connection)
    embedder = _create_embedding_provider(
        settings,
        embedding_connection,
        embedding_dimension=embedding_dimension,
    )
    return CompositeProvider(generator=generator, embedder=embedder)


def _create_generation_provider(
    settings: Settings,
    connection: ProviderConnection,
) -> GenerationProvider:
    if connection.provider == "ollama":
        return OllamaProvider(settings, connection)
    if connection.provider == "openai":
        return OpenAIProvider(settings, connection)
    if connection.provider == "claude":
        return ClaudeProvider(settings, connection)
    if connection.provider == "gemini":
        return GeminiProvider(settings, connection)
    if connection.provider == "huggingface":
        return HuggingFaceProvider(settings, connection)
    raise ValueError(f"Unsupported generation provider: {connection.provider}")


def _create_embedding_provider(
    settings: Settings,
    connection: ProviderConnection,
    *,
    embedding_dimension: int | None,
) -> EmbeddingProvider:
    if connection.provider == "ollama":
        return OllamaProvider(settings, connection)
    if connection.provider == "openai":
        return OpenAIProvider(settings, connection, embedding_dimension=embedding_dimension)
    if connection.provider == "gemini":
        return GeminiProvider(settings, connection)
    if connection.provider == "huggingface":
        return HuggingFaceProvider(settings, connection)
    raise ValueError(f"Unsupported embedding provider: {connection.provider}")
