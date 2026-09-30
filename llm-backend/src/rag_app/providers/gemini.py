from __future__ import annotations

import asyncio
from typing import Any, cast
from urllib.parse import quote

import httpx

from rag_app.config import IndexConfiguration, ProviderConnection, Settings
from rag_app.errors import (
    ProviderError,
    ProviderModelMissing,
    ProviderProtocolError,
    ProviderTimeout,
)
from rag_app.providers.models import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)

MAX_BATCH_EMBED_REQUESTS = 100


class GeminiProvider:
    """Google Gemini generateContent and embedding adapter."""

    def __init__(
        self,
        settings: Settings,
        connection: ProviderConnection | None = None,
    ) -> None:
        self._settings = settings
        active_connection = connection or settings.generation_connection
        headers: dict[str, str] = {}
        if active_connection.api_key:
            headers["x-goog-api-key"] = active_connection.api_key
        self._client = httpx.AsyncClient(
            base_url=f"{active_connection.base_url}/",
            headers=headers,
            timeout=httpx.Timeout(
                connect=settings.provider_connect_timeout_seconds,
                read=settings.provider_read_timeout_seconds,
                write=settings.provider_write_timeout_seconds,
                pool=settings.provider_pool_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=(
                    settings.provider_generation_concurrency
                    + settings.provider_embedding_concurrency
                ),
                max_keepalive_connections=(
                    settings.provider_generation_concurrency
                    + settings.provider_embedding_concurrency
                ),
            ),
        )
        self._generation_gate = asyncio.Semaphore(settings.provider_generation_concurrency)
        self._embedding_gate = asyncio.Semaphore(settings.provider_embedding_concurrency)

    async def close(self) -> None:
        await self._client.aclose()

    async def check_generation_readiness(self) -> None:
        model = self._settings.generation_model
        if not model:
            raise ProviderModelMissing("Generation model is not configured")
        self._require_api_key()
        available = await self._list_models()
        if "generateContent" not in available.get(self._model_name(model), set()):
            raise ProviderModelMissing("Configured Gemini generation model is not available")

    async def check_embedding_readiness(self, config: IndexConfiguration) -> None:
        model = self._settings.embedding_model
        if not model:
            raise ProviderModelMissing("Embedding model is not configured")
        if model != config.embedding_model:
            raise ProviderProtocolError("Embedding model differs from the reviewed index config")
        self._require_embedding_digest(config)
        available = await self._list_models()
        if "embedContent" not in available.get(self._model_name(model), set()):
            raise ProviderModelMissing("Configured Gemini embedding model is not available")
        embedding = await self.embed(EmbeddingInput(texts=["readiness dimension probe"]))
        if embedding.dimension != config.embedding_dimension:
            raise ProviderProtocolError(
                "Live embedding dimension differs from the reviewed index config"
            )

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        model = self._settings.generation_model
        if not model:
            raise ProviderModelMissing("Generation model is not configured")
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {"maxOutputTokens": request.max_output_tokens},
        }
        if request.temperature is not None:
            body["generationConfig"]["temperature"] = request.temperature
        if request.response_format == "json":
            body["generationConfig"]["responseMimeType"] = "application/json"

        timeout = httpx.Timeout(
            connect=self._settings.provider_connect_timeout_seconds,
            read=self._settings.generation_timeout_seconds,
            write=self._settings.provider_write_timeout_seconds,
            pool=self._settings.provider_pool_timeout_seconds,
        )
        path = f"models/{quote(self._model_name(model), safe='')}:generateContent"
        async with self._generation_gate:
            payload = await self._request("POST", path, timeout=timeout, json=body)
        try:
            candidates = payload["candidates"]
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("generation response has no candidates")
            candidate = candidates[0]
            parts = candidate["content"]["parts"]
            response = "".join(
                part["text"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            if not response:
                raise ValueError("generation response has no text")
            usage = payload.get("usageMetadata", {})
            usage = usage if isinstance(usage, dict) else {}
            return GenerationOutput(
                model=model,
                response=response,
                done=True,
                done_reason=candidate.get("finishReason"),
                prompt_eval_count=usage.get("promptTokenCount"),
                eval_count=usage.get("candidatesTokenCount"),
            )
        except Exception as exc:
            raise ProviderProtocolError("Gemini returned an invalid generation response") from exc

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput:
        model = self._settings.embedding_model
        if not model:
            raise ProviderModelMissing("Embedding model is not configured")
        vectors: list[list[float]] = []
        timeout = httpx.Timeout(
            connect=self._settings.provider_connect_timeout_seconds,
            read=self._settings.embedding_timeout_seconds,
            write=self._settings.provider_write_timeout_seconds,
            pool=self._settings.provider_pool_timeout_seconds,
        )
        path = f"models/{quote(self._model_name(model), safe='')}:batchEmbedContents"
        async with self._embedding_gate:
            for start in range(0, len(request.texts), MAX_BATCH_EMBED_REQUESTS):
                texts = request.texts[start : start + MAX_BATCH_EMBED_REQUESTS]
                body = {
                    "requests": [
                        {
                            "model": f"models/{self._model_name(model)}",
                            "content": {"parts": [{"text": text}]},
                        }
                        for text in texts
                    ]
                }
                payload = await self._request("POST", path, timeout=timeout, json=body)
                vectors.extend(self._parse_embedding_batch(payload, len(texts)))
        if not vectors:
            raise ProviderProtocolError("Gemini returned no embeddings")
        return EmbeddingOutput(model=model, vectors=vectors, dimension=len(vectors[0]))

    @staticmethod
    def _parse_embedding_batch(payload: dict[str, Any], expected_count: int) -> list[list[float]]:
        try:
            embeddings = payload["embeddings"]
            if not isinstance(embeddings, list) or len(embeddings) != expected_count:
                raise ValueError("embedding count differs from input count")
            vectors = [item["values"] for item in embeddings if isinstance(item, dict)]
            if len(vectors) != expected_count or any(
                not isinstance(item, list) for item in vectors
            ):
                raise ValueError("embedding values are invalid")
            return cast(list[list[float]], vectors)
        except Exception as exc:
            raise ProviderProtocolError("Gemini returned invalid embeddings") from exc

    async def _list_models(self) -> dict[str, set[str]]:
        payload = await self._request("GET", "models", params={"pageSize": 1000})
        try:
            models = payload["models"]
            if not isinstance(models, list):
                raise ValueError("model list is invalid")
            result: dict[str, set[str]] = {}
            for item in models:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise ValueError("model entry is invalid")
                name = self._model_name(item["name"])
                methods = item.get("supportedGenerationMethods", [])
                if not isinstance(methods, list) or any(
                    not isinstance(method, str) for method in methods
                ):
                    raise ValueError("model methods are invalid")
                result[name] = set(methods)
            return result
        except Exception as exc:
            raise ProviderProtocolError("Gemini returned an invalid model list") from exc

    def _require_api_key(self) -> None:
        if not self._client.headers.get("x-goog-api-key"):
            raise ProviderError("Gemini API key is not configured")

    def _require_embedding_digest(self, config: IndexConfiguration) -> None:
        digest = self._settings.embedding_model_digest
        if not digest:
            raise ProviderProtocolError(
                "EMBEDDING_MODEL_DIGEST must pin Gemini's embedding model"
            )
        if digest.lower() != config.embedding_digest.lower():
            raise ProviderProtocolError(
                "Embedding model digest differs from the reviewed index config"
            )

    @staticmethod
    def _model_name(value: str) -> str:
        return value.removeprefix("models/")

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            if response.status_code == 404:
                raise ProviderModelMissing("Configured Gemini model or endpoint is unavailable")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderProtocolError("Gemini returned an invalid response")
            return cast(dict[str, Any], payload)
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Gemini request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Gemini is unavailable or returned an invalid response") from exc
