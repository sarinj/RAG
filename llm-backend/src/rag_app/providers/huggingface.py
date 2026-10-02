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


class HuggingFaceProvider:
    """Hugging Face Inference Providers chat and feature-extraction adapter."""

    def __init__(
        self,
        settings: Settings,
        connection: ProviderConnection | None = None,
    ) -> None:
        self._settings = settings
        active_connection = connection or settings.generation_connection
        headers = (
            {"Authorization": f"Bearer {active_connection.api_key}"}
            if active_connection.api_key
            else {}
        )
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
        models = await self._list_models()
        if any(
            self._model_id(name) not in models for name in self._settings.required_generation_models
        ):
            raise ProviderModelMissing("Configured Hugging Face chat model is not available")

    async def check_embedding_readiness(self, config: IndexConfiguration) -> None:
        model = self._settings.embedding_model
        if not model:
            raise ProviderModelMissing("Embedding model is not configured")
        if model != config.embedding_model:
            raise ProviderProtocolError("Embedding model differs from the reviewed index config")
        self._require_api_key()
        digest = self._settings.embedding_model_digest
        if not digest:
            raise ProviderProtocolError(
                "EMBEDDING_MODEL_DIGEST must pin Hugging Face's embedding model"
            )
        if digest.lower() != config.embedding_digest.lower():
            raise ProviderProtocolError(
                "Embedding model digest differs from the reviewed index config"
            )
        embedding = await self.embed(EmbeddingInput(texts=["readiness dimension probe"]))
        if embedding.dimension != config.embedding_dimension:
            raise ProviderProtocolError(
                "Live embedding dimension differs from the reviewed index config"
            )

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        model = request.model or self._settings.generation_model
        if not model:
            raise ProviderModelMissing("Generation model is not configured")
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.response_format == "json":
            body["response_format"] = {"type": "json_object"}
        timeout = httpx.Timeout(
            connect=self._settings.provider_connect_timeout_seconds,
            read=self._settings.generation_timeout_seconds,
            write=self._settings.provider_write_timeout_seconds,
            pool=self._settings.provider_pool_timeout_seconds,
        )
        async with self._generation_gate:
            payload = await self._request(
                "POST",
                "v1/chat/completions",
                timeout=timeout,
                json=body,
            )
        try:
            choices = payload["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("completion response has no choices")
            choice = choices[0]
            message = choice["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("completion response has no text content")
            usage = payload.get("usage", {})
            usage = usage if isinstance(usage, dict) else {}
            return GenerationOutput(
                model=payload.get("model", model),
                response=content,
                done=True,
                done_reason=choice.get("finish_reason"),
                prompt_eval_count=usage.get("prompt_tokens"),
                eval_count=usage.get("completion_tokens"),
            )
        except Exception as exc:
            raise ProviderProtocolError(
                "Hugging Face returned an invalid chat completion response"
            ) from exc

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput:
        model = self._settings.embedding_model
        if not model:
            raise ProviderModelMissing("Embedding model is not configured")
        timeout = httpx.Timeout(
            connect=self._settings.provider_connect_timeout_seconds,
            read=self._settings.embedding_timeout_seconds,
            write=self._settings.provider_write_timeout_seconds,
            pool=self._settings.provider_pool_timeout_seconds,
        )
        model_path = quote(model, safe="/@._-")
        async with self._embedding_gate:
            payload = await self._request(
                "POST",
                f"hf-inference/models/{model_path}",
                timeout=timeout,
                json={"inputs": request.texts, "normalize": False},
            )
        vectors = self._parse_vectors(payload, len(request.texts))
        if not vectors:
            raise ProviderProtocolError("Hugging Face returned no embeddings")
        return EmbeddingOutput(model=model, vectors=vectors, dimension=len(vectors[0]))

    @staticmethod
    def _parse_vectors(payload: Any, expected_count: int) -> list[list[float]]:
        try:
            if not isinstance(payload, list):
                raise ValueError("feature extraction response is invalid")
            if payload and all(isinstance(item, (int, float)) for item in payload):
                vectors = [payload] if expected_count == 1 else []
            elif payload and all(
                isinstance(item, list)
                and all(isinstance(value, (int, float)) for value in item)
                for item in payload
            ):
                vectors = payload
            else:
                raise ValueError("feature extraction output must contain pooled vectors")
            if len(vectors) != expected_count:
                raise ValueError("embedding count differs from input count")
            return cast(list[list[float]], vectors)
        except Exception as exc:
            raise ProviderProtocolError("Hugging Face returned invalid embeddings") from exc

    async def _list_models(self) -> set[str]:
        payload = await self._request("GET", "v1/models")
        try:
            data = payload["data"]
            if not isinstance(data, list):
                raise ValueError("model list is invalid")
            model_ids = {
                self._model_id(item["id"])
                for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
            if len(model_ids) != len(data):
                raise ValueError("model list contains invalid entries")
            return model_ids
        except Exception as exc:
            raise ProviderProtocolError("Hugging Face returned an invalid model list") from exc

    def _require_api_key(self) -> None:
        if not self._client.headers.get("Authorization"):
            raise ProviderError("Hugging Face API token is not configured")

    @staticmethod
    def _model_id(model: str) -> str:
        return model.partition(":")[0]

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
            if response.status_code == 404:
                raise ProviderModelMissing(
                    "Configured Hugging Face model or endpoint is unavailable"
                )
            response.raise_for_status()
            return response.json()
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Hugging Face request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(
                "Hugging Face is unavailable or returned an invalid response"
            ) from exc
