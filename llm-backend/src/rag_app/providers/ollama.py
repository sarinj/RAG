from __future__ import annotations

import asyncio
import math
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from rag_app.config import Settings
from rag_app.errors import (
    ProviderError,
    ProviderModelMissing,
    ProviderProtocolError,
    ProviderTimeout,
)


class OllamaModelTag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    model: str | None = None
    digest: str
    size: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class OllamaModelList(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[OllamaModelTag]


class OllamaShowResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modelfile: str | None = None
    parameters: str | None = None
    template: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)


class GenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(min_length=1)
    context_window_tokens: StrictInt | None = Field(default=None, gt=0)
    max_output_tokens: StrictInt = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    think: StrictBool | None = None
    response_format: Literal["json"] | None = None


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    response: str
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class EmbeddingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[StrictStr] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_texts(self) -> EmbeddingInput:
        if any(not text.strip() for text in self.texts):
            raise ValueError("embedding texts cannot be empty")
        return self


class EmbeddingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    vectors: list[list[float]] = Field(min_length=1)
    dimension: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_vectors(self) -> EmbeddingOutput:
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("embedding vectors have inconsistent dimensions")
        if any(not math.isfinite(value) for vector in self.vectors for value in vector):
            raise ValueError("embedding vectors must contain finite values")
        return self


class OllamaProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(
                connect=settings.ollama_connect_timeout_seconds,
                read=settings.ollama_read_timeout_seconds,
                write=settings.ollama_write_timeout_seconds,
                pool=settings.ollama_pool_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=(
                    settings.ollama_generation_concurrency + settings.ollama_embedding_concurrency
                ),
                max_keepalive_connections=settings.ollama_generation_concurrency
                + settings.ollama_embedding_concurrency,
            ),
        )
        self._generation_gate = asyncio.Semaphore(settings.ollama_generation_concurrency)
        self._embedding_gate = asyncio.Semaphore(settings.ollama_embedding_concurrency)

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[OllamaModelTag]:
        payload = await self._request("GET", "/api/tags")
        try:
            return OllamaModelList.model_validate(payload).models
        except Exception as exc:
            raise ProviderProtocolError("Ollama returned an invalid model list") from exc

    async def show_model(self, model: str) -> OllamaShowResponse:
        payload = await self._request("POST", "/api/show", json={"model": model})
        try:
            return OllamaShowResponse.model_validate(payload)
        except Exception as exc:
            raise ProviderProtocolError("Ollama returned invalid model information") from exc

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        if not self._settings.generation_model:
            raise ProviderModelMissing("Generation model is not configured")
        options: dict[str, int | float] = {"num_predict": request.max_output_tokens}
        if request.context_window_tokens is not None:
            options["num_ctx"] = request.context_window_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        timeout = httpx.Timeout(
            connect=self._settings.ollama_connect_timeout_seconds,
            read=self._settings.generation_timeout_seconds,
            write=self._settings.ollama_write_timeout_seconds,
            pool=self._settings.ollama_pool_timeout_seconds,
        )
        body: dict[str, Any] = {
            "model": self._settings.generation_model,
            "prompt": request.prompt,
            "stream": False,
            "options": options,
        }
        if request.think is not None:
            body["think"] = request.think
        if request.response_format is not None:
            body["format"] = request.response_format
        async with self._generation_gate:
            payload = await self._request(
                "POST",
                "/api/generate",
                timeout=timeout,
                json=body,
            )
        try:
            return GenerationOutput.model_validate(payload)
        except Exception as exc:
            raise ProviderProtocolError("Ollama returned an invalid generation response") from exc

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput:
        if not self._settings.embedding_model:
            raise ProviderModelMissing("Embedding model is not configured")
        timeout = httpx.Timeout(
            connect=self._settings.ollama_connect_timeout_seconds,
            read=self._settings.embedding_timeout_seconds,
            write=self._settings.ollama_write_timeout_seconds,
            pool=self._settings.ollama_pool_timeout_seconds,
        )
        async with self._embedding_gate:
            payload = await self._request(
                "POST",
                "/api/embed",
                timeout=timeout,
                json={"model": self._settings.embedding_model, "input": request.texts},
            )
        try:
            vectors = payload["embeddings"]
            if len(vectors) != len(request.texts):
                raise ValueError("embedding count differs from input count")
            dimension = len(vectors[0])
            return EmbeddingOutput(
                model=payload.get("model", self._settings.embedding_model),
                vectors=vectors,
                dimension=dimension,
            )
        except Exception as exc:
            raise ProviderProtocolError("Ollama returned invalid embeddings") from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            if response.status_code == 404:
                raise ProviderModelMissing("Configured Ollama model or endpoint is unavailable")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderProtocolError("Ollama returned an invalid response")
            return cast(dict[str, Any], payload)
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Ollama is unavailable or returned an invalid response") from exc


def find_model(models: list[OllamaModelTag], requested_name: str) -> OllamaModelTag:
    requested = requested_name if ":" in requested_name else f"{requested_name}:latest"
    for model in models:
        if model.name == requested or model.model == requested:
            return model
        if ":" not in requested_name and model.name == requested_name:
            return model
    raise ProviderModelMissing("Configured Ollama model is not available")


def find_context_window(model_info: dict[str, Any]) -> int | None:
    candidates: list[int] = []
    for key, value in model_info.items():
        if isinstance(value, dict):
            nested = find_context_window(cast(dict[str, Any], value))
            if nested is not None:
                candidates.append(nested)
        elif key.lower().endswith("context_length") and isinstance(value, int):
            candidates.append(value)
    return max(candidates) if candidates else None
