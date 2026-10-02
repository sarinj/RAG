from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any, cast

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
from rag_app.providers.protocols import TextSink


def strict_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt Pydantic defaults to OpenAI's strict subset; validate locally as well."""
    result = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            node = cast(dict[str, Any], value)
            node.pop("default", None)
            # UUID format is not in the API's supported string formats. The local
            # Pydantic model still validates UUIDs and authorized evidence membership.
            if node.get("format") == "uuid":
                node.pop("format")
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for child in node.values():
                visit(child)
        elif isinstance(value, list):
            for child in cast(list[Any], value):
                visit(child)

    visit(result)
    return result


class OpenAIProvider:
    """Adapter for OpenAI-style chat-completions and embeddings APIs."""

    def __init__(
        self,
        settings: Settings,
        connection: ProviderConnection | None = None,
        *,
        embedding_dimension: int | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_dimension = embedding_dimension
        active_connection = connection or settings.generation_connection
        headers: dict[str, str] = {}
        if active_connection.api_key:
            headers["Authorization"] = f"Bearer {active_connection.api_key}"
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

    async def check_readiness(
        self,
        config: IndexConfiguration,
        *,
        require_generation: bool = True,
    ) -> None:
        if require_generation:
            await self.check_generation_readiness()
        await self.check_embedding_readiness(config)

    async def check_generation_readiness(self) -> None:
        generation_model = self._settings.generation_model
        if not generation_model:
            raise ProviderModelMissing("Generation model is not configured")
        available = await self._list_models()
        if any(model not in available for model in self._settings.required_generation_models):
            raise ProviderModelMissing("Configured generation model is not available")

    async def check_embedding_readiness(self, config: IndexConfiguration) -> None:
        embedding_model = self._settings.embedding_model
        if not embedding_model:
            raise ProviderModelMissing("Embedding model is not configured")
        if embedding_model != config.embedding_model:
            raise ProviderProtocolError("Embedding model differs from the reviewed index config")
        if self._embedding_dimension != config.embedding_dimension:
            raise ProviderProtocolError(
                "OpenAI embedding dimensions differ from the reviewed index config"
            )

        configured_digest = self._settings.embedding_model_digest
        if not configured_digest:
            raise ProviderProtocolError("EMBEDDING_MODEL_DIGEST must pin OpenAI's embedding model")
        if configured_digest.lower() != config.embedding_digest.lower():
            raise ProviderProtocolError(
                "Embedding model digest differs from the reviewed index config"
            )

        available_models = await self._list_models()
        if embedding_model not in available_models:
            raise ProviderModelMissing("Configured embedding model is not available")

        live_embedding = await self.embed(EmbeddingInput(texts=["readiness dimension probe"]))
        if live_embedding.dimension != config.embedding_dimension:
            raise ProviderProtocolError(
                "Live embedding dimension differs from the reviewed index config"
            )

    def _generation_parameters(
        self, request: GenerationInput
    ) -> tuple[str, dict[str, Any], httpx.Timeout]:
        model = request.model or self._settings.generation_model
        if not model:
            raise ProviderModelMissing("Generation model is not configured")

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_completion_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.reasoning_effort is not None and model.startswith(("gpt-5", "o3", "o4")):
            body["reasoning_effort"] = request.reasoning_effort
            if request.reasoning_effort != "minimal":
                body["max_completion_tokens"] += request.reasoning_token_budget
        elif model.startswith("gpt-5-nano"):
            body["reasoning_effort"] = "minimal"
        elif request.temperature is not None and not model.startswith("gpt-5"):
            body["temperature"] = request.temperature
        if request.verbosity is not None and model.startswith("gpt-5"):
            body["verbosity"] = request.verbosity
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_response",
                    "strict": True,
                    "schema": strict_response_schema(request.response_schema),
                },
            }
        elif request.response_format == "json":
            body["response_format"] = {"type": "json_object"}

        timeout = httpx.Timeout(
            connect=self._settings.provider_connect_timeout_seconds,
            read=self._settings.generation_timeout_seconds,
            write=self._settings.provider_write_timeout_seconds,
            pool=self._settings.provider_pool_timeout_seconds,
        )
        return model, body, timeout

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        model, body, timeout = self._generation_parameters(request)
        async with self._generation_gate:
            payload = await self._request(
                "POST",
                "chat/completions",
                timeout=timeout,
                json=body,
            )
        return self._parse_generation(payload, model)

    async def generate_stream(
        self, request: GenerationInput, on_text: TextSink
    ) -> GenerationOutput:
        model, body, timeout = self._generation_parameters(request)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        fragments: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        completed = False
        response_model = model
        total_chars = 0
        try:
            async with self._generation_gate:
                async with self._client.stream(
                    "POST",
                    "chat/completions",
                    json=body,
                    timeout=timeout,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    if response.status_code == 404:
                        raise ProviderModelMissing(
                            "Configured OpenAI model or endpoint is unavailable"
                        )
                    response.raise_for_status()
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise ProviderProtocolError("OpenAI did not return an event stream")
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip(" "))
                            if sum(len(item) for item in data_lines) > 262_144:
                                raise ProviderProtocolError("OpenAI stream event exceeds the limit")
                        elif not line and data_lines:
                            data = "\n".join(data_lines)
                            data_lines.clear()
                            if data == "[DONE]":
                                completed = True
                                break
                            try:
                                raw_event = json.loads(data)
                                if not isinstance(raw_event, dict):
                                    raise ValueError("invalid chunk")
                                event = cast(dict[str, Any], raw_event)
                                choices = event.get("choices")
                                if not isinstance(choices, list):
                                    raise ValueError("invalid chunk")
                                event_model = event.get("model")
                                if isinstance(event_model, str):
                                    response_model = event_model
                                event_usage = event.get("usage")
                                if isinstance(event_usage, dict):
                                    usage = cast(dict[str, Any], event_usage)
                                for raw_choice in cast(list[Any], choices):
                                    if not isinstance(raw_choice, dict):
                                        continue
                                    choice = cast(dict[str, Any], raw_choice)
                                    if choice.get("index", 0) != 0:
                                        continue
                                    choice_finish_reason = choice.get("finish_reason")
                                    if isinstance(choice_finish_reason, str):
                                        finish_reason = choice_finish_reason
                                    delta = choice.get("delta", {})
                                    content = (
                                        cast(dict[str, Any], delta).get("content")
                                        if isinstance(delta, dict)
                                        else None
                                    )
                                    # Deliberately ignore reasoning/refusal/tool fields.
                                    if isinstance(content, str) and content:
                                        total_chars += len(content)
                                        if total_chars > 262_144:
                                            raise ProviderProtocolError(
                                                "OpenAI streamed answer exceeds the limit"
                                            )
                                        fragments.append(content)
                                        await on_text(content)
                            except (ValueError, TypeError, KeyError) as exc:
                                raise ProviderProtocolError(
                                    "OpenAI returned an invalid stream event"
                                ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("OpenAI stream timed out") from exc
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise ProviderError("OpenAI stream is unavailable") from exc
        if not completed or finish_reason is None:
            raise ProviderProtocolError("OpenAI stream ended before completion")
        return self._parse_generation(
            {
                "model": response_model,
                "choices": [
                    {"message": {"content": "".join(fragments)}, "finish_reason": finish_reason}
                ],
                "usage": usage,
            },
            model,
        )

    @staticmethod
    def _parse_generation(payload: dict[str, Any], model: str) -> GenerationOutput:
        try:
            choices = payload["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("completion response has no choices")
            first_choice = cast(dict[str, Any], choices[0])
            message = first_choice["message"]
            if not isinstance(message, dict):
                raise ValueError("completion response has no text content")
            message = cast(dict[str, Any], message)
            if not isinstance(message.get("content"), str):
                raise ValueError("completion response has no text content")
            usage = payload.get("usage", {})
            usage = cast(dict[str, Any], usage) if isinstance(usage, dict) else {}
            completion_details = usage.get("completion_tokens_details", {})
            completion_details = (
                cast(dict[str, Any], completion_details)
                if isinstance(completion_details, dict)
                else {}
            )
            return GenerationOutput(
                model=payload.get("model", model),
                response=message["content"],
                done=True,
                done_reason=first_choice.get("finish_reason"),
                prompt_eval_count=usage.get("prompt_tokens"),
                eval_count=usage.get("completion_tokens"),
                reasoning_tokens=completion_details.get("reasoning_tokens"),
            )
        except Exception as exc:
            raise ProviderProtocolError("OpenAI returned an invalid generation response") from exc

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
        body: dict[str, Any] = {"model": model, "input": request.texts}
        if self._embedding_dimension is not None:
            body["dimensions"] = self._embedding_dimension
        async with self._embedding_gate:
            payload = await self._request(
                "POST",
                "embeddings",
                timeout=timeout,
                json=body,
            )
        try:
            data = payload["data"]
            if not isinstance(data, list):
                raise ValueError("embedding count differs from input count")
            data = cast(list[Any], data)
            if len(data) != len(request.texts):
                raise ValueError("embedding count differs from input count")
            vectors_by_index: dict[int, list[float]] = {}
            for position, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError("embedding item is invalid")
                item = cast(dict[str, Any], item)
                index = item.get("index", position)
                vector = item.get("embedding")
                if type(index) is not int or not isinstance(vector, list):
                    raise ValueError("embedding item is invalid")
                if index < 0 or index >= len(request.texts) or index in vectors_by_index:
                    raise ValueError("embedding indexes are invalid")
                vectors_by_index[index] = cast(list[float], vector)
            if set(vectors_by_index) != set(range(len(request.texts))):
                raise ValueError("embedding indexes are incomplete")
            vectors = [vectors_by_index[index] for index in range(len(request.texts))]
            return EmbeddingOutput(
                model=payload.get("model", model),
                vectors=vectors,
                dimension=len(vectors[0]),
            )
        except Exception as exc:
            raise ProviderProtocolError("OpenAI returned invalid embeddings") from exc

    async def _list_models(self) -> set[str]:
        payload = await self._request("GET", "models")
        try:
            data = payload["data"]
            if not isinstance(data, list):
                raise ValueError("model list is invalid")
            data = cast(list[Any], data)
            model_names: set[str] = set()
            for item in data:
                if isinstance(item, dict):
                    model_id = cast(dict[str, Any], item).get("id")
                    if isinstance(model_id, str):
                        model_names.add(model_id)
            if len(model_names) != len(data):
                raise ValueError("model list contains invalid entries")
            return model_names
        except Exception as exc:
            raise ProviderProtocolError("OpenAI returned an invalid model list") from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            if response.status_code == 404:
                raise ProviderModelMissing("Configured OpenAI model or endpoint is unavailable")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderProtocolError("OpenAI returned an invalid response")
            return cast(dict[str, Any], payload)
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("OpenAI request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("OpenAI is unavailable or returned an invalid response") from exc
