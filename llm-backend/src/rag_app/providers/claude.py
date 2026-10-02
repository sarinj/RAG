from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx

from rag_app.config import ProviderConnection, Settings
from rag_app.errors import (
    ProviderError,
    ProviderModelMissing,
    ProviderProtocolError,
    ProviderTimeout,
)
from rag_app.providers.models import GenerationInput, GenerationOutput


class ClaudeProvider:
    """Anthropic Messages API adapter; embeddings are composed separately."""

    def __init__(
        self,
        settings: Settings,
        connection: ProviderConnection | None = None,
    ) -> None:
        self._settings = settings
        active_connection = connection or settings.generation_connection
        headers = {"anthropic-version": "2023-06-01"}
        if active_connection.api_key:
            headers["x-api-key"] = active_connection.api_key
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
                max_connections=settings.provider_generation_concurrency,
                max_keepalive_connections=settings.provider_generation_concurrency,
            ),
        )
        self._generation_gate = asyncio.Semaphore(settings.provider_generation_concurrency)

    async def close(self) -> None:
        await self._client.aclose()

    async def check_generation_readiness(self) -> None:
        model = self._settings.generation_model
        if not model:
            raise ProviderModelMissing("Generation model is not configured")
        if not self._client.headers.get("x-api-key"):
            raise ProviderError("Claude API key is not configured")
        available = await self._list_models()
        if any(name not in available for name in self._settings.required_generation_models):
            raise ProviderModelMissing("Configured Claude model is not available")

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        model = request.model or self._settings.generation_model
        if not model:
            raise ProviderModelMissing("Generation model is not configured")

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.response_format == "json":
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            }

        timeout = httpx.Timeout(
            connect=self._settings.provider_connect_timeout_seconds,
            read=self._settings.generation_timeout_seconds,
            write=self._settings.provider_write_timeout_seconds,
            pool=self._settings.provider_pool_timeout_seconds,
        )
        async with self._generation_gate:
            payload = await self._request("POST", "messages", timeout=timeout, json=body)
        try:
            content = payload["content"]
            if not isinstance(content, list):
                raise ValueError("message content is invalid")
            response = "".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
            if not response:
                raise ValueError("message has no text content")
            usage = payload.get("usage", {})
            usage = usage if isinstance(usage, dict) else {}
            return GenerationOutput(
                model=payload.get("model", model),
                response=response,
                done=True,
                done_reason=payload.get("stop_reason"),
                prompt_eval_count=usage.get("input_tokens"),
                eval_count=usage.get("output_tokens"),
            )
        except Exception as exc:
            raise ProviderProtocolError("Claude returned an invalid message response") from exc

    async def _list_models(self) -> set[str]:
        payload = await self._request("GET", "models", params={"limit": 1000})
        try:
            data = payload["data"]
            if not isinstance(data, list):
                raise ValueError("model list is invalid")
            names = {
                item["id"]
                for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
            if len(names) != len(data):
                raise ValueError("model list contains invalid entries")
            return names
        except Exception as exc:
            raise ProviderProtocolError("Claude returned an invalid model list") from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            if response.status_code == 404:
                raise ProviderModelMissing("Configured Claude model or endpoint is unavailable")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderProtocolError("Claude returned an invalid response")
            return cast(dict[str, Any], payload)
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Claude request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Claude is unavailable or returned an invalid response") from exc
