from __future__ import annotations

import asyncio
from time import monotonic

from fastapi import APIRouter, FastAPI, Request

from rag_app.config import load_index_configuration
from rag_app.errors import (
    DatabaseSchemaUnavailable,
    DatabaseUnavailable,
    ProviderError,
    ServiceError,
)
from rag_app.models import ErrorEnvelope, HealthResponse
from rag_app.providers.protocols import ProviderAdapter

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={
        503: {
            "model": ErrorEnvelope,
            "description": "A required dependency is unavailable",
        }
    },
)
async def ready(request: Request) -> HealthResponse:
    checker = request.app.state.readiness
    await checker.check()
    return HealthResponse(status="ready")


class ReadinessChecker:
    def __init__(self, request_app: FastAPI) -> None:
        self._app = request_app
        self._lock = asyncio.Lock()
        self._last_check_at = 0.0

    async def check(self) -> None:
        settings = self._app.state.settings
        if (
            settings.readiness_cache_seconds
            and monotonic() - self._last_check_at < settings.readiness_cache_seconds
        ):
            return
        async with self._lock:
            if (
                settings.readiness_cache_seconds
                and monotonic() - self._last_check_at < settings.readiness_cache_seconds
            ):
                return
            try:
                await self._check_dependencies()
            except (
                DatabaseSchemaUnavailable,
                DatabaseUnavailable,
                ProviderError,
                ValueError,
            ) as exc:
                raise ServiceError(
                    503,
                    "SERVICE_NOT_READY",
                    "Service dependencies are not ready",
                ) from exc
            self._last_check_at = monotonic()

    async def _check_dependencies(self) -> None:
        settings = self._app.state.settings
        provider: ProviderAdapter | None = self._app.state.provider
        database = self._app.state.database
        if not settings.internal_token or self._app.state.chat_executor is None:
            raise ValueError("Required chat, model, or authentication settings are absent")
        if provider is None or not database.configured:
            raise ValueError("Provider or database is not configured")

        config = load_index_configuration(settings.index_config_path)
        database_dimension = await database.check_schema()
        if database_dimension != config.embedding_dimension:
            raise ValueError("Database and index dimensions differ")
        await provider.check_readiness(config)
