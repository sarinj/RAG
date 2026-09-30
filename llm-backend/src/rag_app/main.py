from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import monotonic
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from rag_app.api.errors import install_error_handlers
from rag_app.api.health import ReadinessChecker
from rag_app.api.router import router
from rag_app.config import Settings, load_index_configuration
from rag_app.generation.chat import BaselineChatExecutor
from rag_app.providers.factory import create_provider
from rag_app.providers.protocols import ProviderAdapter
from rag_app.repositories.database import Database
from rag_app.tokenization import load_tokenizer

logger = logging.getLogger("rag_app.http")


def _request_id(value: str | None) -> str:
    try:
        return str(UUID(value)) if value else str(uuid4())
    except ValueError:
        return str(uuid4())


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    provider: ProviderAdapter | None = None,
    chat_executor: object | None = None,
) -> FastAPI:
    active_settings = settings or Settings()
    active_database = database or Database(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        owns_provider = app.state.provider is None
        if owns_provider:
            app.state.provider = create_provider(active_settings)
        try:
            if app.state.chat_executor is None:
                try:
                    config = load_index_configuration(active_settings.index_config_path)
                    if app.state.provider is None:
                        raise ValueError("The language model provider is unavailable")
                    tokenizer = load_tokenizer(
                        active_settings.tokenizer_path,
                        config.tokenizer_version,
                    )
                    app.state.chat_executor = BaselineChatExecutor(
                        active_database,
                        app.state.provider,
                        config,
                        tokenizer,
                    )
                except Exception as exc:
                    logger.error(
                        "event=chat_executor_unavailable error_type=%s",
                        type(exc).__name__,
                    )
            await active_database.open()
            yield
        finally:
            await active_database.close()
            if owns_provider and app.state.provider is not None:
                await app.state.provider.close()
                app.state.provider = None

    app = FastAPI(
        title="Learning RAG internal service",
        description="Internal chat, model readiness, and worker-facing service boundary.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = active_settings
    app.state.database = active_database
    app.state.provider = provider
    app.state.chat_executor = chat_executor
    app.state.readiness = ReadinessChecker(app)
    app.include_router(router)
    install_error_handlers(app)

    @app.middleware("http")
    async def add_request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = _request_id(request.headers.get("X-Request-Id"))
        request.state.request_id = correlation_id
        started_at = monotonic()
        response = await call_next(request)
        response.headers["X-Request-Id"] = correlation_id
        duration_ms = round((monotonic() - started_at) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "event": "request_complete",
                    "requestId": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "durationMs": duration_ms,
                },
                separators=(",", ":"),
            )
        )
        return response

    return app


app = create_app()
