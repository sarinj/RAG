from __future__ import annotations

from logging import getLogger
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_app.errors import ServiceError
from rag_app.models import ErrorEnvelope

logger = getLogger(__name__)


def request_id_for(request: Request) -> UUID:
    value = getattr(request.state, "request_id", None)
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return UUID(int=0)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError) -> JSONResponse:
        envelope = ErrorEnvelope(
            code=error.code,
            message=error.message,
            request_id=request_id_for(request),
        )
        return JSONResponse(
            status_code=error.status_code,
            content=envelope.model_dump(mode="json", by_alias=True),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        envelope = ErrorEnvelope(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            request_id=request_id_for(request),
        )
        return JSONResponse(
            status_code=400,
            content=envelope.model_dump(mode="json", by_alias=True),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code, message = (
            ("NOT_FOUND", "Resource not found")
            if error.status_code == 404
            else ("HTTP_ERROR", "Request could not be completed")
        )
        envelope = ErrorEnvelope(
            code=code,
            message=message,
            request_id=request_id_for(request),
        )
        return JSONResponse(
            status_code=error.status_code,
            content=envelope.model_dump(mode="json", by_alias=True),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
        logger.error(
            "event=unhandled_error request_id=%s exception_type=%s",
            request_id_for(request),
            type(error).__name__,
        )
        envelope = ErrorEnvelope(
            code="INTERNAL_ERROR",
            message="An internal error occurred",
            request_id=request_id_for(request),
        )
        return JSONResponse(
            status_code=500,
            content=envelope.model_dump(mode="json", by_alias=True),
        )
