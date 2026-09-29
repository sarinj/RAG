from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_app.errors import ServiceError

internal_bearer = HTTPBearer(auto_error=False, scheme_name="InternalBearer")


async def require_internal_token(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(internal_bearer),
    ],
) -> None:
    expected = request.app.state.settings.internal_token
    if expected is None:
        raise ServiceError(503, "SERVICE_NOT_READY", "Internal authentication is not configured")
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"), expected.encode("utf-8")
    ):
        raise ServiceError(401, "UNAUTHORIZED", "A valid internal bearer token is required")
