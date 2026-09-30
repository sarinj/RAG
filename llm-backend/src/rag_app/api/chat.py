from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from rag_app.api.dependencies import require_internal_token
from rag_app.errors import (
    DatabaseSchemaUnavailable,
    DatabaseUnavailable,
    ProviderError,
    ProviderTimeout,
    ServiceError,
)
from rag_app.models import (
    CONTRACT_VERSION,
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    EvaluationChatResponse,
    EvidenceRecord,
    RetrievedCandidate,
)
from rag_app.repositories.documents import DocumentRepository

router = APIRouter(tags=["internal chat"])


class ChatExecutor(Protocol):
    async def answer(self, request: ChatRequest) -> ChatExecutionResult: ...


@dataclass(frozen=True)
class ChatExecutionResult:
    response: ChatResponse
    evidence_ids: frozenset[tuple[UUID, UUID, UUID]]
    retrieved_candidates: tuple[EvidenceRecord, ...] = ()


@router.post(
    "/internal/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid request or contract version"},
        401: {"model": ErrorEnvelope, "description": "Missing or invalid internal token"},
        409: {"model": ErrorEnvelope, "description": "Source snapshot changed"},
        422: {"model": ErrorEnvelope, "description": "Citation is outside retrieved evidence"},
        502: {
            "model": ErrorEnvelope,
            "description": "Executor response violated the internal contract",
        },
        503: {"model": ErrorEnvelope, "description": "Service or model is unavailable"},
        504: {"model": ErrorEnvelope, "description": "Generation deadline exceeded"},
    },
    summary="Answer a question over a server-resolved source snapshot",
)
async def chat(
    payload: ChatRequest,
    request: Request,
    _principal: None = Depends(require_internal_token),
) -> ChatResponse:
    result = await _execute_chat(payload, request)
    return result.response


@router.post(
    "/internal/evaluation/chat",
    response_model=EvaluationChatResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid request or retrieval mode"},
        401: {"model": ErrorEnvelope, "description": "Missing or invalid internal token"},
        409: {"model": ErrorEnvelope, "description": "Source snapshot changed"},
        422: {"model": ErrorEnvelope, "description": "Citation is outside retrieved evidence"},
        502: {"model": ErrorEnvelope, "description": "Executor response violated the internal contract"},
        503: {"model": ErrorEnvelope, "description": "Service or model is unavailable"},
        504: {"model": ErrorEnvelope, "description": "Generation deadline exceeded"},
    },
    summary="Run internal chat and include retrieved candidates for evaluation",
)
async def evaluation_chat(
    payload: ChatRequest,
    request: Request,
    _principal: None = Depends(require_internal_token),
) -> EvaluationChatResponse:
    result = await _execute_chat(payload, request)
    return EvaluationChatResponse(
        response=result.response,
        retrieval_candidates=[
            RetrievedCandidate(
                chunk_id=evidence.chunk.id,
                source_start=evidence.chunk.source_start,
                source_end=evidence.chunk.source_end,
            )
            for evidence in result.retrieved_candidates
        ],
    )


async def _execute_chat(payload: ChatRequest, request: Request) -> ChatExecutionResult:
    if payload.contract_version != CONTRACT_VERSION:
        raise ServiceError(
            400,
            "CONTRACT_VERSION_MISMATCH",
            "Unsupported internal contract version",
        )

    try:
        repository = DocumentRepository(request.app.state.database)
        snapshot_is_active = await repository.snapshot_is_active(payload.sources)
    except (DatabaseSchemaUnavailable, DatabaseUnavailable) as exc:
        raise ServiceError(503, "SERVICE_NOT_READY", "Source storage is unavailable") from exc
    if not snapshot_is_active:
        raise ServiceError(409, "SCOPE_CHANGED", "The source snapshot is no longer active")

    executor: ChatExecutor | None = request.app.state.chat_executor
    if executor is None:
        raise ServiceError(503, "SERVICE_NOT_READY", "Chat generation is not configured")
    try:
        async with asyncio.timeout(request.app.state.settings.generation_timeout_seconds):
            result = await executor.answer(payload)
    except (ProviderTimeout, TimeoutError) as exc:
        raise ServiceError(504, "GENERATION_TIMEOUT", "Answer generation timed out") from exc
    except (DatabaseSchemaUnavailable, DatabaseUnavailable) as exc:
        raise ServiceError(503, "SERVICE_NOT_READY", "Chat source storage is unavailable") from exc
    except ProviderError as exc:
        raise ServiceError(503, "MODEL_UNAVAILABLE", "The language model is unavailable") from exc
    response = result.response
    if response.request_id != payload.request_id:
        raise ServiceError(
            502,
            "INTERNAL_CONTRACT_ERROR",
            "Generation returned a mismatched request ID",
        )
    allowed_sources = {(source.document_id, source.generation_id) for source in payload.sources}
    if any(
        (citation.document_id, citation.generation_id) not in allowed_sources
        or (citation.document_id, citation.generation_id, citation.chunk_id)
        not in result.evidence_ids
        for citation in response.citations
    ):
        raise ServiceError(
            422,
            "INVALID_CITATIONS",
            "Answer citations do not match retrieved evidence",
        )
    return result
