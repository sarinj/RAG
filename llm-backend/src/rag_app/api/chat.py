from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from rag_app.api.dependencies import require_internal_token
from rag_app.errors import (
    DatabaseSchemaUnavailable,
    DatabaseUnavailable,
    ProviderError,
    ProviderTimeout,
    ServiceError,
)
from rag_app.generation.streaming import ChatStream, active_stream, progress
from rag_app.models import (
    CONTRACT_VERSION,
    ChatRequest,
    ChatResponse,
    ErrorEnvelope,
    EvaluationChatResponse,
    EvidenceRecord,
    QueryDiagnostics,
    RetrievalEvaluationResponse,
    RetrievalStages,
    RetrievedCandidate,
)
from rag_app.repositories.documents import DocumentRepository
from rag_app.retrieval.tracing import RagTrace, active_trace

router = APIRouter(tags=["internal chat"])


class ChatExecutor(Protocol):
    async def answer(self, request: ChatRequest) -> ChatExecutionResult: ...


class RetrievalEvaluator(Protocol):
    async def retrieve_only(self, request: ChatRequest) -> ChatExecutionResult: ...


@dataclass(frozen=True)
class ChatExecutionResult:
    response: ChatResponse
    evidence_ids: frozenset[tuple[UUID, UUID, UUID]]
    retrieved_candidates: tuple[EvidenceRecord, ...] = ()
    reranked_candidates: tuple[EvidenceRecord, ...] = ()
    diagnostics: QueryDiagnostics = field(default_factory=QueryDiagnostics)
    vector_candidates: tuple[EvidenceRecord, ...] = ()
    lexical_candidates: tuple[EvidenceRecord, ...] = ()
    selected_evidence: tuple[EvidenceRecord, ...] = ()
    neighbor_chunks: tuple[EvidenceRecord, ...] = ()


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
    "/internal/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "description": "Progress, provisional answer deltas, final checked result or error",
        },
        400: {"model": ErrorEnvelope, "description": "Invalid request or contract version"},
        401: {"model": ErrorEnvelope, "description": "Missing or invalid internal token"},
    },
    summary="Stream public progress and a provisional answer, followed by the checked result",
)
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    _principal: None = Depends(require_internal_token),
) -> StreamingResponse:
    if payload.contract_version != CONTRACT_VERSION:
        raise ServiceError(
            400, "CONTRACT_VERSION_MISMATCH", "Unsupported internal contract version"
        )

    async def events():
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=64)

        async def execute() -> None:
            stream = ChatStream(payload.request_id, queue.put)
            token = active_stream.set(stream)
            try:
                result = await _execute_chat(payload, request)
                await stream.emit(
                    "result", response=result.response.model_dump(mode="json", by_alias=True)
                )
            except ServiceError as exc:
                await stream.emit(
                    "error", code=exc.code, message=exc.message, status=exc.status_code
                )
            except Exception:
                # Never publish provider exceptions, prompts or private reasoning.
                await stream.emit(
                    "error",
                    code="MODEL_UNAVAILABLE",
                    message="The answer could not be completed.",
                    status=503,
                )
            finally:
                active_stream.reset(token)
                with suppress(asyncio.QueueFull):
                    queue.put_nowait(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=10)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=True, separators=(',', ':'))}\n\n"
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/internal/evaluation/chat",
    response_model=EvaluationChatResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid request or retrieval mode"},
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
        diagnostics=result.diagnostics,
        retrieval_candidates=_candidates(result.retrieved_candidates),
        reranked_candidates=_candidates(result.reranked_candidates),
        stages=_stages(result),
    )


@router.post(
    "/internal/evaluation/retrieval",
    response_model=RetrievalEvaluationResponse,
    summary="Evaluate scoped retrieval stages without answer generation",
)
async def evaluation_retrieval(
    payload: ChatRequest,
    request: Request,
    _principal: None = Depends(require_internal_token),
) -> RetrievalEvaluationResponse:
    result = await _execute_chat(payload, request, retrieval_only=True)
    return RetrievalEvaluationResponse(stages=_stages(result), diagnostics=result.diagnostics)


def _candidates(records: Sequence[EvidenceRecord]) -> list[RetrievedCandidate]:
    return [
        RetrievedCandidate(
            chunk_id=item.chunk.id,
            document_id=item.document_id,
            generation_id=item.chunk.generation_id,
            ordinal=item.chunk.ordinal,
            rank=rank,
            retrieval=item.retrieval,
            source_start=item.chunk.source_start,
            source_end=item.chunk.source_end,
        )
        for rank, item in enumerate(records, 1)
    ]


def _stages(result: ChatExecutionResult) -> RetrievalStages:
    return RetrievalStages(
        vector=_candidates(result.vector_candidates),
        lexical=_candidates(result.lexical_candidates),
        hybrid=_candidates(result.retrieved_candidates),
        reranked=_candidates(result.reranked_candidates),
        selected=_candidates(result.selected_evidence),
        neighbors=_candidates(result.neighbor_chunks),
    )


async def _execute_chat(
    payload: ChatRequest, request: Request, *, retrieval_only: bool = False
) -> ChatExecutionResult:
    trace = RagTrace(payload, request.app.state.settings)
    trace.fields["http_request_id"] = getattr(request.state, "request_id", None)
    token = active_trace.set(trace)
    result: ChatExecutionResult | None = None
    error: BaseException | None = None
    try:
        result = await _execute_chat_impl(payload, request, retrieval_only=retrieval_only)
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace.finish(result.response if result else None, error)
        if result is not None:
            result.diagnostics.timings_ms.update(trace.timings_ms)
        active_trace.reset(token)


async def _execute_chat_impl(
    payload: ChatRequest, request: Request, *, retrieval_only: bool = False
) -> ChatExecutionResult:
    if payload.contract_version != CONTRACT_VERSION:
        raise ServiceError(
            400,
            "CONTRACT_VERSION_MISMATCH",
            "Unsupported internal contract version",
        )

    try:
        repository = DocumentRepository(request.app.state.database)
        trace = active_trace.get()
        if trace is None:
            raise RuntimeError("Chat execution requires a trace context")
        with trace.measure("scope_validation"):
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
            if retrieval_only:
                retrieve_only = getattr(executor, "retrieve_only", None)
                if not callable(retrieve_only):
                    raise ServiceError(
                        503, "SERVICE_NOT_READY", "Retrieval evaluation is unavailable"
                    )
                result = await cast(RetrievalEvaluator, executor).retrieve_only(payload)
            else:
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
    await progress("finalization", "active")
    with trace.measure("citation_validation"):
        allowed_sources = {(source.document_id, source.generation_id) for source in payload.sources}
        if any(
            (citation.document_id, citation.generation_id) not in allowed_sources
            or (citation.document_id, citation.generation_id, citation.chunk_id)
            not in result.evidence_ids
            for citation in response.citations
        ):
            trace.fields["citation_validation"] = "outside_selected_evidence"
            raise ServiceError(
                422,
                "INVALID_CITATIONS",
                "Answer citations do not match retrieved evidence",
            )
        trace.fields["citation_validation"] = "valid"
    await progress("finalization", "complete")
    return result
