from __future__ import annotations

import json
import logging
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_app.config import RagSettings
from rag_app.models import ChatRequest, EvidenceChunk, EvidenceRecord, QueryHit, RetrievalMetadata
from rag_app.retrieval.tracing import RagTrace


def trace_for(settings: RagSettings) -> RagTrace:
    return RagTrace(
        ChatRequest(
            contract_version="1.1.0",
            request_id=uuid4(),
            question="Private question",
            history=[],
            retrieval_mode="hybrid",
            sources=[{"document_id": uuid4(), "generation_id": uuid4()}],
        ),
        settings,
    )


def record_for(trace: RagTrace) -> EvidenceRecord:
    source = trace.request.sources[0]
    return EvidenceRecord(
        document_id=source.document_id,
        document_name="Private file",
        chunk=EvidenceChunk(
            id=uuid4(),
            generation_id=source.generation_id,
            ordinal=0,
            original_text="Private source text",
            searchable_text="Private source text",
            context_text=None,
            page_start=None,
            page_end=None,
            heading=None,
            source_start=0,
            source_end=19,
            token_count=3,
        ),
        retrieval=RetrievalMetadata(
            vector_score=0.8,
            query_hits=[QueryHit(query="Private rewritten query", channel="vector", rank=1)],
        ),
    )


def test_trace_correlates_ids_and_timings_without_text(caplog) -> None:
    trace = trace_for(RagSettings())
    trace.retrieval_performed = True
    trace.candidates("vector_candidates", [record_for(trace)])
    with trace.measure("vector_retrieval"):
        pass
    with caplog.at_level(logging.INFO, logger="rag_app.rag"):
        trace.finish()
    message = caplog.records[-1].message
    data = json.loads(message)
    assert data["request_id"] == str(trace.request.request_id)
    assert data["vector_candidates"][0]["vector_score"] == 0.8
    assert data["timings_ms"]["total"] >= data["timings_ms"]["vector_retrieval"]
    assert "Private" not in message


def test_text_tracing_requires_explicit_debug_mode_and_is_configurable(caplog) -> None:
    with pytest.raises(ValidationError, match="RAG_DEBUG_MODE"):
        RagSettings(rag_trace_include_text=True)
    trace = trace_for(RagSettings(rag_debug_mode=True, rag_trace_include_text=True))
    trace.retrieval_performed = True
    trace.candidates("selected_evidence", [record_for(trace)])
    with caplog.at_level(logging.INFO, logger="rag_app.rag"):
        trace.finish()
    assert "Private source text" in caplog.records[-1].message
    caplog.clear()
    trace.settings = RagSettings(rag_trace_enabled=False)
    trace.finish()
    assert not caplog.records


def test_failure_trace_contains_error_type_without_exception_text(caplog) -> None:
    trace = trace_for(RagSettings())
    with caplog.at_level(logging.INFO, logger="rag_app.rag"):
        trace.finish(error=RuntimeError("Private provider payload"))
    assert "RuntimeError" in caplog.records[-1].message
    assert "Private" not in caplog.records[-1].message
