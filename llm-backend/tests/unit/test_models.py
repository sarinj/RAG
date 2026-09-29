from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from rag_app.models import ChatRequest, ChatResponse

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
GENERATION_ID = UUID("22222222-2222-4222-8222-222222222222")
CHUNK_ID = UUID("33333333-3333-4333-8333-333333333333")
REQUEST_ID = UUID("44444444-4444-4444-8444-444444444444")


def valid_chat_request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "contractVersion": "1.0.0",
        "requestId": str(REQUEST_ID),
        "question": "What does the report say?",
        "history": [
            {"role": "user", "content": "What is the topic?"},
            {"role": "assistant", "content": "The report is about retrieval."},
        ],
        "sources": [{"documentId": str(DOCUMENT_ID), "generationId": str(GENERATION_ID)}],
        "retrievalMode": "vector",
    }
    payload.update(overrides)
    return payload


def test_chat_request_uses_contract_aliases_and_uuid_types() -> None:
    request = ChatRequest.model_validate(valid_chat_request())

    assert request.request_id == REQUEST_ID
    assert request.sources[0].document_id == DOCUMENT_ID
    assert set(request.model_dump(by_alias=True)) == {
        "contractVersion",
        "requestId",
        "question",
        "history",
        "sources",
        "retrievalMode",
    }


def test_chat_request_rejects_unknown_fields_duplicate_documents_and_bad_history() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(valid_chat_request(unknownField=True))
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            valid_chat_request(
                sources=[
                    {"documentId": str(DOCUMENT_ID), "generationId": str(GENERATION_ID)},
                    {"documentId": str(DOCUMENT_ID), "generationId": str(REQUEST_ID)},
                ]
            )
        )
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            valid_chat_request(
                history=[
                    {"role": "assistant", "content": "An answer without a preceding question."},
                    {"role": "user", "content": "A question."},
                ]
            )
        )


def test_chat_response_requires_citation_markers_to_match_evidence() -> None:
    citation = {
        "label": "S1",
        "documentId": str(DOCUMENT_ID),
        "generationId": str(GENERATION_ID),
        "chunkId": str(CHUNK_ID),
        "documentName": "report.md",
        "pageStart": None,
        "pageEnd": None,
        "heading": "Summary",
    }
    response = ChatResponse.model_validate(
        {
            "contractVersion": "1.0.0",
            "requestId": str(REQUEST_ID),
            "answer": "The report describes retrieval [S1].",
            "answerStatus": "answered",
            "citations": [citation],
        }
    )

    assert response.citations[0].label == "S1"
    assert response.model_dump(by_alias=True)["answerStatus"] == "answered"

    with pytest.raises(ValidationError):
        ChatResponse.model_validate(
            {
                "contractVersion": "1.0.0",
                "requestId": str(REQUEST_ID),
                "answer": "No matching source [S0].",
                "answerStatus": "insufficient_evidence",
                "citations": [],
            }
        )


def test_insufficient_evidence_must_not_include_citations() -> None:
    with pytest.raises(ValidationError):
        ChatResponse.model_validate(
            {
                "contractVersion": "1.0.0",
                "requestId": str(REQUEST_ID),
                "answer": "This is supported [S1].",
                "answerStatus": "insufficient_evidence",
                "citations": [
                    {
                        "label": "S1",
                        "documentId": str(DOCUMENT_ID),
                        "generationId": str(GENERATION_ID),
                        "chunkId": str(CHUNK_ID),
                        "documentName": "report.md",
                        "pageStart": None,
                        "pageEnd": None,
                        "heading": None,
                    }
                ],
            }
        )
