from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.alias_generators import to_camel

CONTRACT_VERSION = "1.1.0"
DocumentStatus = Literal["queued", "processing", "ready", "failed", "deleted"]
CitationLabel = Annotated[str, StringConstraints(pattern=r"^S[1-9][0-9]*$")]
MessageText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8000),
]
HistoryText = Annotated[str, StringConstraints(min_length=1)]


class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class HistoryMessage(WireModel):
    role: Literal["user", "assistant"]
    content: HistoryText


class SourceGeneration(WireModel):
    document_id: UUID
    generation_id: UUID


class ChatRequest(WireModel):
    contract_version: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    request_id: UUID
    question: MessageText
    history: list[HistoryMessage] = Field(max_length=20)
    sources: list[SourceGeneration] = Field(min_length=1, max_length=100)
    retrieval_mode: Literal["vector", "hybrid"]

    @model_validator(mode="after")
    def validate_history_and_sources(self) -> ChatRequest:
        if len(self.history) % 2:
            raise ValueError("history must contain complete user/assistant pairs")
        for index, message in enumerate(self.history):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.role != expected_role:
                raise ValueError("history must contain ordered user/assistant pairs")
        document_ids = [source.document_id for source in self.sources]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("each document may appear only once in the source snapshot")
        return self


class Citation(WireModel):
    label: CitationLabel
    document_id: UUID
    generation_id: UUID
    chunk_id: UUID
    document_name: Annotated[str, StringConstraints(min_length=1)]
    page_start: int | None = Field(ge=1)
    page_end: int | None = Field(ge=1)
    heading: str | None

    @model_validator(mode="after")
    def validate_page_range(self) -> Citation:
        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                raise ValueError("pageEnd must be greater than or equal to pageStart")
        return self


class ChatResponse(WireModel):
    contract_version: Literal["1.1.0"]
    request_id: UUID
    answer: str
    answer_status: Literal["answered", "insufficient_evidence", "direct_answer"]
    citations: list[Citation]

    @model_validator(mode="after")
    def validate_citations(self) -> ChatResponse:
        labels = [citation.label for citation in self.citations]
        if len(labels) != len(set(labels)):
            raise ValueError("citation labels must be unique")
        answer_labels = set(re.findall(r"\[(S[0-9]+)\]", self.answer))
        if answer_labels != set(labels):
            raise ValueError("answer citation markers must match its citations")
        if not self.answer.strip():
            raise ValueError("chat responses need a nonempty answer")
        if self.answer_status == "answered" and not labels:
            raise ValueError("answered responses need an answer and at least one citation")
        if self.answer_status != "answered" and labels:
            raise ValueError("non-evidence responses cannot include citations")
        return self


class RetrievedCandidate(WireModel):
    chunk_id: UUID
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_source_range(self) -> RetrievedCandidate:
        if self.source_end <= self.source_start:
            raise ValueError("retrieved chunk source range must be nonempty")
        return self


class EvaluationChatResponse(WireModel):
    response: ChatResponse
    retrieval_candidates: list[RetrievedCandidate]
    reranked_candidates: list[RetrievedCandidate]


class ErrorEnvelope(WireModel):
    code: str
    message: str
    request_id: UUID


class HealthResponse(WireModel):
    status: Literal["ok", "ready"]


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    status: DocumentStatus
    storage_key: str
    active_generation_id: UUID | None
    deleted_at: datetime | None


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    generation_id: UUID
    ordinal: int
    original_text: str
    context_text: str | None
    searchable_text: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    source_start: int
    source_end: int
    token_count: int
    embedding: list[float]


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    generation_id: UUID
    ordinal: int
    original_text: str
    context_text: str | None
    searchable_text: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    source_start: int
    source_end: int
    token_count: int


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: EvidenceChunk
    document_id: UUID
    document_name: str
