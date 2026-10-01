from __future__ import annotations

import json
import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter
from typing import Any

from rag_app.config import RagSettings
from rag_app.models import ChatRequest, ChatResponse, EvidenceRecord

_LOGGER = logging.getLogger("rag_app.rag")
_STAGES = (
    "planner",
    "query_rewrite",
    "vector_retrieval",
    "lexical_retrieval",
    "fusion",
    "reranking",
    "neighbor_expansion",
    "evidence_selection",
    "evidence_sufficiency",
    "generation",
    "claim_validation",
    "citation_validation",
    "total",
)


def text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class RagTrace:
    request: ChatRequest
    settings: RagSettings
    fields: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    candidate_records: dict[str, list[list[EvidenceRecord]]] = field(
        default_factory=lambda: dict[str, list[list[EvidenceRecord]]]()
    )
    timings_ms: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_STAGES, 0.0))
    retrieval_performed: bool = False
    started: float = field(default_factory=perf_counter)

    def __post_init__(self) -> None:
        for stage in (
            "vector_candidates",
            "lexical_candidates",
            "fused_candidates",
            "reranked_candidates",
            "selected_evidence",
            "expanded_neighbor_chunks",
        ):
            self.fields[stage] = []
        self.fields["evidence_sufficiency"] = {"status": "not_checked"}
        self.fields["configuration"] = {
            name: getattr(self.settings, name) for name in RagSettings.model_fields
        }

    def text_field(self, name: str, value: str) -> None:
        self.fields[f"{name}_hash"] = text_hash(value)
        if self.settings.rag_trace_include_text:
            self.fields[name] = value

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.timings_ms[stage] = round(
                self.timings_ms.get(stage, 0.0) + (perf_counter() - started) * 1000, 3
            )

    def candidates(
        self, stage: str, records: Sequence[EvidenceRecord], *, append: bool = False
    ) -> None:
        if append:
            self.candidate_records.setdefault(stage, []).append(list(records))
        else:
            self.candidate_records[stage] = [list(records)]
        items: list[dict[str, Any]] = []
        for rank, record in enumerate(records, 1):
            metadata = record.retrieval.model_dump(mode="json")
            for hit in metadata["query_hits"]:
                hit["query_hash"] = text_hash(hit["query"])
                if not self.settings.rag_trace_include_text:
                    hit.pop("query")
                    hit.pop("subquestion")
            item: dict[str, Any] = {
                "chunk_id": str(record.chunk.id),
                "document_id": str(record.document_id),
                "generation_id": str(record.chunk.generation_id),
                "ordinal": record.chunk.ordinal,
                "page_start": record.chunk.page_start,
                "page_end": record.chunk.page_end,
                "has_context": record.chunk.context_text is not None,
                "source_start": record.chunk.source_start,
                "source_end": record.chunk.source_end,
                "token_count": record.chunk.token_count,
                "rank": rank,
                **metadata,
            }
            if self.settings.rag_trace_include_text:
                item["original_text"] = record.chunk.original_text
                item["context_text"] = record.chunk.context_text
                item["searchable_text"] = record.chunk.searchable_text
            items.append(item)
        if append:
            self.fields.setdefault(stage, []).extend(items)
        else:
            self.fields[stage] = items

    def finish(
        self, response: ChatResponse | None = None, error: BaseException | None = None
    ) -> None:
        self.timings_ms["total"] = round((perf_counter() - self.started) * 1000, 3)
        if response is not None:
            self.text_field("answer", response.answer)
            self.fields["answer_status"] = response.answer_status
            self.fields["citations"] = [
                {
                    "label": item.label,
                    "document_id": str(item.document_id),
                    "generation_id": str(item.generation_id),
                    "chunk_id": str(item.chunk_id),
                }
                for item in response.citations
            ]
        if error is not None:
            # Provider exception messages can contain source text or credentials.
            self.fields["error_type"] = type(error).__name__
            self.fields["error_code"] = getattr(error, "code", None)
        if not self.settings.rag_trace_enabled or not (self.retrieval_performed or error):
            return
        self.text_field("question", self.request.question)
        _LOGGER.info(
            json.dumps(
                {
                    "event": "rag_trace",
                    "request_id": str(self.request.request_id),
                    "conversation_id": str(self.request.conversation_id)
                    if self.request.conversation_id
                    else None,
                    "authorized_scope": [
                        source.model_dump(mode="json") for source in self.request.sources
                    ],
                    **self.fields,
                    "timings_ms": self.timings_ms,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


active_trace: ContextVar[RagTrace | None] = ContextVar("rag_trace", default=None)
