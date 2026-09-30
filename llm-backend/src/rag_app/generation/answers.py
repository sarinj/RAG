from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import cast

from rag_app.models import CONTRACT_VERSION, ChatRequest, ChatResponse, Citation, EvidenceRecord

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don’t have enough information in the selected documents to answer that."
)
_CITATION_MARKER = re.compile(r"\[S([0-9]+)\]")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def parse_answer(
    raw_answer: str,
    request: ChatRequest,
    evidence: Sequence[EvidenceRecord],
) -> ChatResponse | None:
    clean = _THINK_BLOCK.sub("", raw_answer).strip()
    clean = clean.removeprefix("```json").removesuffix("```").strip()
    try:
        decoded_value: object = json.loads(clean)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded_value, dict):
        return None
    decoded = cast(dict[str, object], decoded_value)
    if len(decoded) != 1 or "answer" not in decoded:
        return None
    answer_value: object = decoded["answer"]
    if not isinstance(answer_value, str):
        return None
    answer = answer_value.strip()
    if answer.casefold() in {"insufficient_evidence", "insufficient evidence"}:
        return insufficient_evidence_response(request)

    marker_numbers = list(dict.fromkeys(_CITATION_MARKER.findall(answer)))
    if not answer or not marker_numbers:
        return None
    allowed_numbers = {str(index) for index in range(1, len(evidence) + 1)}
    if any(number not in allowed_numbers for number in marker_numbers):
        return None

    citations = [
        Citation(
            label=f"S{number}",
            document_id=evidence[int(number) - 1].document_id,
            generation_id=evidence[int(number) - 1].chunk.generation_id,
            chunk_id=evidence[int(number) - 1].chunk.id,
            document_name=evidence[int(number) - 1].document_name,
            page_start=evidence[int(number) - 1].chunk.page_start,
            page_end=evidence[int(number) - 1].chunk.page_end,
            heading=evidence[int(number) - 1].chunk.heading,
        )
        for number in marker_numbers
    ]
    try:
        return ChatResponse(
            contract_version=CONTRACT_VERSION,
            request_id=request.request_id,
            answer=answer,
            answer_status="answered",
            citations=citations,
        )
    except ValueError:
        return None


def insufficient_evidence_response(request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        contract_version=CONTRACT_VERSION,
        request_id=request.request_id,
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        answer_status="insufficient_evidence",
        citations=[],
    )
