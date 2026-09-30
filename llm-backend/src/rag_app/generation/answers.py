from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import cast

from rag_app.models import CONTRACT_VERSION, ChatRequest, ChatResponse, Citation, EvidenceRecord

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I couldn’t find enough relevant information in the selected documents. "
    "Try asking about a topic they cover, or choose another document set."
)
_CITATION_MARKER = re.compile(r"\[S([0-9]+)\]")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def source_definition_response(
    request: ChatRequest, evidence: Sequence[EvidenceRecord], subject: str | None
) -> ChatResponse | None:
    """Recover an explicit source definition when generation incorrectly abstains."""
    if subject is None:
        return None
    definition_pattern = re.compile(
        rf"^[\t ]*(?P<definition>(?:the\s+)?(?:(?:project|program|initiative)\s+)?"
        rf"{re.escape(subject)}\s+(?:is|are|means|refers to)\b[^\r\n]+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = [
        (index, match.group("definition").strip())
        for index, record in enumerate(evidence, start=1)
        for match in definition_pattern.finditer(record.chunk.original_text)
        if match.group("definition").strip().endswith(".")
    ]
    # An ambiguous name needs the generation path to distinguish its meanings.
    if len({definition.casefold() for _, definition in matches}) != 1:
        return None
    index, definition = matches[0]
    return parse_answer(json.dumps({"answer": f"{definition} [S{index}]"}), request, evidence)


def source_excerpt_response(
    request: ChatRequest,
    evidence: Sequence[EvidenceRecord],
    draft: str,
    word_limit: int,
) -> ChatResponse | None:
    """Shorten a stubborn draft using a complete source statement matching its opening."""
    draft_text = " ".join(_CITATION_MARKER.sub("", draft).replace("’", "'").casefold().split())
    excerpts: list[tuple[int, str]] = []
    for index, record in enumerate(evidence, start=1):
        for line in record.chunk.original_text.splitlines():
            statement = line.strip()
            if not statement.endswith(".") or statement.endswith(".."):
                continue
            if len(statement.split()) + 1 > word_limit:
                continue
            prefix = " ".join(statement[:-1].replace("’", "'").casefold().split())
            if prefix and (
                draft_text == prefix
                or any(draft_text.startswith(prefix + boundary) for boundary in (" ", ".", ","))
            ):
                excerpts.append((index, statement))
    for index, statement in sorted(excerpts, key=lambda item: len(item[1].split())):
        response = parse_answer(
            json.dumps({"answer": f"{statement} [S{index}]"}), request, evidence
        )
        if response is not None:
            return response
    return None


def parse_answer(
    raw_answer: str,
    request: ChatRequest,
    evidence: Sequence[EvidenceRecord],
) -> ChatResponse | None:
    answer = answer_text(raw_answer)
    if answer is None:
        return None
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


def answer_text(raw_answer: str) -> str | None:
    """Read the answer field without treating uncited model text as a valid response."""
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
    return answer_value.strip()


def insufficient_evidence_response(request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        contract_version=CONTRACT_VERSION,
        request_id=request.request_id,
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        answer_status="insufficient_evidence",
        citations=[],
    )


def small_talk_response(request: ChatRequest, answer: str) -> ChatResponse:
    return ChatResponse(
        contract_version=CONTRACT_VERSION,
        request_id=request.request_id,
        answer=answer,
        answer_status="direct_answer",
        citations=[],
    )


def document_catalog_response(request: ChatRequest, document_names: Sequence[str]) -> ChatResponse:
    if document_names:
        listed_names = "\n".join(f"- {_safe_markdown_name(name)}" for name in document_names)
        answer = f"I can access these documents in this chat:\n\n{listed_names}"
    else:
        answer = "There are no ready documents in this chat’s scope."
    return ChatResponse(
        contract_version=CONTRACT_VERSION,
        request_id=request.request_id,
        answer=answer,
        answer_status="direct_answer",
        citations=[],
    )


def _safe_markdown_name(name: str) -> str:
    normalized = " ".join(name.replace("\r", " ").replace("\n", " ").split())
    normalized = normalized.replace("[", "［").replace("]", "］")
    return re.sub(r"([\\`*_{}()#+.!|>\-])", r"\\\1", normalized)
