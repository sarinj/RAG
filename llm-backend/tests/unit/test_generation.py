from __future__ import annotations

from uuid import uuid4

from rag_app.generation.answers import parse_answer
from rag_app.models import ChatRequest, EvidenceChunk, EvidenceRecord


def test_answer_parser_accepts_only_cited_json_answer() -> None:
    document_id, generation_id, chunk_id = uuid4(), uuid4(), uuid4()
    request_id = uuid4()
    request = ChatRequest.model_validate(
        {
            "contractVersion": "1.0.0",
            "requestId": str(request_id),
            "question": "Who leads Project Orion?",
            "history": [],
            "sources": [
                {"documentId": str(document_id), "generationId": str(generation_id)}
            ],
            "retrievalMode": "vector",
        }
    )
    evidence = EvidenceRecord(
        chunk=EvidenceChunk(
            id=chunk_id,
            generation_id=generation_id,
            ordinal=0,
            original_text="Project Orion is led by Priya Sharma.",
            context_text=None,
            searchable_text="Project Orion is led by Priya Sharma.",
            page_start=None,
            page_end=None,
            heading="Project Orion",
            source_start=0,
            source_end=38,
            token_count=9,
        ),
        document_id=document_id,
        document_name="report.txt",
    )
    answer = parse_answer(
        '{"answer":"Project Orion is led by Priya Sharma [S1]."}', request, [evidence]
    )
    unsupported = parse_answer('{"answer":"INSUFFICIENT_EVIDENCE"}', request, [evidence])
    invalid = parse_answer(
        '{"answer":"Project Orion is led by Priya Sharma [S2]."}', request, [evidence]
    )

    assert answer is not None
    assert answer.answer == "Project Orion is led by Priya Sharma [S1]."
    assert answer.citations[0].chunk_id == chunk_id
    assert unsupported is not None
    assert unsupported.answer_status == "insufficient_evidence"
    assert invalid is None
