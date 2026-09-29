from __future__ import annotations

from uuid import uuid4

from rag_app.config import IndexConfiguration
from rag_app.generation.chat import BaselineChatExecutor
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
    config = IndexConfiguration.model_validate(
        {
            "embeddingModel": "embedding-test:latest",
            "embeddingDigest": "a" * 64,
            "embeddingDimension": 3,
            "chunkTargetTokens": 512,
            "chunkOverlapTokens": 64,
            "chunkerVersion": "chunker-v1",
            "tokenizerVersion": "tokenizer-v1",
            "contextPromptVersion": None,
            "contextModel": None,
            "contextModelDigest": None,
            "contextWindowTokens": 8192,
            "maxOutputTokens": 512,
        }
    )
    executor = BaselineChatExecutor(
        database=object(),  # type: ignore[arg-type]
        provider=object(),  # type: ignore[arg-type]
        config=config,
        tokenizer=object(),  # type: ignore[arg-type]
    )

    answer = executor._parse_answer(
        '{"answer":"Project Orion is led by Priya Sharma [S1]."}', request, [evidence]
    )
    unsupported = executor._parse_answer(
        '{"answer":"INSUFFICIENT_EVIDENCE"}', request, [evidence]
    )
    invalid = executor._parse_answer(
        '{"answer":"Project Orion is led by Priya Sharma [S2]."}', request, [evidence]
    )

    assert answer is not None
    assert answer.answer == "Project Orion is led by Priya Sharma [S1]."
    assert answer.citations[0].chunk_id == chunk_id
    assert unsupported is not None
    assert unsupported.answer_status == "insufficient_evidence"
    assert invalid is None
