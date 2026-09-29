from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from rag_app.config import IndexConfiguration
from rag_app.models import SourceGeneration
from rag_app.repositories.retrieval import RetrievalRepository


class FakeConnection:
    def __init__(self, row: tuple[object, ...]) -> None:
        self.row = row

    async def execute(self, _query: str, _params: tuple[object, ...]) -> FakeConnection:
        return self

    async def fetchall(self) -> list[tuple[object, ...]]:
        return [self.row]


class FakeDatabase:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[FakeConnection]:
        yield self._connection


async def test_retrieval_builds_evidence_without_loading_vector_values() -> None:
    chunk_id, generation_id, document_id = uuid4(), uuid4(), uuid4()
    row: tuple[object, ...] = (
        chunk_id,
        generation_id,
        document_id,
        "report.txt",
        0,
        "The project lead is Priya Sharma.",
        None,
        "The project lead is Priya Sharma.",
        None,
        None,
        "Project Orion",
        0,
        34,
        8,
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
    repository = RetrievalRepository(FakeDatabase(FakeConnection(row)))  # type: ignore[arg-type]

    evidence = await repository.retrieve(
        [SourceGeneration(document_id=document_id, generation_id=generation_id)],
        [0.1, 0.2, 0.3],
        config,
    )

    assert len(evidence) == 1
    assert evidence[0].chunk.id == chunk_id
    assert evidence[0].chunk.original_text == "The project lead is Priya Sharma."
    assert evidence[0].document_name == "report.txt"
