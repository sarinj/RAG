from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from rag_app.config import IndexConfiguration
from rag_app.errors import ServiceError
from rag_app.models import SourceGeneration
from rag_app.repositories.retrieval import ACTIVE_CHUNK_LIMIT, RetrievalRepository


class FakeConnection:
    def __init__(self, row: tuple[object, ...]) -> None:
        self.row = row
        self.statement: object | None = None

    async def execute(self, _query: object, _params: tuple[object, ...]) -> FakeConnection:
        self.statement = _query
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


async def test_complete_corpus_retrieval_is_ordered_scoped_and_detects_overflow() -> None:
    class OverflowConnection(FakeConnection):
        async def fetchall(self) -> list[tuple[object, ...]]:
            return [self.row] * (ACTIVE_CHUNK_LIMIT + 1)

    connection = OverflowConnection(())
    config = IndexConfiguration(
        embedding_model="test",
        embedding_digest="a" * 64,
        embedding_dimension=3,
        chunk_target_tokens=512,
        chunk_overlap_tokens=64,
        chunker_version="test",
        tokenizer_version="test",
        context_prompt_version=None,
        context_model=None,
        context_model_digest=None,
        context_window_tokens=8192,
        max_output_tokens=512,
    )
    repository = RetrievalRepository(FakeDatabase(connection))  # type: ignore[arg-type]
    source = SourceGeneration(document_id=uuid4(), generation_id=uuid4())

    with pytest.raises(ServiceError) as failure:
        await repository.retrieve_all([source], config)

    assert failure.value.code == "CORPUS_LIMIT_EXCEEDED"
    sql = str(connection.statement)
    assert "ORDER BY documents.id, chunks.ordinal, chunks.id" in sql
    assert "documents.active_generation_id = index_generations.id" in sql
    assert "documents.deleted_at IS NULL" in sql
    assert "index_generations.embedding_dimension" in sql
