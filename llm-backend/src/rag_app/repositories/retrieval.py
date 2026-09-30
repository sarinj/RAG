from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select, tuple_

from rag_app.config import IndexConfiguration
from rag_app.models import EvidenceChunk, EvidenceRecord, SourceGeneration
from rag_app.repositories.database import Database
from rag_app.repositories.orm import Chunk, Document, IndexGeneration

RETRIEVAL_CANDIDATE_LIMIT = 30
ACTIVE_CHUNK_LIMIT = 5_000


class RetrievalRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def sources_match_index(
        self,
        sources: list[SourceGeneration],
        config: IndexConfiguration,
    ) -> bool:
        source_pairs = [(source.document_id, source.generation_id) for source in sources]
        statement = (
            select(func.count())
            .select_from(IndexGeneration)
            .where(
                tuple_(IndexGeneration.document_id, IndexGeneration.id).in_(source_pairs),
                IndexGeneration.status == "ready",
                IndexGeneration.embedding_model == config.embedding_model,
                func.lower(IndexGeneration.embedding_digest) == func.lower(config.embedding_digest),
                IndexGeneration.embedding_dimension == config.embedding_dimension,
            )
        )
        async with self._session() as session:
            count = await session.scalar(statement)
        return count == len(sources)

    async def list_accessible_document_names(
        self,
        sources: list[SourceGeneration],
    ) -> list[str]:
        source_pairs = [(source.document_id, source.generation_id) for source in sources]
        statement = (
            select(Document.name)
            .select_from(Document)
            .join(
                IndexGeneration,
                (IndexGeneration.document_id == Document.id)
                & (IndexGeneration.id == Document.active_generation_id),
            )
            .where(
                tuple_(Document.id, IndexGeneration.id).in_(source_pairs),
                Document.status == "ready",
                Document.deleted_at.is_(None),
                IndexGeneration.status == "ready",
            )
            .order_by(func.lower(Document.name), Document.id)
        )
        async with self._session() as session:
            rows = await self._fetchall(session, statement)
        return [str(row[0]) for row in rows]

    async def retrieve(
        self,
        sources: list[SourceGeneration],
        query_embedding: list[float],
        config: IndexConfiguration,
    ) -> list[EvidenceRecord]:
        statement = self._active_statement(sources, config).order_by(
            Chunk.embedding.cosine_distance(query_embedding), Chunk.id
        ).limit(RETRIEVAL_CANDIDATE_LIMIT)
        async with self._session() as session:
            rows = await self._fetchall(session, statement)
        return self._evidence_records(rows)

    async def retrieve_active(
        self,
        sources: list[SourceGeneration],
        config: IndexConfiguration,
    ) -> list[EvidenceRecord]:
        """Load the active, source-scoped corpus used to rebuild per-request BM25."""
        statement = self._active_statement(sources, config).order_by(Chunk.id).limit(
            ACTIVE_CHUNK_LIMIT
        )
        async with self._session() as session:
            rows = await self._fetchall(session, statement)
        return self._evidence_records(rows)

    @staticmethod
    def _active_statement(sources: list[SourceGeneration], config: IndexConfiguration) -> Any:
        source_pairs = [(source.document_id, source.generation_id) for source in sources]
        return (
            select(
                Chunk.id,
                Chunk.generation_id,
                Document.id,
                Document.name,
                Chunk.ordinal,
                Chunk.original_text,
                Chunk.context_text,
                Chunk.searchable_text,
                Chunk.page_start,
                Chunk.page_end,
                Chunk.heading,
                Chunk.source_start,
                Chunk.source_end,
                Chunk.token_count,
            )
            .select_from(Chunk)
            .join(IndexGeneration, IndexGeneration.id == Chunk.generation_id)
            .join(Document, Document.id == IndexGeneration.document_id)
            .where(
                tuple_(Document.id, IndexGeneration.id).in_(source_pairs),
                Document.status == "ready",
                Document.deleted_at.is_(None),
                Document.active_generation_id == IndexGeneration.id,
                IndexGeneration.status == "ready",
                IndexGeneration.embedding_model == config.embedding_model,
                func.lower(IndexGeneration.embedding_digest) == func.lower(config.embedding_digest),
                IndexGeneration.embedding_dimension == config.embedding_dimension,
            )
        )

    @staticmethod
    def _evidence_records(rows: list[Any]) -> list[EvidenceRecord]:
        return [
            EvidenceRecord(
                chunk=EvidenceChunk(
                    id=row[0],
                    generation_id=row[1],
                    ordinal=row[4],
                    original_text=row[5],
                    context_text=row[6],
                    searchable_text=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    heading=row[10],
                    source_start=row[11],
                    source_end=row[12],
                    token_count=row[13],
                ),
                document_id=row[2],
                document_name=row[3],
            )
            for row in rows
        ]

    async def _fetchall(self, session: Any, statement: Any) -> list[Any]:
        result = await session.execute(statement, {})
        rows = result.fetchall()
        if inspect.isawaitable(rows):
            rows = await rows
        return list(rows)

    @asynccontextmanager
    async def _session(self) -> AsyncGenerator[Any, None]:
        try:
            session_context: Any = self._database.session
        except AttributeError:
            session_context = cast(Any, self._database).connection
        async with session_context() as session:
            yield session


def evidence_key(evidence: EvidenceRecord) -> tuple[UUID, UUID, UUID]:
    return (evidence.document_id, evidence.chunk.generation_id, evidence.chunk.id)
