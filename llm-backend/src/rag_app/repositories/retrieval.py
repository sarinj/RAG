from __future__ import annotations

from uuid import UUID

from pgvector import Vector

from rag_app.config import IndexConfiguration
from rag_app.models import EvidenceChunk, EvidenceRecord, SourceGeneration
from rag_app.repositories.database import PsycopgDatabase

RETRIEVAL_CANDIDATE_LIMIT = 30


class RetrievalRepository:
    def __init__(self, database: PsycopgDatabase) -> None:
        self._database = database

    async def sources_match_index(
        self,
        sources: list[SourceGeneration],
        config: IndexConfiguration,
    ) -> bool:
        document_ids = [source.document_id for source in sources]
        generation_ids = [source.generation_id for source in sources]
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*)
                FROM unnest(%s::uuid[], %s::uuid[])
                     AS requested(document_id, generation_id)
                JOIN index_generations AS generation
                  ON generation.id = requested.generation_id
                 AND generation.document_id = requested.document_id
                WHERE generation.status = 'ready'
                  AND generation.embedding_model = %s
                  AND lower(generation.embedding_digest) = lower(%s)
                  AND generation.embedding_dimension = %s
                """,
                (
                    document_ids,
                    generation_ids,
                    config.embedding_model,
                    config.embedding_digest,
                    config.embedding_dimension,
                ),
            )
            row = await cursor.fetchone()
        return row is not None and row[0] == len(sources)

    async def retrieve(
        self,
        sources: list[SourceGeneration],
        query_embedding: list[float],
        config: IndexConfiguration,
    ) -> list[EvidenceRecord]:
        document_ids = [source.document_id for source in sources]
        generation_ids = [source.generation_id for source in sources]
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT c.id,c.generation_id,d.id,d.name,c.ordinal,c.original_text,
                       c.context_text,c.searchable_text,c.page_start,c.page_end,c.heading,
                       c.source_start,c.source_end,c.token_count
                FROM unnest(%s::uuid[], %s::uuid[])
                     AS requested(document_id, generation_id)
                JOIN documents AS d ON d.id = requested.document_id
                JOIN index_generations AS g
                  ON g.id = requested.generation_id AND g.document_id = d.id
                JOIN chunks AS c ON c.generation_id = g.id
                WHERE d.status = 'ready'
                  AND d.deleted_at IS NULL
                  AND d.active_generation_id = requested.generation_id
                  AND g.status = 'ready'
                  AND g.embedding_model = %s
                  AND lower(g.embedding_digest) = lower(%s)
                  AND g.embedding_dimension = %s
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (
                    document_ids,
                    generation_ids,
                    config.embedding_model,
                    config.embedding_digest,
                    config.embedding_dimension,
                    Vector(query_embedding),
                    RETRIEVAL_CANDIDATE_LIMIT,
                ),
            )
            rows = await cursor.fetchall()

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


def evidence_key(evidence: EvidenceRecord) -> tuple[UUID, UUID, UUID]:
    return (evidence.document_id, evidence.chunk.generation_id, evidence.chunk.id)
