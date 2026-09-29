from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from pgvector import Vector
from psycopg import AsyncConnection

from rag_app.errors import DatabaseUnavailable
from rag_app.ingestion.chunking import IngestionChunk
from rag_app.ingestion.errors import CorpusCapacityExceeded, LeaseLost
from rag_app.repositories.database import PsycopgDatabase

IngestionStage = Literal[
    "queued",
    "parsing",
    "chunking",
    "contextualizing",
    "embedding",
    "publishing",
    "done",
]
MAX_ATTEMPTS = 3
LEASE_SECONDS = 90
CHUNK_CAPACITY_LOCK = 71001


@dataclass(frozen=True, slots=True)
class JobClaim:
    job_id: UUID
    document_id: UUID
    generation_id: UUID
    lease_token: UUID
    attempts: int


@dataclass(frozen=True, slots=True)
class WorkerJob:
    claim: JobClaim
    storage_key: str
    size_bytes: int
    content_hash: str
    mode: str
    embedding_model: str
    embedding_digest: str
    embedding_dimension: int
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StagedChunk:
    chunk: IngestionChunk
    embedding: Sequence[float]


class IngestionRepository:
    def __init__(self, database: PsycopgDatabase) -> None:
        self._database = database

    async def claim_next(self) -> JobClaim | None:
        async with self._database.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    """
                    SELECT id,document_id,generation_id
                    FROM ingestion_jobs
                    WHERE attempts < %s
                      AND (
                        status = 'queued'
                        OR (status = 'processing' AND lease_expires_at <= now())
                      )
                    ORDER BY created_at,id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                    (MAX_ATTEMPTS,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                job_id, document_id, generation_id = row
                lease_token = uuid4()
                cursor = await connection.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'processing',attempts = attempts + 1,lease_token = %s,
                        lease_expires_at = now() + (%s * interval '1 second'),
                        heartbeat_at = now(),error_code = NULL,error_message = NULL,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING attempts
                    """,
                    (lease_token, LEASE_SECONDS, job_id),
                )
                updated = await cursor.fetchone()
                if updated is None:
                    return None
                return JobClaim(
                    job_id=job_id,
                    document_id=document_id,
                    generation_id=generation_id,
                    lease_token=lease_token,
                    attempts=updated[0],
                )

    async def begin(self, claim: JobClaim) -> WorkerJob:
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                if document is None:
                    raise LeaseLost("The document no longer exists")
                self._require_live_lease(document, job, claim)
                if document[1] == "deleted" or document[6] is not None:
                    raise LeaseLost("The document was deleted")

                cursor = await connection.execute(
                    """
                    SELECT mode,status,embedding_model,embedding_digest,embedding_dimension,config
                    FROM index_generations
                    WHERE id = %s AND document_id = %s
                    FOR UPDATE
                    """,
                    (claim.generation_id, claim.document_id),
                )
                generation = await cursor.fetchone()
                if generation is None or generation[1] not in {"queued", "processing"}:
                    raise LeaseLost("The generation is no longer active")

                await connection.execute("SELECT pg_advisory_xact_lock(%s)", (CHUNK_CAPACITY_LOCK,))
                await connection.execute(
                    "DELETE FROM chunks WHERE generation_id = %s", (claim.generation_id,)
                )
                await connection.execute(
                    """
                    UPDATE index_generations
                    SET status = 'processing',updated_at = now()
                    WHERE id = %s
                    """,
                    (claim.generation_id,),
                )
                await connection.execute(
                    """
                    UPDATE documents
                    SET status = CASE
                      WHEN active_generation_id IS NULL THEN 'processing'
                      ELSE status
                    END,
                    updated_at = now()
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (claim.document_id,),
                )
                await connection.execute(
                    """
                    UPDATE ingestion_jobs
                    SET stage = 'parsing',processed_chunks = 0,total_chunks = NULL,
                        error_code = NULL,error_message = NULL,updated_at = now()
                    WHERE id = %s
                    """,
                    (claim.job_id,),
                )

                raw_config = generation[5]
                config: Any = raw_config
                if isinstance(raw_config, str):
                    config = json.loads(raw_config)
                if not isinstance(config, Mapping):
                    raise DatabaseUnavailable("Generation configuration is invalid")
                generation_config = cast(Mapping[str, Any], config)
                return WorkerJob(
                    claim=claim,
                    storage_key=document[3],
                    size_bytes=int(document[4]),
                    content_hash=document[5],
                    mode=generation[0],
                    embedding_model=generation[2],
                    embedding_digest=generation[3],
                    embedding_dimension=generation[4],
                    config=generation_config,
                )

    async def heartbeat(self, claim: JobClaim) -> bool:
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                if (
                    document is None
                    or document[6] is not None
                    or document[1] == "deleted"
                    or not self._owns_live_lease(job, claim)
                ):
                    return False
                await connection.execute(
                    """
                    UPDATE ingestion_jobs
                    SET lease_expires_at = now() + (%s * interval '1 second'),
                        heartbeat_at = now(),updated_at = now()
                    WHERE id = %s
                    """,
                    (LEASE_SECONDS, claim.job_id),
                )
                return True

    async def set_stage(
        self,
        claim: JobClaim,
        stage: IngestionStage,
        *,
        total_chunks: int | None = None,
    ) -> None:
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                self._require_live_lease(document, job, claim)
                if total_chunks is None:
                    await connection.execute(
                        "UPDATE ingestion_jobs SET stage = %s,updated_at = now() WHERE id = %s",
                        (stage, claim.job_id),
                    )
                else:
                    await connection.execute(
                        """
                        UPDATE ingestion_jobs
                        SET stage = %s,total_chunks = %s,updated_at = now()
                        WHERE id = %s
                        """,
                        (stage, total_chunks, claim.job_id),
                    )

    async def check_capacity(self, claim: JobClaim, additional_chunks: int) -> None:
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                self._require_live_lease(document, job, claim)
                await connection.execute("SELECT pg_advisory_xact_lock(%s)", (CHUNK_CAPACITY_LOCK,))
                cursor = await connection.execute(
                    "SELECT count(*),rag_chunk_capacity_limit() FROM chunks"
                )
                row = await cursor.fetchone()
                if row is None or row[0] + additional_chunks > row[1]:
                    raise CorpusCapacityExceeded()

    async def stage_batch(self, claim: JobClaim, staged: Sequence[StagedChunk]) -> None:
        if not staged:
            return
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                self._require_live_lease(document, job, claim)
                await connection.execute("SELECT pg_advisory_xact_lock(%s)", (CHUNK_CAPACITY_LOCK,))
                cursor = await connection.execute(
                    "SELECT count(*),rag_chunk_capacity_limit() FROM chunks"
                )
                row = await cursor.fetchone()
                if row is None or row[0] + len(staged) > row[1]:
                    raise CorpusCapacityExceeded()
                await connection.execute(
                    "SELECT rag_reserve_chunk_capacity(%s)", (len(staged),)
                )
                values: list[tuple[Any, ...]] = [
                    (
                        uuid4(),
                        claim.generation_id,
                        item.chunk.ordinal,
                        item.chunk.original_text,
                        None,
                        item.chunk.original_text,
                        item.chunk.page_start,
                        item.chunk.page_end,
                        item.chunk.heading,
                        item.chunk.source_start,
                        item.chunk.source_end,
                        item.chunk.token_count,
                        Vector(list(item.embedding)),
                    )
                    for item in staged
                ]
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO chunks (
                          id,generation_id,ordinal,original_text,context_text,searchable_text,
                          page_start,page_end,heading,source_start,source_end,token_count,embedding
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        values,
                    )
                await connection.execute(
                    """
                    UPDATE ingestion_jobs
                    SET processed_chunks = processed_chunks + %s,updated_at = now()
                    WHERE id = %s
                    """,
                    (len(staged), claim.job_id),
                )

    async def publish(self, claim: JobClaim) -> None:
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                if job is None:
                    raise LeaseLost("The ingestion job no longer exists")
                self._require_live_lease(document, job, claim)
                if job[5] is None or job[4] != job[5]:
                    raise LeaseLost("The staged chunk count is incomplete")
                cursor = await connection.execute(
                    "SELECT count(*) FROM chunks WHERE generation_id = %s",
                    (claim.generation_id,),
                )
                count_row = await cursor.fetchone()
                if count_row is None or count_row[0] != job[5]:
                    raise LeaseLost("The staged chunk count changed")
                await connection.execute(
                    """
                    UPDATE index_generations
                    SET status = 'ready',updated_at = now()
                    WHERE id = %s AND document_id = %s AND status = 'processing'
                    """,
                    (claim.generation_id, claim.document_id),
                )
                await connection.execute(
                    """
                    UPDATE documents
                    SET status = 'ready',active_generation_id = %s,updated_at = now()
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (claim.generation_id, claim.document_id),
                )
                await connection.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'succeeded',stage = 'done',lease_token = NULL,
                        lease_expires_at = NULL,heartbeat_at = now(),error_code = NULL,
                        error_message = NULL,updated_at = now()
                    WHERE id = %s
                    """,
                    (claim.job_id,),
                )

    async def fail(
        self,
        claim: JobClaim,
        *,
        code: str,
        message: str,
        retryable: bool,
    ) -> bool:
        async with self._database.connection() as connection:
            async with connection.transaction():
                document, job = await self._lock_document_job(connection, claim)
                if (
                    document is None
                    or document[6] is not None
                    or document[1] == "deleted"
                    or not self._owns_live_lease(job, claim)
                ):
                    return False
                if retryable and claim.attempts < MAX_ATTEMPTS:
                    await connection.execute(
                        """
                        UPDATE ingestion_jobs
                        SET lease_expires_at = now(),error_code = %s,error_message = %s,
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (code, message, claim.job_id),
                    )
                    return True
                await self._mark_failed(connection, claim, code, message)
                return True

    async def fail_expired_exhausted(self, batch_size: int = 16) -> int:
        async with self._database.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT id,document_id,generation_id,lease_token,attempts
                FROM ingestion_jobs
                WHERE status = 'processing' AND attempts >= %s AND lease_expires_at <= now()
                ORDER BY created_at,id
                LIMIT %s
                """,
                (MAX_ATTEMPTS, batch_size),
            )
            candidates = await cursor.fetchall()
        failed = 0
        for row in candidates:
            claim = JobClaim(
                job_id=row[0],
                document_id=row[1],
                generation_id=row[2],
                lease_token=row[3],
                attempts=row[4],
            )
            async with self._database.connection() as connection:
                async with connection.transaction():
                    document, job = await self._lock_document_job(connection, claim)
                    if (
                        document is None
                        or document[6] is not None
                        or document[1] == "deleted"
                        or job is None
                        or job[0] != "processing"
                        or job[3] < MAX_ATTEMPTS
                        or job[2]
                    ):
                        continue
                    await self._mark_failed(
                        connection,
                        claim,
                        "WORKER_ATTEMPTS_EXHAUSTED",
                        "The worker could not process this document after several attempts.",
                    )
                    failed += 1
        return failed

    async def _lock_document_job(
        self,
        connection: AsyncConnection[tuple[Any, ...]],
        claim: JobClaim,
    ) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
        cursor = await connection.execute(
            """
            SELECT id,status,active_generation_id,storage_key,size_bytes,content_hash,deleted_at
            FROM documents WHERE id = %s FOR UPDATE
            """,
            (claim.document_id,),
        )
        document = await cursor.fetchone()
        cursor = await connection.execute(
            """
            SELECT status,lease_token,lease_expires_at > now(),attempts,
                   processed_chunks,total_chunks,stage
            FROM ingestion_jobs WHERE id = %s AND document_id = %s FOR UPDATE
            """,
            (claim.job_id, claim.document_id),
        )
        job = await cursor.fetchone()
        return document, job

    @staticmethod
    def _owns_live_lease(job: tuple[Any, ...] | None, claim: JobClaim) -> bool:
        return bool(
            job is not None
            and job[0] == "processing"
            and job[1] == claim.lease_token
            and job[2]
        )

    def _require_live_lease(
        self,
        document: tuple[Any, ...] | None,
        job: tuple[Any, ...] | None,
        claim: JobClaim,
    ) -> None:
        if (
            document is None
            or document[6] is not None
            or document[1] == "deleted"
            or not self._owns_live_lease(job, claim)
        ):
            raise LeaseLost("The worker no longer owns this ingestion job")

    async def _mark_failed(
        self,
        connection: AsyncConnection[tuple[Any, ...]],
        claim: JobClaim,
        code: str,
        message: str,
    ) -> None:
        await connection.execute(
            """
            UPDATE index_generations
            SET status = 'failed',updated_at = now()
            WHERE id = %s AND document_id = %s AND status IN ('queued','processing')
            """,
            (claim.generation_id, claim.document_id),
        )
        await connection.execute(
            """
            UPDATE documents
            SET status = 'failed',updated_at = now()
            WHERE id = %s AND deleted_at IS NULL AND active_generation_id IS NULL
            """,
            (claim.document_id,),
        )
        await connection.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'failed',lease_token = NULL,lease_expires_at = NULL,
                error_code = %s,error_message = %s,updated_at = now()
            WHERE id = %s AND status = 'processing'
            """,
            (code, message, claim.job_id),
        )
