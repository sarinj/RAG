from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.errors import DatabaseUnavailable
from rag_app.ingestion.chunking import IngestionChunk
from rag_app.ingestion.errors import CorpusCapacityExceeded, LeaseLost
from rag_app.repositories.database import Database
from rag_app.repositories.orm import Chunk, Document, IndexGeneration, IngestionJob

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
    context_text: str | None
    searchable_text: str
    embedding: Sequence[float]


class IngestionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def claim_next(self) -> JobClaim | None:
        async with self._database.session() as session:
            async with session.begin():
                statement = (
                    select(IngestionJob)
                    .where(
                        IngestionJob.attempts < MAX_ATTEMPTS,
                        or_(
                            IngestionJob.status == "queued",
                            (IngestionJob.status == "processing")
                            & (IngestionJob.lease_expires_at <= func.now()),
                        ),
                    )
                    .order_by(IngestionJob.created_at, IngestionJob.id)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                job = await session.scalar(statement)
                if job is None:
                    return None

                lease_token = uuid4()
                job.status = "processing"
                job.attempts += 1
                job.lease_token = lease_token
                job.lease_expires_at = func.now() + timedelta(seconds=LEASE_SECONDS)
                job.heartbeat_at = func.now()
                job.error_code = None
                job.error_message = None
                job.updated_at = func.now()
                return JobClaim(
                    job_id=job.id,
                    document_id=job.document_id,
                    generation_id=job.generation_id,
                    lease_token=lease_token,
                    attempts=job.attempts,
                )

    async def begin(self, claim: JobClaim) -> WorkerJob:
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                self._require_live_lease(document, job, lease_is_live, claim)
                if document is None or job is None:
                    raise LeaseLost("The document or ingestion job no longer exists")

                generation = await session.scalar(
                    select(IndexGeneration)
                    .where(
                        IndexGeneration.id == claim.generation_id,
                        IndexGeneration.document_id == claim.document_id,
                    )
                    .with_for_update()
                )
                if generation is None or generation.status not in {"queued", "processing"}:
                    raise LeaseLost("The generation is no longer active")

                await session.scalar(select(func.pg_advisory_xact_lock(CHUNK_CAPACITY_LOCK)))
                await session.execute(
                    delete(Chunk).where(Chunk.generation_id == claim.generation_id)
                )
                generation.status = "processing"
                generation.updated_at = func.now()
                if document.active_generation_id is None:
                    document.status = "processing"
                document.updated_at = func.now()
                job.stage = "parsing"
                job.processed_chunks = 0
                job.total_chunks = None
                job.error_code = None
                job.error_message = None
                job.updated_at = func.now()

                raw_config: Any = generation.config
                if isinstance(raw_config, str):
                    raw_config = json.loads(raw_config)
                if not isinstance(raw_config, Mapping):
                    raise DatabaseUnavailable("Generation configuration is invalid")
                generation_config = cast(Mapping[str, Any], raw_config)
                return WorkerJob(
                    claim=claim,
                    storage_key=document.storage_key,
                    size_bytes=int(document.size_bytes),
                    content_hash=document.content_hash,
                    mode=generation.mode,
                    embedding_model=generation.embedding_model,
                    embedding_digest=generation.embedding_digest,
                    embedding_dimension=generation.embedding_dimension,
                    config=generation_config,
                )

    async def heartbeat(self, claim: JobClaim) -> bool:
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                if (
                    document is None
                    or document.deleted_at is not None
                    or document.status == "deleted"
                    or not self._owns_live_lease(job, lease_is_live, claim)
                ):
                    return False
                assert job is not None
                job.lease_expires_at = func.now() + timedelta(seconds=LEASE_SECONDS)
                job.heartbeat_at = func.now()
                job.updated_at = func.now()
                return True

    async def set_stage(
        self,
        claim: JobClaim,
        stage: IngestionStage,
        *,
        total_chunks: int | None = None,
    ) -> None:
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                self._require_live_lease(document, job, lease_is_live, claim)
                assert job is not None
                job.stage = stage
                if total_chunks is not None:
                    job.total_chunks = total_chunks
                job.updated_at = func.now()

    async def check_capacity(self, claim: JobClaim, additional_chunks: int) -> None:
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                self._require_live_lease(document, job, lease_is_live, claim)
                await self._lock_chunk_capacity(session)
                row = (
                    await session.execute(
                        select(func.count(Chunk.id), func.rag_chunk_capacity_limit())
                    )
                ).one_or_none()
                if row is None or row[0] + additional_chunks > row[1]:
                    raise CorpusCapacityExceeded()

    async def stage_batch(self, claim: JobClaim, staged: Sequence[StagedChunk]) -> None:
        if not staged:
            return
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                self._require_live_lease(document, job, lease_is_live, claim)
                await self._lock_chunk_capacity(session)
                row = (
                    await session.execute(
                        select(func.count(Chunk.id), func.rag_chunk_capacity_limit())
                    )
                ).one_or_none()
                if row is None or row[0] + len(staged) > row[1]:
                    raise CorpusCapacityExceeded()
                await session.scalar(select(func.rag_reserve_chunk_capacity(len(staged))))

                session.add_all(
                    [
                        Chunk(
                            id=uuid4(),
                            generation_id=claim.generation_id,
                            ordinal=item.chunk.ordinal,
                            original_text=item.chunk.original_text,
                            context_text=item.context_text,
                            searchable_text=item.searchable_text,
                            page_start=item.chunk.page_start,
                            page_end=item.chunk.page_end,
                            heading=item.chunk.heading,
                            source_start=item.chunk.source_start,
                            source_end=item.chunk.source_end,
                            token_count=item.chunk.token_count,
                            embedding=list(item.embedding),
                        )
                        for item in staged
                    ]
                )
                await session.flush()
                assert job is not None
                job.processed_chunks += len(staged)
                job.updated_at = func.now()

    async def publish(self, claim: JobClaim) -> None:
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                if job is None:
                    raise LeaseLost("The ingestion job no longer exists")
                self._require_live_lease(document, job, lease_is_live, claim)
                if job.total_chunks is None or job.processed_chunks != job.total_chunks:
                    raise LeaseLost("The staged chunk count is incomplete")
                count = await session.scalar(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.generation_id == claim.generation_id)
                )
                if count != job.total_chunks:
                    raise LeaseLost("The staged chunk count changed")

                await session.execute(
                    update(IndexGeneration)
                    .where(
                        IndexGeneration.id == claim.generation_id,
                        IndexGeneration.document_id == claim.document_id,
                        IndexGeneration.status == "processing",
                    )
                    .values(status="ready", updated_at=func.now())
                )
                await session.execute(
                    update(Document)
                    .where(Document.id == claim.document_id, Document.deleted_at.is_(None))
                    .values(
                        status="ready",
                        active_generation_id=claim.generation_id,
                        updated_at=func.now(),
                    )
                )
                await session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == claim.job_id)
                    .values(
                        status="succeeded",
                        stage="done",
                        lease_token=None,
                        lease_expires_at=None,
                        heartbeat_at=func.now(),
                        error_code=None,
                        error_message=None,
                        updated_at=func.now(),
                    )
                )

    async def fail(
        self,
        claim: JobClaim,
        *,
        code: str,
        message: str,
        retryable: bool,
    ) -> bool:
        async with self._database.session() as session:
            async with session.begin():
                document, job, lease_is_live = await self._lock_document_job(session, claim)
                if (
                    document is None
                    or document.deleted_at is not None
                    or document.status == "deleted"
                    or not self._owns_live_lease(job, lease_is_live, claim)
                ):
                    return False
                assert job is not None
                if retryable and claim.attempts < MAX_ATTEMPTS:
                    job.lease_expires_at = func.now()
                    job.error_code = code
                    job.error_message = message
                    job.updated_at = func.now()
                    return True
                await self._mark_failed(session, claim, code, message)
                return True

    async def fail_expired_exhausted(self, batch_size: int = 16) -> int:
        async with self._database.session() as session:
            result = await session.scalars(
                select(IngestionJob)
                .where(
                    IngestionJob.status == "processing",
                    IngestionJob.attempts >= MAX_ATTEMPTS,
                    IngestionJob.lease_expires_at <= func.now(),
                )
                .order_by(IngestionJob.created_at, IngestionJob.id)
                .limit(batch_size)
            )
            candidates = [
                JobClaim(
                    job_id=job.id,
                    document_id=job.document_id,
                    generation_id=job.generation_id,
                    lease_token=job.lease_token,
                    attempts=job.attempts,
                )
                for job in result
                if job.lease_token is not None
            ]

        failed = 0
        for claim in candidates:
            async with self._database.session() as session:
                async with session.begin():
                    document, job, lease_is_live = await self._lock_document_job(session, claim)
                    if (
                        document is None
                        or document.deleted_at is not None
                        or document.status == "deleted"
                        or job is None
                        or job.status != "processing"
                        or job.attempts < MAX_ATTEMPTS
                        or lease_is_live
                    ):
                        continue
                    await self._mark_failed(
                        session,
                        claim,
                        "WORKER_ATTEMPTS_EXHAUSTED",
                        "The worker could not process this document after several attempts.",
                    )
                    failed += 1
        return failed

    async def _lock_document_job(
        self,
        session: AsyncSession,
        claim: JobClaim,
    ) -> tuple[Document | None, IngestionJob | None, bool]:
        document = await session.scalar(
            select(Document).where(Document.id == claim.document_id).with_for_update()
        )
        result = await session.execute(
            select(IngestionJob, IngestionJob.lease_expires_at > func.now())
            .where(
                IngestionJob.id == claim.job_id,
                IngestionJob.document_id == claim.document_id,
            )
            .with_for_update()
        )
        row = result.one_or_none()
        if row is None:
            return document, None, False
        job, lease_is_live = row
        return document, job, bool(lease_is_live)

    @staticmethod
    def _owns_live_lease(
        job: IngestionJob | None,
        lease_is_live: bool,
        claim: JobClaim,
    ) -> bool:
        return bool(
            job is not None
            and job.status == "processing"
            and job.lease_token == claim.lease_token
            and lease_is_live
        )

    def _require_live_lease(
        self,
        document: Document | None,
        job: IngestionJob | None,
        lease_is_live: bool,
        claim: JobClaim,
    ) -> None:
        if (
            document is None
            or document.deleted_at is not None
            or document.status == "deleted"
            or not self._owns_live_lease(job, lease_is_live, claim)
        ):
            raise LeaseLost("The worker no longer owns this ingestion job")

    @staticmethod
    async def _lock_chunk_capacity(session: AsyncSession) -> None:
        await session.scalar(select(func.pg_advisory_xact_lock(CHUNK_CAPACITY_LOCK)))

    async def _mark_failed(
        self,
        session: AsyncSession,
        claim: JobClaim,
        code: str,
        message: str,
    ) -> None:
        await session.execute(
            update(IndexGeneration)
            .where(
                IndexGeneration.id == claim.generation_id,
                IndexGeneration.document_id == claim.document_id,
                IndexGeneration.status.in_(("queued", "processing")),
            )
            .values(status="failed", updated_at=func.now())
        )
        await session.execute(
            update(Document)
            .where(
                Document.id == claim.document_id,
                Document.deleted_at.is_(None),
                Document.active_generation_id.is_(None),
            )
            .values(status="failed", updated_at=func.now())
        )
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == claim.job_id, IngestionJob.status == "processing")
            .values(
                status="failed",
                lease_token=None,
                lease_expires_at=None,
                error_code=code,
                error_message=message,
                updated_at=func.now(),
            )
        )
