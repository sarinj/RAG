from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

from rag_app.config import IndexConfiguration, Settings, load_index_configuration
from rag_app.errors import (
    DatabaseSchemaUnavailable,
    DatabaseUnavailable,
    ProviderError,
)
from rag_app.ingestion.chunking import CHUNKER_VERSION, IngestionChunk, chunk_document
from rag_app.ingestion.contextualization import (
    CONTEXT_PROMPT_VERSION,
    ContextualizationLimitError,
    ContextualizationOutputError,
    contextualize_chunk,
)
from rag_app.ingestion.errors import IngestionFailure, LeaseLost, RetryableIngestionFailure
from rag_app.ingestion.parsing import parse_file, resolve_storage_path
from rag_app.providers.factory import create_provider
from rag_app.providers.models import EmbeddingInput
from rag_app.providers.protocols import LLMProvider, ProviderAdapter
from rag_app.repositories.database import Database
from rag_app.repositories.ingestion import (
    LEASE_SECONDS,
    MAX_ATTEMPTS,
    IngestionRepository,
    JobClaim,
    StagedChunk,
    WorkerJob,
)
from rag_app.tokenization import load_tokenizer

logger = logging.getLogger("rag_app.worker")
# Persist progress after each chunk so the document list can display real chunk counts.
EMBEDDING_BATCH_SIZE = 1
OPENAI_EMBEDDING_BATCH_SIZE = 32
HEARTBEAT_SECONDS = 20


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    chunk: IngestionChunk
    context_text: str | None
    searchable_text: str


def _emit(event: str, **fields: object) -> None:
    logger.info(
        json.dumps(
            {"event": event, **fields},
            separators=(",", ":"),
            default=str,
        )
    )


def _load_tokenizer(settings: Settings, config: IndexConfiguration) -> Tokenizer:
    if config.chunker_version != CHUNKER_VERSION:
        raise ValueError("The configured chunker version is not supported by this worker")
    return load_tokenizer(settings.tokenizer_path, config.tokenizer_version)


async def _check_dependencies() -> int:
    settings = Settings()
    database = Database(settings)
    provider: ProviderAdapter | None = None
    try:
        if settings.upload_dir is None:
            raise ValueError("UPLOAD_DIR is not configured")
        if not settings.upload_dir.is_dir():
            raise ValueError("UPLOAD_DIR is unavailable")
        config = load_index_configuration(settings.index_config_path)
        _load_tokenizer(settings, config)
        provider = create_provider(settings, embedding_dimension=config.embedding_dimension)
        if settings.database_dsn is None:
            raise ValueError("DATABASE_URL is not configured")
        await database.open()
        dimension = await database.check_schema()
        if dimension != config.embedding_dimension:
            raise ValueError("index_dimension_mismatch")
        await provider.check_readiness(config, require_generation=False)
    except (DatabaseSchemaUnavailable, DatabaseUnavailable, ProviderError, ValueError) as exc:
        print(json.dumps({"workerStatus": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 1
    finally:
        await database.close()
        if provider is not None:
            await provider.close()
    print(json.dumps({"workerStatus": "dependencies_ready"}))
    return 0


def _required_int(config: Mapping[str, Any], key: str, *, minimum: int = 1) -> int:
    value = config.get(key)
    if type(value) is not int or value < minimum:
        raise IngestionFailure("INVALID_INDEX_CONFIG", "The generation configuration is invalid.")
    return value


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise IngestionFailure("INVALID_INDEX_CONFIG", "The generation configuration is invalid.")
    return value


class IngestionWorker:
    def __init__(
        self,
        settings: Settings,
        config: IndexConfiguration,
        tokenizer: Tokenizer,
        database: Database,
        provider: LLMProvider,
        stop_event: asyncio.Event,
    ) -> None:
        self._settings = settings
        self._config = config
        self._tokenizer = tokenizer
        self._repository = IngestionRepository(database)
        self._provider = provider
        self._stop = stop_event

    async def run(self) -> None:
        _emit("worker_started", leaseSeconds=LEASE_SECONDS, heartbeatSeconds=HEARTBEAT_SECONDS)
        while not self._stop.is_set():
            await self._repository.fail_expired_exhausted()
            claim = await self._repository.claim_next()
            if claim is None:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._settings.worker_poll_interval_seconds
                    )
                except TimeoutError:
                    pass
                continue
            _emit(
                "job_claimed",
                jobId=claim.job_id,
                documentId=claim.document_id,
                generationId=claim.generation_id,
                attempt=claim.attempts,
            )
            await self._process_with_heartbeat(claim)
        _emit("worker_stopped")

    async def _process_with_heartbeat(self, claim: JobClaim) -> None:
        finished = asyncio.Event()
        processing = asyncio.create_task(self._process(claim))
        heartbeat = asyncio.create_task(self._heartbeat(claim, finished))
        try:
            done, _ = await asyncio.wait(
                {processing, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                still_owned = heartbeat.result()
                if not still_owned and not processing.done():
                    processing.cancel()
                    await asyncio.gather(processing, return_exceptions=True)
                    _emit("job_lease_lost", jobId=claim.job_id)
                    return
            await processing
        finally:
            finished.set()
            for task in (processing, heartbeat):
                if not task.done():
                    task.cancel()
            await asyncio.gather(processing, heartbeat, return_exceptions=True)

    async def _heartbeat(self, claim: JobClaim, finished: asyncio.Event) -> bool:
        while True:
            try:
                await asyncio.wait_for(finished.wait(), timeout=HEARTBEAT_SECONDS)
                return True
            except TimeoutError:
                pass
            if not await self._repository.heartbeat(claim):
                return False

    async def _process(self, claim: JobClaim) -> None:
        try:
            job = await self._repository.begin(claim)
            if job.mode not in {"baseline", "contextual"}:
                raise IngestionFailure(
                    "UNSUPPORTED_INDEX_MODE",
                    "This worker cannot process the configured index mode.",
                )
            target_tokens = _required_int(job.config, "chunkTargetTokens")
            overlap_tokens = _required_int(job.config, "chunkOverlapTokens", minimum=0)
            if overlap_tokens >= target_tokens:
                raise IngestionFailure(
                    "INVALID_INDEX_CONFIG", "The generation configuration is invalid."
                )
            if _required_string(job.config, "chunkerVersion") != CHUNKER_VERSION:
                raise IngestionFailure(
                    "UNSUPPORTED_CHUNKER_VERSION",
                    "This worker cannot process the configured chunker version.",
                )
            if (
                _required_string(job.config, "tokenizerVersion")
                != self._config.tokenizer_version
            ):
                raise IngestionFailure(
                    "UNSUPPORTED_TOKENIZER_VERSION",
                    "This worker cannot load the configured tokenizer version.",
                )
            if (
                job.embedding_model != self._config.embedding_model
                or job.embedding_digest.lower() != self._config.embedding_digest.lower()
                or job.embedding_dimension != self._config.embedding_dimension
            ):
                raise IngestionFailure(
                    "EMBEDDING_CONFIG_MISMATCH",
                    "The generation uses a different embedding configuration.",
                )
            if job.mode == "contextual":
                self._validate_contextual_config(job)
            await self._ingest(job, target_tokens, overlap_tokens)
            _emit("job_published", jobId=claim.job_id, generationId=claim.generation_id)
        except asyncio.CancelledError:
            raise
        except LeaseLost:
            _emit("job_abandoned", jobId=claim.job_id, reason="lease_lost_or_cancelled")
        except RetryableIngestionFailure as exc:
            await self._record_failure(claim, exc, retryable=True)
        except IngestionFailure as exc:
            await self._record_failure(claim, exc, retryable=False)
        except (ProviderError, DatabaseUnavailable) as exc:
            retry = RetryableIngestionFailure(
                "INGESTION_RETRYABLE", "A required service is temporarily unavailable."
            )
            await self._record_failure(claim, retry, retryable=True)
            _emit("job_dependency_error", jobId=claim.job_id, errorType=type(exc).__name__)
        except Exception as exc:
            failure = IngestionFailure(
                "INGESTION_FAILED", "The document could not be processed."
            )
            await self._record_failure(claim, failure, retryable=False)
            _emit("job_processing_error", jobId=claim.job_id, errorType=type(exc).__name__)

    async def _ingest(self, job: WorkerJob, target_tokens: int, overlap_tokens: int) -> None:
        upload_root = self._settings.upload_dir
        if upload_root is None:
            raise IngestionFailure("SOURCE_UNAVAILABLE", "The stored document is unavailable.")
        source = resolve_storage_path(upload_root, job.storage_key)
        document = await asyncio.to_thread(
            parse_file,
            source,
            job.size_bytes,
            self._settings.max_upload_bytes,
            job.content_hash,
        )
        await self._repository.set_stage(job.claim, "chunking")
        chunks = await asyncio.to_thread(
            chunk_document, document, self._tokenizer, target_tokens, overlap_tokens
        )
        if not chunks:
            raise IngestionFailure("EMPTY_DOCUMENT", "The document does not contain any text.")
        await self._repository.check_capacity(job.claim, len(chunks))

        prepared_chunks: list[PreparedChunk] = []
        if job.mode == "contextual":
            await self._repository.set_stage(
                job.claim, "contextualizing", total_chunks=len(chunks)
            )
            context_window_tokens = _required_int(job.config, "contextWindowTokens")
            max_output_tokens = _required_int(job.config, "maxOutputTokens")
            for chunk in chunks:
                try:
                    context_text = await contextualize_chunk(
                        self._provider,
                        self._tokenizer,
                        document_text=document.text,
                        chunk=chunk,
                        context_window_tokens=context_window_tokens,
                        max_output_tokens=max_output_tokens,
                    )
                except ContextualizationLimitError as exc:
                    raise IngestionFailure(
                        "CONTEXT_LIMIT_EXCEEDED",
                        "The document exceeds the configured model context limit.",
                    ) from exc
                except ContextualizationOutputError as exc:
                    raise RetryableIngestionFailure(
                        "CONTEXTUALIZATION_FAILED",
                        "The document could not be contextualized.",
                    ) from exc
                prepared_chunks.append(
                    PreparedChunk(
                        chunk=chunk,
                        context_text=context_text,
                        searchable_text=f"{context_text}\n\n{chunk.original_text}",
                    )
                )
        else:
            prepared_chunks = [
                PreparedChunk(chunk=chunk, context_text=None, searchable_text=chunk.original_text)
                for chunk in chunks
            ]

        await self._repository.set_stage(job.claim, "embedding", total_chunks=len(chunks))

        batch_size = (
            OPENAI_EMBEDDING_BATCH_SIZE
            if self._settings.active_embedding_provider == "openai"
            else EMBEDDING_BATCH_SIZE
        )
        for offset in range(0, len(prepared_chunks), batch_size):
            batch = prepared_chunks[offset : offset + batch_size]
            embeddings = await self._embed_batch(job, batch)
            await self._repository.stage_batch(
                job.claim,
                [
                    StagedChunk(
                        chunk=item.chunk,
                        context_text=item.context_text,
                        searchable_text=item.searchable_text,
                        embedding=vector,
                    )
                    for item, vector in zip(batch, embeddings, strict=True)
                ],
            )
            _emit(
                "job_progress",
                jobId=job.claim.job_id,
                processedChunks=min(offset + len(batch), len(prepared_chunks)),
                totalChunks=len(chunks),
            )
        await self._repository.set_stage(job.claim, "publishing")
        await self._repository.publish(job.claim)

    async def _embed_batch(
        self,
        job: WorkerJob,
        chunks: list[PreparedChunk],
    ) -> list[list[float]]:
        response = await self._provider.embed(
            EmbeddingInput(texts=[chunk.searchable_text for chunk in chunks])
        )
        if (
            response.dimension != job.embedding_dimension
            or len(response.vectors) != len(chunks)
            or response.model != job.embedding_model
        ):
            raise IngestionFailure(
                "EMBEDDING_RESPONSE_INVALID", "The embedding service returned an invalid response."
            )
        return response.vectors

    def _validate_contextual_config(self, job: WorkerJob) -> None:
        prompt_version = _required_string(job.config, "contextPromptVersion")
        context_model = _required_string(job.config, "contextModel")
        context_digest = _required_string(job.config, "contextModelDigest")
        if prompt_version != CONTEXT_PROMPT_VERSION:
            raise IngestionFailure(
                "UNSUPPORTED_CONTEXT_PROMPT",
                "This worker cannot process the configured context prompt version.",
            )
        if (
            prompt_version != self._config.context_prompt_version
            or context_model != self._config.context_model
            or context_digest.lower() != (self._config.context_model_digest or "").lower()
        ):
            raise IngestionFailure(
                "CONTEXT_CONFIG_MISMATCH",
                "The generation uses a different contextualization configuration.",
            )
        if context_model != self._settings.generation_model:
            raise IngestionFailure(
                "CONTEXT_MODEL_MISMATCH",
                "The configured contextualization model is unavailable.",
            )
        context_window = _required_int(job.config, "contextWindowTokens")
        if context_window != self._config.context_window_tokens:
            raise IngestionFailure(
                "CONTEXT_CONFIG_MISMATCH",
                "The generation uses a different contextualization context window.",
            )

    async def _record_failure(
        self,
        claim: JobClaim,
        failure: IngestionFailure,
        *,
        retryable: bool,
    ) -> None:
        recorded = await self._repository.fail(
            claim,
            code=failure.code,
            message=failure.message,
            retryable=retryable,
        )
        if recorded:
            _emit(
                "job_retry_scheduled"
                if retryable and claim.attempts < MAX_ATTEMPTS
                else "job_failed",
                jobId=claim.job_id,
                attempt=claim.attempts,
                errorCode=failure.code,
            )


async def _run_worker() -> int:
    settings = Settings()
    if settings.upload_dir is None or not settings.upload_dir.is_dir():
        print(
            json.dumps({"workerStatus": "blocked", "reason": "UPLOAD_DIR is unavailable"}),
            file=sys.stderr,
        )
        return 1
    try:
        config = load_index_configuration(settings.index_config_path)
        tokenizer = _load_tokenizer(settings, config)
    except ValueError as exc:
        print(json.dumps({"workerStatus": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 1

    database = Database(settings)
    provider = create_provider(settings, embedding_dimension=config.embedding_dimension)
    try:
        if settings.database_dsn is None:
            raise ValueError("DATABASE_URL is not configured")
        await database.open()
        dimension = await database.check_schema()
        if dimension != config.embedding_dimension:
            raise ValueError("index_dimension_mismatch")
        await provider.check_readiness(config, require_generation=False)
    except (DatabaseSchemaUnavailable, DatabaseUnavailable, ProviderError, ValueError) as exc:
        print(json.dumps({"workerStatus": "blocked", "reason": str(exc)}), file=sys.stderr)
        await database.close()
        await provider.close()
        return 1

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stopped_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stopped_signal, stop_event.set)
        except NotImplementedError:
            pass
    worker = IngestionWorker(settings, config, tokenizer, database, provider, stop_event)
    try:
        await worker.run()
    finally:
        await database.close()
        await provider.close()
    return 0


async def _async_main(check_only: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if check_only:
        return await _check_dependencies()
    return await _run_worker()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM ingestion worker")
    parser.add_argument("--check", action="store_true", help="check worker dependencies only")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args.check)))


if __name__ == "__main__":
    main()
