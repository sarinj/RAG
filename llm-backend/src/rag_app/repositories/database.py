from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from psycopg_pool import AsyncConnectionPool, PoolClosed, PoolTimeout

from rag_app.config import Settings
from rag_app.errors import DatabaseSchemaUnavailable, DatabaseUnavailable


class PsycopgDatabase:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: AsyncConnectionPool[AsyncConnection[tuple[Any, ...]]] | None = None

    @property
    def configured(self) -> bool:
        return self._settings.database_dsn is not None

    async def open(self) -> None:
        dsn = self._settings.database_dsn
        if dsn is None:
            return

        async def configure(connection: AsyncConnection[tuple[Any, ...]]) -> None:
            await register_vector_async(connection)

        pool: AsyncConnectionPool[AsyncConnection[tuple[Any, ...]]] = AsyncConnectionPool(
            conninfo=dsn,
            min_size=self._settings.database_pool_min_size,
            max_size=self._settings.database_pool_max_size,
            timeout=self._settings.database_pool_timeout_seconds,
            configure=configure,
            open=False,
        )
        self._pool = pool
        await pool.open()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection[tuple[Any, ...]], None]:
        if self._pool is None:
            raise DatabaseUnavailable("Database is not configured")
        try:
            async with self._pool.connection() as connection:
                yield connection
        except (PsycopgError, PoolClosed, PoolTimeout, TimeoutError) as exc:
            raise DatabaseUnavailable("Database is unavailable") from exc

    async def check_schema(self) -> int:
        async with self.connection() as connection:
            try:
                cursor = await connection.execute(
                    """
                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                    FROM pg_attribute AS attribute
                    WHERE attribute.attrelid = to_regclass('public.chunks')
                      AND attribute.attname = 'embedding'
                      AND attribute.attnum > 0
                      AND NOT attribute.attisdropped
                    """
                )
                row = await cursor.fetchone()
            except Exception as exc:
                raise DatabaseSchemaUnavailable("Shared database schema is unavailable") from exc
        if row is None or not isinstance(row[0], str):
            raise DatabaseSchemaUnavailable("Shared chunk embedding column is unavailable")
        match = re.fullmatch(r"vector\((\d+)\)", row[0])
        if match is None:
            raise DatabaseSchemaUnavailable("Shared chunk embedding dimension is not fixed")
        return int(match.group(1))

    async def validate_sources(self, sources: list[tuple[UUID, UUID]]) -> bool:
        if not sources:
            return False
        document_ids = [document_id for document_id, _ in sources]
        generation_ids = [generation_id for _, generation_id in sources]
        async with self.connection() as connection:
            try:
                cursor = await connection.execute(
                    """
                    SELECT count(*)
                    FROM unnest(%s::uuid[], %s::uuid[])
                         AS requested(document_id, generation_id)
                    JOIN documents AS document
                      ON document.id = requested.document_id
                    JOIN index_generations AS generation
                      ON generation.id = requested.generation_id
                     AND generation.document_id = document.id
                    WHERE document.status = 'ready'
                      AND document.deleted_at IS NULL
                      AND document.active_generation_id = requested.generation_id
                      AND generation.status = 'ready'
                    """,
                    (document_ids, generation_ids),
                )
                row = await cursor.fetchone()
            except Exception as exc:
                raise DatabaseSchemaUnavailable("Shared source schema is unavailable") from exc
        return row is not None and row[0] == len(sources)
