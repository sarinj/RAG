from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from pgvector.psycopg import register_vector_async
from sqlalchemy import column, event, func, select, table, tuple_
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_app.config import Settings
from rag_app.errors import DatabaseSchemaUnavailable, DatabaseUnavailable
from rag_app.repositories.orm import Document, IndexGeneration

_PG_ATTRIBUTE = table(
    "pg_attribute",
    column("attrelid"),
    column("attname"),
    column("atttypid"),
    column("atttypmod"),
    column("attnum"),
    column("attisdropped"),
    schema="pg_catalog",
)


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def configured(self) -> bool:
        return self._settings.database_dsn is not None

    async def open(self) -> None:
        dsn = self._settings.database_dsn
        if dsn is None:
            return

        try:
            url = make_url(dsn)
            if url.drivername in {"postgres", "postgresql"}:
                url = url.set(drivername="postgresql+psycopg")
            elif url.drivername != "postgresql+psycopg":
                raise DatabaseUnavailable("DATABASE_URL must use PostgreSQL with psycopg")

            engine = create_async_engine(
                url,
                pool_size=self._settings.database_pool_max_size,
                max_overflow=0,
                pool_timeout=self._settings.database_pool_timeout_seconds,
            )

            @event.listens_for(engine.sync_engine, "connect")
            def register_vector(dbapi_connection: Any, _connection_record: Any) -> None:
                dbapi_connection.run_async(register_vector_async)

            self._engine = engine
            self._session_factory = async_sessionmaker(engine, expire_on_commit=False)
        except SQLAlchemyError as exc:
            raise DatabaseUnavailable("Database is unavailable") from exc

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session_factory is None:
            raise DatabaseUnavailable("Database is not configured")
        try:
            async with self._session_factory() as session:
                yield session
        except SQLAlchemyError as exc:
            raise DatabaseUnavailable("Database is unavailable") from exc

    async def check_schema(self) -> int:
        async with self.session() as session:
            try:
                statement = select(
                    func.format_type(_PG_ATTRIBUTE.c.atttypid, _PG_ATTRIBUTE.c.atttypmod)
                ).where(
                    _PG_ATTRIBUTE.c.attrelid == func.to_regclass("public.chunks"),
                    _PG_ATTRIBUTE.c.attname == "embedding",
                    _PG_ATTRIBUTE.c.attnum > 0,
                    _PG_ATTRIBUTE.c.attisdropped.is_(False),
                )
                result = await session.execute(statement)
                row = result.scalar_one_or_none()
            except Exception as exc:
                raise DatabaseSchemaUnavailable("Shared database schema is unavailable") from exc
        if not isinstance(row, str):
            raise DatabaseSchemaUnavailable("Shared chunk embedding column is unavailable")
        match = re.fullmatch(r"vector\((\d+)\)", row)
        if match is None:
            raise DatabaseSchemaUnavailable("Shared chunk embedding dimension is not fixed")
        return int(match.group(1))

    async def validate_sources(self, sources: list[tuple[UUID, UUID]]) -> bool:
        if not sources:
            return False
        statement = (
            select(func.count())
            .select_from(Document)
            .join(
                IndexGeneration,
                (IndexGeneration.document_id == Document.id)
                & (IndexGeneration.id == Document.active_generation_id),
            )
            .where(
                tuple_(Document.id, IndexGeneration.id).in_(sources),
                Document.status == "ready",
                Document.deleted_at.is_(None),
                IndexGeneration.status == "ready",
            )
        )
        async with self.session() as session:
            try:
                count = await session.scalar(statement)
            except SQLAlchemyError as exc:
                raise DatabaseSchemaUnavailable("Shared source schema is unavailable") from exc
        return count == len(sources)
