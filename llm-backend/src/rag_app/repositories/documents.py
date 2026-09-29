from __future__ import annotations

from rag_app.models import SourceGeneration
from rag_app.repositories.database import PsycopgDatabase


class DocumentRepository:
    def __init__(self, database: PsycopgDatabase) -> None:
        self._database = database

    async def snapshot_is_active(self, sources: list[SourceGeneration]) -> bool:
        return await self._database.validate_sources(
            [(source.document_id, source.generation_id) for source in sources]
        )
