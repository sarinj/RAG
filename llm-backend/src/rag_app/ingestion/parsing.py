from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import UUID

from pypdf import PdfReader

from rag_app.ingestion.errors import IngestionFailure

_STORAGE_SUFFIXES = {"pdf", "txt", "md"}
_HEADING = re.compile(r"(?m)^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int
    value: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    text: str
    pages: tuple[TextSpan, ...]
    headings: tuple[TextSpan, ...]


def resolve_storage_path(upload_root: Path, storage_key: str) -> Path:
    try:
        file_id, suffix = storage_key.rsplit(".", 1)
        if suffix.lower() not in _STORAGE_SUFFIXES or str(UUID(file_id)) != file_id.lower():
            raise ValueError("invalid storage key")
        root = upload_root.resolve(strict=True)
        path = (root / storage_key).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("invalid storage path")
        return path
    except (OSError, ValueError) as exc:
        raise IngestionFailure("SOURCE_UNAVAILABLE", "The stored document is unavailable.") from exc


def parse_file(
    path: Path,
    expected_size: int,
    max_size: int,
    expected_hash: str,
) -> ParsedDocument:
    try:
        size = path.stat().st_size
        if size != expected_size or size > max_size:
            raise IngestionFailure("SOURCE_UNAVAILABLE", "The stored document is unavailable.")
        data = path.read_bytes()
    except IngestionFailure:
        raise
    except OSError as exc:
        raise IngestionFailure("SOURCE_UNAVAILABLE", "The stored document is unavailable.") from exc
    if sha256(data).hexdigest() != expected_hash:
        raise IngestionFailure(
            "SOURCE_UNAVAILABLE", "The stored document failed its integrity check."
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parsed = _parse_pdf(data)
    elif suffix in {".txt", ".md"}:
        parsed = _parse_text(data, markdown=suffix == ".md")
    else:
        raise IngestionFailure("UNSUPPORTED_FILE_TYPE", "This file type is not supported.")
    if not parsed.text.strip():
        code = "UNSUPPORTED_FILE_TYPE" if suffix == ".pdf" else "EMPTY_DOCUMENT"
        message = (
            "This PDF does not contain extractable text."
            if suffix == ".pdf"
            else "The document does not contain any text."
        )
        raise IngestionFailure(code, message)
    return parsed


def _normalize(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized)


def _parse_text(data: bytes, *, markdown: bool) -> ParsedDocument:
    try:
        text = _normalize(data.decode("utf-8-sig", errors="strict"))
    except UnicodeDecodeError as exc:
        raise IngestionFailure(
            "UNSUPPORTED_FILE_TYPE", "Text files must use UTF-8 encoding."
        ) from exc
    headings: tuple[TextSpan, ...] = ()
    if markdown:
        headings = tuple(
            TextSpan(match.start(), match.end(), cast(str, match.group(1)).strip())
            for match in _HEADING.finditer(text)
        )
    return ParsedDocument(text=text, pages=(), headings=headings)


def _parse_pdf(data: bytes) -> ParsedDocument:
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise IngestionFailure("UNSUPPORTED_FILE_TYPE", "Encrypted PDFs are not supported.")
        page_texts = [_normalize(page.extract_text() or "") for page in reader.pages]
    except IngestionFailure:
        raise
    except Exception as exc:
        raise IngestionFailure("UNSUPPORTED_FILE_TYPE", "The PDF could not be read.") from exc

    text_parts: list[str] = []
    pages: list[TextSpan] = []
    cursor = 0
    for index, page_text in enumerate(page_texts, start=1):
        if index > 1:
            text_parts.append("\n\n")
            cursor += 2
        start = cursor
        text_parts.append(page_text)
        cursor += len(page_text)
        pages.append(TextSpan(start, cursor, str(index)))
    text = "".join(text_parts)
    return ParsedDocument(text=text, pages=tuple(pages), headings=())
