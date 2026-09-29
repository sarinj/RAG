from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from tokenizers import Tokenizer

from rag_app.ingestion.parsing import ParsedDocument, TextSpan

CHUNKER_VERSION = "structure-token-offsets-v1"
_STRUCTURAL_BOUNDARY = re.compile(
    r"\n[ \t]*\n|(?<=[.!?])(?:[ \t]+|\n)|\n(?=#{1,6}[ \t]+)"
)


@dataclass(frozen=True, slots=True)
class IngestionChunk:
    ordinal: int
    original_text: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    source_start: int
    source_end: int
    token_count: int


def chunk_document(
    document: ParsedDocument,
    tokenizer: Tokenizer,
    target_tokens: int,
    overlap_tokens: int,
) -> list[IngestionChunk]:
    if target_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("invalid chunk token budget")
    if not document.text:
        return []

    encoding = tokenizer.encode(document.text, add_special_tokens=False)
    offsets = [offset for offset in encoding.offsets if offset[1] > offset[0]]
    if not offsets:
        return []
    if len(offsets) != len(encoding.ids):
        raise ValueError("tokenizer returned unaligned token offsets")

    ends = [end for _, end in offsets]
    boundaries = [match.end() for match in _STRUCTURAL_BOUNDARY.finditer(document.text)]
    chunks: list[IngestionChunk] = []
    token_start = 0
    while token_start < len(offsets):
        hard_end = min(token_start + target_tokens, len(offsets))
        token_end = _choose_end(token_start, hard_end, ends, boundaries, target_tokens)
        source_start = offsets[token_start][0]
        source_end = offsets[token_end - 1][1]
        if source_end <= source_start:
            raise ValueError("tokenizer returned invalid source offsets")
        page_numbers = [
            int(span.value)
            for span in document.pages
            if span.end > source_start and span.start < source_end and span.value.isdecimal()
        ]
        heading = _heading_at(document.headings, source_start)
        chunks.append(
            IngestionChunk(
                ordinal=len(chunks),
                original_text=document.text[source_start:source_end],
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                heading=heading,
                source_start=source_start,
                source_end=source_end,
                token_count=token_end - token_start,
            )
        )
        if token_end == len(offsets):
            break
        token_start = max(token_start + 1, token_end - overlap_tokens)
    return chunks


def _choose_end(
    token_start: int,
    hard_end: int,
    token_ends: list[int],
    boundaries: list[int],
    target_tokens: int,
) -> int:
    if hard_end - token_start <= 1:
        return hard_end
    earliest = token_start + max(1, target_tokens // 2)
    selected = hard_end
    for boundary in boundaries:
        if boundary > token_ends[hard_end - 1]:
            break
        candidate = bisect_right(token_ends, boundary)
        if earliest <= candidate <= hard_end:
            selected = candidate
    return selected


def _heading_at(headings: tuple[TextSpan, ...], source_start: int) -> str | None:
    current: str | None = None
    for heading in headings:
        if heading.start > source_start:
            break
        current = heading.value
    return current
