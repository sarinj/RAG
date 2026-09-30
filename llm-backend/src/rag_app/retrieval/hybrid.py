from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

from rag_app.models import EvidenceRecord

BM25_CANDIDATE_LIMIT = 30
HYBRID_CANDIDATE_LIMIT = 30
RRF_K = 60


def bm25_candidates(
    query: str,
    records: Sequence[EvidenceRecord],
    tokenizer: Tokenizer,
    *,
    limit: int = BM25_CANDIDATE_LIMIT,
) -> list[EvidenceRecord]:
    tokenized_corpus = [_tokens(record.chunk.searchable_text, tokenizer) for record in records]
    nonempty = [
        (record, tokens)
        for record, tokens in zip(records, tokenized_corpus, strict=True)
        if tokens
    ]
    query_tokens = _tokens(query, tokenizer)
    if not nonempty or not query_tokens:
        return []

    model = BM25Okapi([tokens for _, tokens in nonempty])
    scores = model.get_scores(query_tokens)
    query_terms = set(query_tokens)
    ranked = [
        (record, float(score))
        for (record, tokens), score in zip(nonempty, scores, strict=True)
        if query_terms.intersection(tokens)
    ]
    ranked.sort(key=lambda item: (-item[1], str(_evidence_key(item[0]))))
    return [record for record, _ in ranked[:limit]]


def reciprocal_rank_fusion(
    vector_records: Sequence[EvidenceRecord],
    lexical_records: Sequence[EvidenceRecord],
    *,
    limit: int = HYBRID_CANDIDATE_LIMIT,
    rrf_k: int = RRF_K,
) -> list[EvidenceRecord]:
    records: dict[tuple[UUID, UUID, UUID], EvidenceRecord] = {}
    scores: dict[tuple[UUID, UUID, UUID], float] = {}
    for ranking in (vector_records, lexical_records):
        for rank, record in enumerate(ranking, start=1):
            key = _evidence_key(record)
            records[key] = record
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)

    ordered = sorted(records, key=lambda key: (-scores[key], str(key)))
    return [records[key] for key in ordered[:limit]]


def _tokens(text: str, tokenizer: Tokenizer) -> list[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False).tokens
    normalized = [token.casefold().lstrip("▁Ġ") for token in tokens]
    return [token for token in normalized if any(character.isalnum() for character in token)]


def _evidence_key(record: EvidenceRecord) -> tuple[UUID, UUID, UUID]:
    return record.document_id, record.chunk.generation_id, record.chunk.id
