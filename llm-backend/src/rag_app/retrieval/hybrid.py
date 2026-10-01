from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingTypeStubs]
from tokenizers import Tokenizer

from rag_app.models import EvidenceRecord, RetrievalMetadata
from rag_app.repositories.retrieval import evidence_key

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
    return bm25_rankings([query], records, tokenizer, limit=limit)[0]


def bm25_rankings(
    queries: Sequence[str],
    records: Sequence[EvidenceRecord],
    tokenizer: Tokenizer,
    *,
    limit: int = BM25_CANDIDATE_LIMIT,
) -> list[list[EvidenceRecord]]:
    """Build the scoped lexical corpus once, then search each query independently."""
    tokenized_corpus = [_tokens(record.chunk.searchable_text, tokenizer) for record in records]
    nonempty = [
        (record, tokens) for record, tokens in zip(records, tokenized_corpus, strict=True) if tokens
    ]
    if not nonempty:
        return [[] for _ in queries]

    model: Any = BM25Okapi([tokens for _, tokens in nonempty])
    rankings: list[list[EvidenceRecord]] = []
    for query in queries:
        query_tokens = _tokens(query, tokenizer)
        scores = model.get_scores(query_tokens)
        query_terms = set(query_tokens)
        ranked = [
            (record, float(score))
            for (record, tokens), score in zip(nonempty, scores, strict=True)
            if query_terms.intersection(tokens)
        ]
        ranked.sort(key=lambda item: (-item[1], str(evidence_key(item[0]))))
        rankings.append(
            [
                record.model_copy(
                    update={
                        "retrieval": record.retrieval.model_copy(
                            update={
                                "lexical_score": score,
                                "lexical_rank": rank,
                            }
                        )
                    }
                )
                for rank, (record, score) in enumerate(ranked[:limit], 1)
            ]
        )
    return rankings


def reciprocal_rank_fusion(
    vector_records: Sequence[EvidenceRecord],
    lexical_records: Sequence[EvidenceRecord],
    *,
    limit: int = HYBRID_CANDIDATE_LIMIT,
    rrf_k: int = RRF_K,
) -> list[EvidenceRecord]:
    return fuse_rankings([vector_records, lexical_records], limit=limit, rrf_k=rrf_k)


def merge_metadata(left: RetrievalMetadata, right: RetrievalMetadata) -> RetrievalMetadata:
    hits = list(left.query_hits)
    for hit in right.query_hits:
        if hit not in hits:
            hits.append(hit)

    def highest(a: float | None, b: float | None) -> float | None:
        values = [value for value in (a, b) if value is not None]
        return max(values) if values else None

    def first(a: int | None, b: int | None) -> int | None:
        values = [value for value in (a, b) if value is not None]
        return min(values) if values else None

    return left.model_copy(
        update={
            "vector_score": highest(left.vector_score, right.vector_score),
            "lexical_score": highest(left.lexical_score, right.lexical_score),
            "vector_rank": first(left.vector_rank, right.vector_rank),
            "lexical_rank": first(left.lexical_rank, right.lexical_rank),
            "query_hits": hits,
        }
    )


def fuse_rankings(
    rankings: Sequence[Sequence[EvidenceRecord]],
    *,
    limit: int = HYBRID_CANDIDATE_LIMIT,
    rrf_k: int = RRF_K,
) -> list[EvidenceRecord]:
    """Deduplicate chunk identities while retaining scores and every query/channel hit."""
    records: dict[tuple[UUID, UUID, UUID], EvidenceRecord] = {}
    scores: dict[tuple[UUID, UUID, UUID], float] = {}
    for ranking in rankings:
        seen: set[tuple[UUID, UUID, UUID]] = set()
        for rank, record in enumerate(ranking, start=1):
            key = evidence_key(record)
            if key in seen:
                continue
            seen.add(key)
            previous = records.get(key)
            records[key] = (
                record.model_copy(
                    update={"retrieval": merge_metadata(previous.retrieval, record.retrieval)}
                )
                if previous
                else record
            )
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)

    ordered = sorted(records, key=lambda key: (-scores[key], str(key)))
    return [
        records[key].model_copy(
            update={
                "retrieval": records[key].retrieval.model_copy(
                    update={
                        "rrf_score": scores[key],
                        "fused_rank": rank,
                    }
                )
            }
        )
        for rank, key in enumerate(ordered[:limit], 1)
    ]


def _tokens(text: str, tokenizer: Tokenizer) -> list[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False).tokens
    normalized = [token.casefold().lstrip("▁Ġ") for token in tokens]
    return [token for token in normalized if any(character.isalnum() for character in token)]
