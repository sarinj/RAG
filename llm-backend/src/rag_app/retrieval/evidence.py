from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID

from rag_app.models import EvidenceRecord, SourceGeneration
from rag_app.repositories.retrieval import evidence_key


def expand_neighbors(
    ranked: Sequence[EvidenceRecord],
    neighbors: Sequence[EvidenceRecord],
    sources: Sequence[SourceGeneration],
    *,
    window: int,
    top_n: int,
) -> list[EvidenceRecord]:
    """Defend the scope boundary again before adding same-generation ordered neighbors."""
    allowed = {(source.document_id, source.generation_id) for source in sources}
    anchors = [
        item for item in ranked[:top_n] if (item.document_id, item.chunk.generation_id) in allowed
    ]
    merged = {
        evidence_key(item): item
        for item in ranked
        if (item.document_id, item.chunk.generation_id) in allowed
    }
    for neighbor in sorted(neighbors, key=lambda item: (str(item.document_id), item.chunk.ordinal)):
        key = evidence_key(neighbor)
        matches = [
            anchor
            for anchor in anchors
            if (anchor.document_id, anchor.chunk.generation_id) == key[:2]
            and 0 < abs(anchor.chunk.ordinal - neighbor.chunk.ordinal) <= window
        ]
        if key in merged or not matches:
            continue
        merged[key] = neighbor.model_copy(
            update={
                "retrieval": neighbor.retrieval.model_copy(
                    update={
                        "neighbor_of": [item.chunk.id for item in matches],
                        "reranker_rank": min(
                            item.retrieval.reranker_rank or rank + 1
                            for rank, item in enumerate(matches)
                        ),
                    }
                )
            }
        )
    return list(merged.values())


def near_duplicate(left: str, right: str) -> bool:
    left = " ".join(left.casefold().split())
    right = " ".join(right.casefold().split())
    if left == right:
        return True
    # Keep distinct identifiers/figures even in otherwise repeated boilerplate.
    if re.findall(r"\d+(?:[.,]\d+)*", left) != re.findall(r"\d+(?:[.,]\d+)*", right):
        return False
    # A small additional fact in a heavily repeated passage still needs to survive.
    if set(re.findall(r"\w+", left)) != set(re.findall(r"\w+", right)):
        return False

    def shingles(text: str) -> set[str]:
        words = re.findall(r"\w+", text)
        return {" ".join(words[index : index + 5]) for index in range(len(words) - 4)}

    a, b = shingles(left), shingles(right)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.92


def prioritize_evidence(
    candidates: Sequence[EvidenceRecord],
    subquestions: Sequence[str],
    required: Sequence[UUID] = (),
) -> list[EvidenceRecord]:
    """Retain independent-question coverage first, then relevance and boundary context."""
    records = list({evidence_key(item): item for item in candidates}.values())
    priority = [item for item in records if item.chunk.id in required]
    for question in subquestions:
        matches = [
            item
            for item in records
            if any(hit.subquestion == question for hit in item.retrieval.query_hits)
        ]
        if matches:
            # Final reranker order is authoritative; raw subquestion ranks break ties.
            best = min(
                matches,
                key=lambda item: (
                    item.retrieval.reranker_rank or len(records),
                    min(
                        hit.rank for hit in item.retrieval.query_hits if hit.subquestion == question
                    ),
                    sum(other.document_id == item.document_id for other in priority),
                ),
            )
            if best not in priority:
                priority.append(best)
    original = [item for item in records if not item.retrieval.neighbor_of]
    for anchor in original:
        priority.append(anchor)
        adjacent = [item for item in records if anchor.chunk.id in item.retrieval.neighbor_of]
        priority.extend(sorted(adjacent, key=lambda item: item.chunk.ordinal))
    return list({evidence_key(item): item for item in priority}.values())


def order_source_groups(evidence: Sequence[EvidenceRecord]) -> list[EvidenceRecord]:
    """Present selected boundary context in document order after relevance/budget decisions."""
    pairs = {
        (item.document_id, item.chunk.generation_id)
        for item in evidence
        if item.retrieval.neighbor_of
    }
    ordered: list[EvidenceRecord] = []
    handled: set[tuple[UUID, UUID]] = set()
    for item in evidence:
        pair = (item.document_id, item.chunk.generation_id)
        if pair not in pairs:
            ordered.append(item)
        elif pair not in handled:
            ordered.extend(
                sorted(
                    (
                        other
                        for other in evidence
                        if (other.document_id, other.chunk.generation_id) == pair
                    ),
                    key=lambda other: other.chunk.ordinal,
                )
            )
            handled.add(pair)
    return ordered
