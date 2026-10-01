from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from rag_app.config import Settings
from rag_app.models import HistoryMessage
from rag_app.retrieval.tracing import text_hash

CUTOFFS = (5, 10, 20, 50)
STAGES = ("vector", "lexical", "hybrid", "reranked", "selected")
FailureCategory = Literal[
    "wrong_scope",
    "wrong_route",
    "query_rewrite",
    "chunking",
    "retrieval",
    "lexical_retrieval",
    "vector_retrieval",
    "fusion",
    "reranker",
    "evidence_selection",
    "insufficient_evidence_detection",
    "generation",
    "citation_validation",
    "unknown",
]


class RetrievalCase(BaseModel):
    """A manual failure case can also serve as a gold retrieval evaluation case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=8000)
    expected_answer: str | None = None
    expected_chunk_ids: list[str] = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    generation_ids: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    history: list[HistoryMessage] = Field(
        default_factory=lambda: list[HistoryMessage](), max_length=20
    )
    request_id: str | None = None
    expected_route: str | None = None
    actual_vector_chunk_ids: list[str] | None = None
    actual_lexical_chunk_ids: list[str] | None = None
    actual_retrieved_chunk_ids: list[str] | None = None
    actual_reranked_chunk_ids: list[str] | None = None
    actual_selected_chunk_ids: list[str] | None = None
    failure_category: FailureCategory = "unknown"
    notes: str | None = None


def load_cases(path: Path) -> list[RetrievalCase]:
    text = path.read_text(encoding="utf-8")
    raw: Any = (
        json.loads(text)
        if path.suffix == ".json"
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if not isinstance(raw, list) or not raw:
        raise ValueError("Retrieval cases must be a nonempty JSON array or JSONL file")
    return [RetrievalCase.model_validate(item) for item in cast(list[object], raw)]


def ranking_metrics(expected: Sequence[str], ranked: Sequence[str]) -> dict[str, float]:
    gold = set(expected)
    if not gold:
        raise ValueError("Retrieval metrics require at least one gold chunk")
    ranked = list(dict.fromkeys(ranked))
    result: dict[str, float] = {}
    for cutoff in CUTOFFS:
        relevance = [int(item in gold) for item in ranked[:cutoff]]
        result[f"recall@{cutoff}"] = sum(relevance) / len(gold)
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(relevance, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), cutoff) + 1))
        result[f"ndcg@{cutoff}"] = dcg / ideal
    result["mrr"] = next((1 / rank for rank, item in enumerate(ranked, 1) if item in gold), 0.0)
    result["source_coverage"] = len(gold.intersection(ranked)) / len(gold)
    return result


def failure_hint(expected: Sequence[str], rankings: dict[str, list[str]]) -> str:
    gold = set(expected)
    if "selected" in rankings and gold <= set(rankings["selected"]):
        return "unknown"
    raw = set(rankings.get("vector", [])) | set(rankings.get("lexical", []))
    if ("vector" in rankings or "lexical" in rankings) and not gold <= raw:
        return "retrieval"
    for stage, category in (
        ("hybrid", "fusion"),
        ("reranked", "reranker"),
        ("selected", "evidence_selection"),
    ):
        if stage in rankings and not gold <= set(rankings[stage]):
            return category
    return "unknown"


def replay_rankings(case: RetrievalCase) -> dict[str, list[str]]:
    mapping = {
        "vector": case.actual_vector_chunk_ids,
        "lexical": case.actual_lexical_chunk_ids,
        "hybrid": case.actual_retrieved_chunk_ids,
        "reranked": case.actual_reranked_chunk_ids,
        "selected": case.actual_selected_chunk_ids,
    }
    return {stage: items for stage, items in mapping.items() if items is not None}


def trace_rankings(trace: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for stage, field in (
        ("vector", "vector_candidates"),
        ("lexical", "lexical_candidates"),
        ("hybrid", "fused_candidates"),
        ("reranked", "reranked_candidates"),
        ("selected", "selected_evidence"),
    ):
        records: list[dict[str, Any]] = trace.get(field, [])
        if stage in {"vector", "lexical"}:
            # Same channel aggregation as the live endpoint: RRF across per-query rankings.
            groups: dict[str, list[dict[str, Any]]] = {}
            for item in records:
                hits: list[dict[str, Any]] = item.get("query_hits") or []
                hit: dict[str, Any] = hits[-1] if hits else {}
                key = str(hit.get("query_hash") or hit.get("query") or "single_query")
                groups.setdefault(key, []).append(item)
            scores: dict[str, float] = {}
            identities: dict[str, str] = {}
            for ranking in groups.values():
                seen: set[str] = set()
                for index, item in enumerate(ranking, 1):
                    chunk = item["chunk_id"]
                    if chunk in seen:
                        continue
                    seen.add(chunk)
                    scores[chunk] = scores.get(chunk, 0.0) + 1 / (60 + index)
                    identities[chunk] = str(
                        (item.get("document_id"), item.get("generation_id"), chunk)
                    )
            result[stage] = sorted(scores, key=lambda chunk: (-scores[chunk], identities[chunk]))
        else:
            result[stage] = list(dict.fromkeys(item["chunk_id"] for item in records))
    return result


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(results),
        "failed_cases": sum(bool(item.get("error")) for item in results),
        "stages": {},
    }
    for stage in STAGES:
        metrics = [item["metrics"][stage] for item in results if stage in item["metrics"]]
        summary["stages"][stage] = (
            {
                "measured_cases": len(metrics),
                **{
                    name: round(sum(item[name] for item in metrics) / len(metrics), 6)
                    for name in metrics[0]
                    if metrics
                },
            }
            if metrics
            else {"measured_cases": 0}
        )
    return summary


async def run_retrieval_evaluation(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.retrieval_cases))
    if args.limit:
        cases = cases[: args.limit]
    traces: list[dict[str, Any]] = []
    if args.trace_file:
        for line in Path(args.trace_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item: Any = json.loads(line)
                if item.get("event") == "rag_trace":
                    traces.append(item)
    offline = args.offline or bool(args.trace_file)
    settings = None if offline else Settings()
    if settings is not None and not settings.internal_token:
        raise ValueError("INTERNAL_API_TOKEN is required for live retrieval evaluation")
    results: list[dict[str, Any]] = []
    failure_cases: list[RetrievalCase] = []
    async with (
        httpx.AsyncClient(
            base_url=args.backend_url.rstrip("/"), timeout=args.timeout_seconds
        ) as backend,
        httpx.AsyncClient(base_url=args.llm_url.rstrip("/"), timeout=args.timeout_seconds) as llm,
    ):
        for index, case in enumerate(cases, 1):
            error: str | None = None
            diagnostics: dict[str, Any] = {}
            rankings: dict[str, list[str]] = {}
            try:
                if args.trace_file:
                    matching = [
                        trace
                        for trace in traces
                        if (
                            trace.get("request_id") == case.request_id
                            if case.request_id
                            else trace.get("question_hash") == text_hash(case.question)
                            or trace.get("question") == case.question
                        )
                    ]
                    if not matching:
                        raise ValueError("No matching request trace")
                    trace = matching[-1]
                    rankings = trace_rankings(trace)
                    diagnostics = {
                        key: trace.get(key)
                        for key in ("planner", "timings_ms", "configuration", "error_type")
                    }
                    if trace.get("error_type"):
                        error = str(trace["error_type"])
                elif offline:
                    rankings = replay_rankings(case)
                    if not rankings:
                        raise ValueError("Offline cases need actual stage rankings or --trace-file")
                else:
                    if settings is None:
                        raise RuntimeError("Live evaluation requires runtime settings")
                    sources: list[dict[str, str]] = []
                    if not case.document_ids:
                        raise ValueError("Live cases require document_ids resolved through NestJS")
                    for document_id in case.document_ids:
                        response = await backend.get(f"/api/documents/{document_id}")
                        response.raise_for_status()
                        document = response.json()
                        generation = document.get("activeGenerationId")
                        if document.get("status") != "ready" or not generation:
                            raise ValueError(
                                "Evaluation source must have a ready active generation"
                            )
                        if case.generation_ids.get(document_id, generation) != generation:
                            raise ValueError("Gold case refers to a different indexed generation")
                        sources.append({"documentId": document_id, "generationId": generation})
                    response = await llm.post(
                        "/internal/evaluation/retrieval",
                        headers={
                            "Authorization": f"Bearer {settings.internal_token}",
                        },
                        json={
                            "contractVersion": "1.1.0",
                            "requestId": case.request_id or str(uuid4()),
                            "question": case.question,
                            "history": [
                                message.model_dump(by_alias=True) for message in case.history
                            ],
                            "sources": sources,
                            "retrievalMode": args.retrieval_mode,
                        },
                    )
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                    rankings = {
                        stage: [item["chunkId"] for item in data["stages"][stage]]
                        for stage in STAGES
                    }
                    diagnostics = data["diagnostics"]
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                error = str(exc)
            metrics = {
                stage: ranking_metrics(case.expected_chunk_ids, ranked)
                for stage, ranked in rankings.items()
            }
            hint = failure_hint(case.expected_chunk_ids, rankings) if not error else "unknown"
            planner: dict[str, Any] = diagnostics.get("planner") or {}
            route = diagnostics.get("route") or planner.get("route")
            if case.expected_route and route and case.expected_route != route:
                hint = "wrong_route"
            results.append(
                {
                    "question": case.question,
                    "request_id": case.request_id,
                    "expected_chunk_ids": case.expected_chunk_ids,
                    "rankings": rankings,
                    "metrics": metrics,
                    "failure_hint": hint,
                    "failure_category": case.failure_category,
                    "diagnostics": diagnostics,
                    "error": error,
                }
            )
            if error or ("selected" in metrics and metrics["selected"]["source_coverage"] < 1):
                failure_cases.append(
                    case.model_copy(
                        update={
                            "actual_vector_chunk_ids": rankings.get("vector"),
                            "actual_lexical_chunk_ids": rankings.get("lexical"),
                            "actual_retrieved_chunk_ids": rankings.get("hybrid"),
                            "actual_reranked_chunk_ids": rankings.get("reranked"),
                            "actual_selected_chunk_ids": rankings.get("selected"),
                        }
                    )
                )
            print(f"[retrieval {index}/{len(cases)}] {error or hint}", flush=True)
    report: dict[str, Any] = {
        "dataset": str(args.retrieval_cases),
        "mode": "offline" if offline else "live",
        "metrics": summarize_results(results),
        "cases": results,
    }
    directory = Path(args.output_dir or "evals/reports")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ("retrieval-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + ".json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage_summaries: dict[str, dict[str, Any]] = report["metrics"]["stages"]
    for stage, metrics in stage_summaries.items():
        print(stage + ": " + ", ".join(f"{key}={value}" for key, value in metrics.items()))
    print(f"Report: {target}")
    if args.failure_output:
        Path(args.failure_output).write_text(
            "".join(case.model_dump_json() + "\n" for case in failure_cases), encoding="utf-8"
        )
    return 1 if any(item.get("error") for item in results) else 0
