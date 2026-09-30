from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select

from rag_app.config import Settings, load_index_configuration
from rag_app.evals.datasets import EvaluationCase, EvaluationDataset, load_datasets
from rag_app.repositories.database import Database
from rag_app.repositories.orm import Document

DEFAULT_BACKEND_URL = "http://backend:3001"
DEFAULT_LLM_URL = "http://llm-backend:8000"
DEFAULT_TIMEOUT_SECONDS = 150.0
REPORT_DIRECTORY = Path(__file__).resolve().parents[3] / "evals" / "reports"


async def run_evaluation(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.internal_token:
        raise RuntimeError("INTERNAL_API_TOKEN must be configured to run RAG evaluation")
    if not settings.generation_model:
        raise RuntimeError("GENERATION_MODEL must be configured to run RAG evaluation")
    if not settings.database_dsn:
        raise RuntimeError("DATABASE_URL must be configured to verify evaluation source content")

    index_config = load_index_configuration(settings.index_config_path)
    retrieval_mode = index_config.retrieval_mode

    datasets = load_datasets()
    selected = list(datasets.values()) if args.dataset == "all" else [datasets[args.dataset]]
    overrides = _parse_source_overrides(args.source)
    unknown_overrides = set(overrides) - {dataset.slug for dataset in selected}
    if unknown_overrides:
        names = ", ".join(sorted(unknown_overrides))
        raise ValueError(f"Source overrides name datasets not selected: {names}")

    backend_url = args.backend_url.rstrip("/")
    llm_url = args.llm_url.rstrip("/")
    output_directory = Path(args.output_dir) if args.output_dir else REPORT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    request_timeout = httpx.Timeout(args.timeout_seconds, connect=10.0)
    headers = {"Authorization": f"Bearer {settings.internal_token}"}
    started_at = datetime.now(UTC)
    started = monotonic()
    database = Database(settings)
    await database.open()

    try:
        async with (
            httpx.AsyncClient(base_url=backend_url, timeout=30.0) as backend,
            httpx.AsyncClient(base_url=llm_url, timeout=request_timeout) as llm,
        ):
            await _require_ready(llm)
            dataset_reports: list[dict[str, Any]] = []
            cleanup_failures = 0
            for dataset in selected:
                document, created_by_run, ingestion_ms = await _prepare_document(
                    backend,
                    database,
                    dataset,
                    overrides.get(dataset.slug),
                    ingestion_timeout_seconds=args.ingestion_timeout_seconds,
                )
                cleanup_error: str | None = None
                try:
                    await _require_fixture_content(database, document, dataset)
                    case_results = await _evaluate_dataset(
                        llm,
                        headers,
                        document,
                        dataset,
                        retrieval_mode=retrieval_mode,
                        limit=args.limit,
                    )
                finally:
                    if created_by_run and not args.keep_documents:
                        try:
                            await _delete_document(backend, document["id"])
                        except (httpx.HTTPError, RuntimeError) as exc:
                            cleanup_failures += 1
                            cleanup_error = type(exc).__name__
                            print(
                                f"Warning: could not delete temporary evaluation document "
                                f"{document['id']}: {cleanup_error}",
                                file=sys.stderr,
                            )
                dataset_reports.append(
                    {
                        "dataset": {
                            "slug": dataset.slug,
                            "name": dataset.name,
                            "version": dataset.version,
                            "sourceSha256": dataset.source_sha256,
                            "corpusSha256": dataset.corpus_sha256,
                            "corpusBytes": dataset.corpus_bytes,
                            "caseCount": len(case_results),
                        },
                        "sourceDocument": {
                            "id": document["id"],
                            "name": document["name"],
                            "sizeBytes": document["sizeBytes"],
                            "generationId": document["activeGenerationId"],
                            "fixtureCorpusBytes": dataset.corpus_bytes,
                            "corpusSizeMatchesFixture": (
                                document["sizeBytes"] == dataset.corpus_bytes
                            ),
                            "corpusHashMatchesFixture": True,
                            "ingestionMs": ingestion_ms,
                            "temporary": created_by_run,
                            "createdDuringRun": created_by_run,
                            "deleteRequested": (
                                created_by_run and not args.keep_documents and cleanup_error is None
                            ),
                            "cleanupError": cleanup_error,
                        },
                        "metrics": _summarize(case_results),
                        "categoryMetrics": {
                            category: _summarize(
                                [case for case in case_results if case["category"] == category]
                            )
                            for category in sorted({case["category"] for case in case_results})
                        },
                        "cases": case_results,
                    }
                )
    finally:
        await database.close()

    index_identifiers: dict[str, Any] = {
        "embeddingModel": index_config.embedding_model,
        "embeddingDigest": index_config.embedding_digest,
        "embeddingDimension": index_config.embedding_dimension,
        "chunkerVersion": index_config.chunker_version,
        "tokenizerVersion": index_config.tokenizer_version,
    }

    report = {
        "reportVersion": 1,
        "startedAt": started_at.isoformat(),
        "durationSeconds": round(monotonic() - started, 2),
        "configuration": {
            "provider": settings.llm_provider,
            "embeddingProvider": settings.active_embedding_provider,
            "generationModel": settings.generation_model,
            "retrievalMode": retrieval_mode,
            **index_identifiers,
        },
        "metrics": _summarize(
            [case for dataset in dataset_reports for case in dataset["cases"]]
        ),
        "cleanupFailures": cleanup_failures,
        "datasets": dataset_reports,
        "metricNotes": [
            (
                "Answer quality uses lexical token F1 against the supplied gold answer; "
                "it does not establish semantic correctness."
            ),
            (
                "Retrieval Hit@5/10 and Recall@5/10 use manually labeled evidence spans "
                "for a subset of answerable cases; they measure candidate overlap, not answer grounding."
            ),
            (
                "The chat API validates that citations resolve to retrieved evidence; "
                "valid citations alone do not prove factual grounding."
            ),
        ],
    }
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    report_json = output_directory / f"rag-eval-{stamp}.json"
    report_markdown = output_directory / f"rag-eval-{stamp}.md"
    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_markdown.write_text(_render_markdown(report), encoding="utf-8")

    print(_render_console_summary(report))
    print(f"JSON report: {report_json}")
    print(f"Markdown report: {report_markdown}")
    return 1 if report["metrics"]["failedCases"] or report["cleanupFailures"] else 0


async def _require_ready(llm: httpx.AsyncClient) -> None:
    deadline = monotonic() + 60.0
    while monotonic() < deadline:
        try:
            response = await llm.get("/health/ready", timeout=5.0)
            if response.status_code == 200 and response.json().get("status") == "ready":
                return
        except (httpx.HTTPError, ValueError):
            pass
        await asyncio.sleep(2.0)
    raise RuntimeError(
        "The LLM backend did not become ready within 60 seconds; "
        "check its database and model configuration"
    )


async def _prepare_document(
    backend: httpx.AsyncClient,
    database: Database,
    dataset: EvaluationDataset,
    document_id_override: str | None,
    *,
    ingestion_timeout_seconds: float,
) -> tuple[dict[str, Any], bool, float | None]:
    if document_id_override:
        document = await _get_document(backend, document_id_override)
        return _require_active_document(document), False, None

    existing_document_id = await _find_document_by_hash(database, dataset.corpus_sha256)
    if existing_document_id is not None:
        document = await _wait_for_document(
            backend,
            existing_document_id,
            timeout_seconds=ingestion_timeout_seconds,
        )
        return _require_active_document(document), False, None

    upload_name = f"rag-eval-{dataset.slug}-{dataset.corpus_sha256[:12]}.txt"
    ingestion_started = monotonic()
    response = await backend.post(
        "/api/documents",
        files={"file": (upload_name, dataset.corpus_text.encode("utf-8"), "text/plain")},
    )
    if response.status_code == 409:
        existing_document_id = await _find_document_by_hash(database, dataset.corpus_sha256)
        if existing_document_id is not None:
            document = await _wait_for_document(
                backend,
                existing_document_id,
                timeout_seconds=ingestion_timeout_seconds,
            )
            return _require_active_document(document), False, None
    if response.status_code != 202:
        try:
            error_body = response.json()
        except ValueError:
            error_body = {}
        code = error_body.get("code", "upload_failed")
        raise RuntimeError(
            f"Could not upload {dataset.slug} evaluation corpus: HTTP "
            f"{response.status_code} ({code})"
        )
    document_id = response.json()["id"]
    try:
        document = await _wait_for_document(
            backend,
            document_id,
            timeout_seconds=ingestion_timeout_seconds,
        )
        document = _require_active_document(document)
    except (httpx.HTTPError, RuntimeError):
        await _delete_document(backend, document_id)
        raise
    return document, True, round((monotonic() - ingestion_started) * 1000, 2)


async def _find_document_by_hash(database: Database, expected_hash: str) -> str | None:
    statement = select(Document.id).where(
        Document.content_hash == expected_hash,
        Document.deleted_at.is_(None),
    )
    async with database.session() as session:
        document_id = await session.scalar(statement)
    return str(document_id) if document_id is not None else None


async def _require_fixture_content(
    database: Database,
    document: dict[str, Any],
    dataset: EvaluationDataset,
) -> None:
    statement = select(Document.content_hash).where(
        Document.id == UUID(document["id"]),
        Document.deleted_at.is_(None),
    )
    async with database.session() as session:
        content_hash = await session.scalar(statement)
    if content_hash != dataset.corpus_sha256:
        raise RuntimeError(
            f"Evaluation source document {document['id']} does not match "
            f"the {dataset.slug} fixture corpus"
        )


async def _get_document(backend: httpx.AsyncClient, document_id: str) -> dict[str, Any]:
    try:
        UUID(document_id)
    except ValueError as exc:
        raise ValueError(f"Invalid evaluation document UUID: {document_id}") from exc
    response = await backend.get(f"/api/documents/{document_id}")
    response.raise_for_status()
    return response.json()


async def _wait_for_document(
    backend: httpx.AsyncClient,
    document_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    while True:
        document = await _get_document(backend, document_id)
        if document.get("status") == "ready" and document.get("activeGenerationId"):
            return document
        if document.get("status") == "failed":
            job = document.get("latestJob") or {}
            error = job.get("error") or {}
            raise RuntimeError(
                f"Evaluation corpus ingestion failed ({error.get('code', 'unknown error')})"
            )
        if monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for evaluation corpus {document_id} to be ready")
        await asyncio.sleep(2.0)


def _require_active_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("status") != "ready" or not document.get("activeGenerationId"):
        raise RuntimeError(
            f"Evaluation document {document.get('name', document.get('id'))} must be ready "
            "with an active generation"
        )
    return document


async def _delete_document(backend: httpx.AsyncClient, document_id: str) -> None:
    response = await backend.delete(f"/api/documents/{document_id}")
    if response.status_code not in {200, 204}:
        raise RuntimeError(f"Document deletion returned HTTP {response.status_code}")


async def _evaluate_dataset(
    llm: httpx.AsyncClient,
    headers: dict[str, str],
    document: dict[str, Any],
    dataset: EvaluationDataset,
    *,
    retrieval_mode: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    cases = dataset.cases[:limit] if limit else dataset.cases
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        result = await _evaluate_case(
            llm,
            headers,
            document,
            case,
            retrieval_mode=retrieval_mode,
        )
        result["dataset"] = dataset.slug
        results.append(result)
        print(
            f"[{dataset.slug} {index}/{len(cases)}] {case.case_id}: "
            f"{result.get('answerStatus', 'error')}",
            flush=True,
        )
    return results


async def _evaluate_case(
    llm: httpx.AsyncClient,
    headers: dict[str, str],
    document: dict[str, Any],
    case: EvaluationCase,
    *,
    retrieval_mode: str,
) -> dict[str, Any]:
    payload = {
        "contractVersion": "1.0.0",
        "requestId": str(uuid4()),
        "question": case.question,
        "history": [],
        "sources": [
            {
                "documentId": document["id"],
                "generationId": document["activeGenerationId"],
            }
        ],
        "retrievalMode": retrieval_mode,
    }
    started = monotonic()
    try:
        response = await llm.post("/internal/evaluation/chat", json=payload, headers=headers)
        elapsed_ms = round((monotonic() - started) * 1000, 2)
        if response.status_code != 200:
            try:
                error_body = response.json()
            except ValueError:
                error_body = {}
            return {
                "caseId": case.case_id,
                "category": case.category,
                "question": case.question,
                "expectedAnswer": case.expected_answer,
                "answer": None,
                "answerStatus": None,
                "answerTokenF1": 0.0 if case.expected_answer is not None else None,
                "correctAbstention": False if case.expected_answer is None else None,
                "citationCount": 0,
                "retrieval": (
                    {"error": {"type": "EvaluationRequestFailed"}}
                    if case.evidence_spans
                    else None
                ),
                "latencyMs": elapsed_ms,
                "error": {
                    "status": response.status_code,
                    "code": error_body.get("code"),
                },
            }

        evaluation_data = response.json()
        answer_data = evaluation_data["response"]
        answer = answer_data["answer"]
        answer_status = answer_data["answerStatus"]
        citations = answer_data.get("citations", [])
        is_unanswerable = case.expected_answer is None
        candidates = evaluation_data["retrievalCandidates"]
        return {
            "caseId": case.case_id,
            "category": case.category,
            "question": case.question,
            "expectedAnswer": case.expected_answer,
            "answer": answer,
            "answerStatus": answer_status,
            "answerTokenF1": (
                None
                if is_unanswerable
                else _token_f1(_remove_citations(answer), case.expected_answer or "")
            ),
            "correctAbstention": (
                answer_status == "insufficient_evidence" if is_unanswerable else None
            ),
            "citationCount": len(citations),
            "retrieval": (
                _retrieval_metrics(case, candidates) if case.evidence_spans else None
            ),
            "latencyMs": elapsed_ms,
            "error": None,
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return {
            "caseId": case.case_id,
            "category": case.category,
            "question": case.question,
            "expectedAnswer": case.expected_answer,
            "answer": None,
            "answerStatus": None,
            "answerTokenF1": 0.0 if case.expected_answer is not None else None,
            "correctAbstention": False if case.expected_answer is None else None,
            "citationCount": 0,
            "retrieval": (
                {"error": {"type": type(exc).__name__}} if case.evidence_spans else None
            ),
            "latencyMs": round((monotonic() - started) * 1000, 2),
            "error": {"type": type(exc).__name__},
        }


def _retrieval_metrics(
    case: EvaluationCase,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "annotatedEvidenceCount": len(case.evidence_spans),
        "retrievedChunkIds": [candidate["chunkId"] for candidate in candidates],
    }
    for cutoff in (5, 10):
        top_candidates = candidates[:cutoff]
        covered = [
            any(
                candidate["sourceStart"] < evidence_end
                and candidate["sourceEnd"] > evidence_start
                for candidate in top_candidates
            )
            for evidence_start, evidence_end in case.evidence_spans
        ]
        result[f"hitAt{cutoff}"] = any(covered)
        result[f"recallAt{cutoff}"] = round(sum(covered) / len(covered), 4)
    return result


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [case for case in cases if case["expectedAnswer"] is not None]
    unanswerable = [case for case in cases if case["expectedAnswer"] is None]
    f1_values = [float(case["answerTokenF1"]) for case in answerable]
    latencies = [float(case["latencyMs"]) for case in cases]
    answered = [case for case in cases if case["answerStatus"] == "answered"]
    retrieval_cases = [case["retrieval"] for case in cases if case.get("retrieval") is not None]
    successful_retrieval_cases = [case for case in retrieval_cases if case.get("error") is None]
    failed_cases = sum(
        case["error"] is not None
        or (
            case.get("retrieval") is not None
            and case["retrieval"].get("error") is not None
        )
        for case in cases
    )
    return {
        "totalCases": len(cases),
        "failedCases": failed_cases,
        "answerableCases": len(answerable),
        "unanswerableCases": len(unanswerable),
        "retrievalAnnotatedCases": len(retrieval_cases),
        "retrievalFailedCases": sum(case.get("error") is not None for case in retrieval_cases),
        "retrievalHitRateAt5": _mean_metric(successful_retrieval_cases, "hitAt5"),
        "retrievalRecallAt5": _mean_metric(successful_retrieval_cases, "recallAt5"),
        "retrievalHitRateAt10": _mean_metric(successful_retrieval_cases, "hitAt10"),
        "retrievalRecallAt10": _mean_metric(successful_retrieval_cases, "recallAt10"),
        "answerableMeanTokenF1": round(statistics.mean(f1_values), 4) if f1_values else None,
        "answerableTokenF1AtLeast0_5": (
            round(sum(value >= 0.5 for value in f1_values) / len(f1_values), 4)
            if f1_values
            else None
        ),
        "unanswerableAbstentionAccuracy": (
            round(
                sum(case["correctAbstention"] is True for case in unanswerable)
                / len(unanswerable),
                4,
            )
            if unanswerable
            else None
        ),
        "answeredWithCitationRate": (
            round(sum(case["citationCount"] > 0 for case in answered) / len(answered), 4)
            if answered
            else None
        ),
        "latencyMsMean": round(statistics.mean(latencies), 2) if latencies else None,
        "latencyMsP50": _percentile(latencies, 0.50),
        "latencyMsP95": _percentile(latencies, 0.95),
    }


def _mean_metric(cases: list[dict[str, Any]], key: str) -> float | None:
    if not cases:
        return None
    return round(statistics.mean(float(case[key]) for case in cases), 4)


def _token_f1(actual: str, expected: str) -> float:
    actual_tokens = Counter(_tokens(actual))
    expected_tokens = Counter(_tokens(expected))
    if not actual_tokens or not expected_tokens:
        return 0.0
    overlap = sum((actual_tokens & expected_tokens).values())
    precision = overlap / sum(actual_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def _remove_citations(text: str) -> str:
    return re.sub(r"\[S\d+\]", "", text)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(quantile * len(ordered)) - 1)], 2)


def _parse_source_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        slug, separator, document_id = value.partition("=")
        if not separator or not slug or not document_id:
            raise ValueError("Source overrides must use DATASET=DOCUMENT_UUID")
        if slug in overrides:
            raise ValueError(f"Duplicate source override for {slug}")
        overrides[slug] = document_id
    return overrides


def _render_console_summary(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "RAG evaluation",
        f"Cases: {metrics['totalCases']} ({metrics['failedCases']} failed)",
        f"Mean answer token F1: {metrics['answerableMeanTokenF1']}",
        f"Answerable F1 >= 0.5: {metrics['answerableTokenF1AtLeast0_5']}",
        f"Unanswerable abstention accuracy: {metrics['unanswerableAbstentionAccuracy']}",
        (
            f"Retrieval Hit@5 / Recall@5: {metrics['retrievalHitRateAt5']} / "
            f"{metrics['retrievalRecallAt5']} "
            f"({metrics['retrievalAnnotatedCases']} labeled cases)"
        ),
        f"Answered with citations: {metrics['answeredWithCitationRate']}",
        f"Latency ms p50/p95: {metrics['latencyMsP50']} / {metrics['latencyMsP95']}",
    ]
    for dataset in report["datasets"]:
        item = dataset["metrics"]
        lines.append(
            f"{dataset['dataset']['slug']}: {item['totalCases']} cases, "
            f"token F1 {item['answerableMeanTokenF1']}, "
            f"abstention {item['unanswerableAbstentionAccuracy']}"
        )
    return "\n".join(lines)


def _render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    configuration = report["configuration"]
    lines = [
        "# RAG evaluation report",
        "",
        f"Started: {report['startedAt']}",
        f"Duration: {report['durationSeconds']} seconds",
        f"Provider: `{configuration['provider']}`",
        f"Generation model: `{configuration['generationModel']}`",
        f"Retrieval mode: `{configuration['retrievalMode']}`",
        "",
        "## Aggregate metrics",
        "",
        (
            "| Cases | Failed | Answerable mean token F1 | Answerable F1 ≥ 0.5 | "
            "Abstention accuracy | Retrieval Hit@5 / Recall@5 | Citation rate | "
            "Latency p50 / p95 ms |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {metrics['totalCases']} | {metrics['failedCases']} | "
            f"{metrics['answerableMeanTokenF1']} | "
            f"{metrics['answerableTokenF1AtLeast0_5']} | "
            f"{metrics['unanswerableAbstentionAccuracy']} | "
            f"{metrics['retrievalHitRateAt5']} / {metrics['retrievalRecallAt5']} "
            f"({metrics['retrievalAnnotatedCases']} labeled) | "
            f"{metrics['answeredWithCitationRate']} | "
            f"{metrics['latencyMsP50']} / {metrics['latencyMsP95']} |"
        ),
        "",
    ]
    for dataset in report["datasets"]:
        metadata = dataset["dataset"]
        source = dataset["sourceDocument"]
        lines.extend(
            [
                f"## {metadata['name']} ({metadata['version']})",
                "",
                f"Cases: {metadata['caseCount']} · source `{source['name']}`",
                f"Generation: `{source['generationId']}`",
                (
                    f"Indexed corpus bytes: {source['sizeBytes']} "
                    f"(fixture: {source['fixtureCorpusBytes']}) · "
                    f"SHA-256 verified · ingestion: {source['ingestionMs'] or 'reused'} ms"
                ),
                f"Dataset SHA-256: `{metadata['sourceSha256']}`",
                f"Corpus SHA-256: `{metadata['corpusSha256']}`",
                "",
                "| ID | Type | Token F1 | Hit@5 | Recall@5 | Status | Latency ms | Question | Answer | Gold answer |",
                "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |",
            ]
        )
        for case in dataset["cases"]:
            lines.append(
                (
                    "| {caseId} | {category} | {f1} | {hit5} | {recall5} | {status} | {latency} | "
                    "{question} | {answer} | {expected} |"
                ).format(
                    caseId=_cell(case["caseId"]),
                    category=_cell(case["category"]),
                    f1=case["answerTokenF1"] if case["answerTokenF1"] is not None else "—",
                    hit5=(
                        case["retrieval"]["hitAt5"]
                        if case.get("retrieval") is not None
                        and case["retrieval"].get("error") is None
                        else "—"
                    ),
                    recall5=(
                        case["retrieval"]["recallAt5"]
                        if case.get("retrieval") is not None
                        and case["retrieval"].get("error") is None
                        else "—"
                    ),
                    status=_cell(case["answerStatus"] or case["error"] or "error"),
                    latency=case["latencyMs"],
                    question=_cell(case["question"]),
                    answer=_cell(case["answer"] or ""),
                    expected=_cell(case["expectedAnswer"] or "(abstain)"),
                )
            )
        lines.append("")
    lines.extend(["## Metric notes", ""])
    lines.extend(f"- {note}" for note in report["metricNotes"])
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the running RAG service on bundled datasets"
    )
    parser.add_argument("--dataset", choices=["all", "astrarail", "helios_city"], default="all")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="DATASET=DOCUMENT_UUID",
        help="Override automatic source-document matching; repeat for multiple datasets",
    )
    parser.add_argument(
        "--backend-url",
        default=None,
        help="NestJS API base URL (default: RAG_EVAL_BACKEND_URL or http://backend:3001)",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        help="LLM service base URL (default: RAG_EVAL_LLM_URL or http://llm-backend:8000)",
    )
    parser.add_argument("--output-dir", default=None, help="Report directory")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--ingestion-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases per dataset",
    )
    parser.add_argument(
        "--keep-documents",
        action="store_true",
        help="Keep the temporary evaluation corpus documents after the run",
    )
    parsed = parser.parse_args()
    parsed.backend_url = parsed.backend_url or _env("RAG_EVAL_BACKEND_URL", DEFAULT_BACKEND_URL)
    parsed.llm_url = parsed.llm_url or _env("RAG_EVAL_LLM_URL", DEFAULT_LLM_URL)
    if parsed.limit is not None and parsed.limit <= 0:
        parser.error("--limit must be greater than zero")
    if parsed.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    if parsed.ingestion_timeout_seconds <= 0:
        parser.error("--ingestion-timeout-seconds must be greater than zero")
    return parsed


def _env(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def main() -> None:
    try:
        exit_code = asyncio.run(run_evaluation(_arguments()))
    except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
        print(f"rag-eval: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
