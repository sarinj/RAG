"""Optional RAGAS evaluation of frozen answers and the exact selected original chunks."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_app.config import Settings, load_index_configuration
from rag_app.models import SourceGeneration
from rag_app.repositories.database import Database
from rag_app.repositories.retrieval import RetrievalRepository
from rag_app.retrieval.tracing import text_hash


class RagasCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    question: str
    user_input: str
    response: str
    reference: str | None
    answer_status: str | None
    retrieved_contexts: list[str]
    selected_chunk_ids: list[str]
    source_scope: list[SourceGeneration] = Field(default_factory=lambda: list[SourceGeneration]())


def selected_ids(case: dict[str, Any], trace: dict[str, Any] | None) -> list[str]:
    """Never substitute the larger retrieved/reranked set for actual answer evidence."""
    stages: dict[str, Any] = case.get("retrievalStages") or {}
    if "selected" in stages:
        return list(dict.fromkeys(item["chunkId"] for item in stages["selected"]))
    if trace is not None:
        return list(dict.fromkeys(item["chunk_id"] for item in trace["selected_evidence"]))
    raise ValueError("Report lacks selected evidence; provide its --ragas-trace-file")


def make_case(
    case: dict[str, Any],
    scope: list[SourceGeneration],
    contexts: dict[str, str],
    trace: dict[str, Any] | None = None,
) -> RagasCase:
    ids = selected_ids(case, trace)
    if set(ids) - contexts.keys():
        raise ValueError("Selected chunk is unavailable in the report's pinned source generation")
    if trace is not None:
        authorized = {(str(s.document_id), str(s.generation_id)) for s in scope}
        if any(
            (item["document_id"], item["generation_id"]) not in authorized
            for item in trace["selected_evidence"]
        ):
            raise ValueError("Trace evidence is outside the report's source scope")
    diagnostics: dict[str, Any] = case.get("queryDiagnostics") or {}
    return RagasCase(
        case_id=case["caseId"],
        question=case["question"],
        user_input=diagnostics.get("standaloneQuery") or case["question"],
        response=re.sub(r"\[S[1-9][0-9]*\]", "", case.get("answer") or "").strip(),
        reference=case.get("expectedAnswer"),
        answer_status=case.get("answerStatus"),
        retrieved_contexts=[contexts[key] for key in ids],
        selected_chunk_ids=ids,
        source_scope=scope,
    )


async def export_cases(args: argparse.Namespace, settings: Settings) -> list[RagasCase]:
    report = json.loads(Path(args.ragas_report).read_text(encoding="utf-8"))
    traces: list[dict[str, Any]] = []
    if args.ragas_trace_file:
        traces = [
            json.loads(line)
            for line in Path(args.ragas_trace_file).read_text().splitlines()
            if line.strip()
        ]
    config = load_index_configuration(settings.index_config_path)
    database = Database(settings)
    await database.open()
    rows: list[RagasCase] = []
    try:
        repository = RetrievalRepository(database)
        for dataset in report["datasets"]:
            source = dataset["sourceDocument"]
            scope = [
                SourceGeneration(document_id=source["id"], generation_id=source["generationId"])
            ]
            evidence = await repository.retrieve_all(scope, config)
            contexts = {str(item.chunk.id): item.chunk.original_text for item in evidence}
            for case in dataset["cases"]:
                if case.get("error"):
                    continue
                matches = [
                    trace
                    for trace in traces
                    if trace.get("event") == "rag_trace"
                    and (
                        trace.get("request_id") == case.get("requestId")
                        if case.get("requestId")
                        else trace.get("question_hash") == text_hash(case["question"])
                    )
                ]
                if len(matches) > 1:
                    raise ValueError("Multiple matching traces; use a file from exactly one run")
                rows.append(make_case(case, scope, contexts, matches[0] if matches else None))
                if args.limit is not None and len(rows) >= args.limit:
                    return rows
    finally:
        await database.close()
    return rows


async def score_case(case: RagasCase, scorers: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name, scorer in scorers.items():
        if case.reference is None:
            scores[name] = None  # Abstention evaluated separately, never a perfect empty answer.
            continue
        if name in {
            "faithfulness",
            "factual_correctness",
            "answer_accuracy",
            "answer_relevancy",
        } and (case.answer_status != "answered"):
            scores[name] = None if name == "faithfulness" else 0.0
            continue
        data: dict[str, Any] = {}
        if name != "factual_correctness":
            data["user_input"] = case.user_input
        if name in {"faithfulness", "factual_correctness", "answer_accuracy", "answer_relevancy"}:
            data["response"] = case.response
        if name in {
            "factual_correctness",
            "answer_accuracy",
            "context_precision",
            "context_recall",
        }:
            data["reference"] = case.reference
        if name in {"faithfulness", "context_precision", "context_recall"}:
            data["retrieved_contexts"] = case.retrieved_contexts
        try:
            result = await scorer.ascore(**data)
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError("Non-finite metric value")
            scores[name] = value
        except Exception as exc:
            # Exceptions can contain provider credentials or full prompts.
            errors[name] = type(exc).__name__
            scores[name] = None
    return {
        "case_id": case.case_id,
        "question": case.question,
        "answer_status": case.answer_status,
        "scores": scores,
        "errors": errors,
        "correct_abstention": (
            case.answer_status == "insufficient_evidence" if case.reference is None else None
        ),
    }


async def run_ragas_evaluation(args: argparse.Namespace) -> int:
    # Optional imports stay outside production startup and ordinary retrieval evaluation.
    try:
        openai = import_module("openai")
        llms = import_module("ragas.llms")
        metrics = import_module("ragas.metrics.collections")
        embeddings = import_module("ragas.embeddings.base")
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies with uv sync --group eval") from exc
    settings = Settings()
    connection = settings.generation_connection
    if connection.provider != "openai" or not connection.api_key:
        raise ValueError("RAGAS currently requires a configured OpenAI judge connection")
    model = args.ragas_judge_model or settings.generation_model
    if not model:
        raise ValueError("Configure GENERATION_MODEL or --ragas-judge-model")
    cases = (
        [
            RagasCase.model_validate_json(line)
            for line in Path(args.ragas_inputs).read_text().splitlines()
            if line.strip()
        ]
        if args.ragas_inputs
        else await export_cases(args, settings)
    )
    cases = cases[: args.limit] if args.limit else cases
    if not cases:
        raise ValueError("No evaluable cases")
    output_dir = Path(args.output_dir or Path(__file__).resolve().parents[3] / "evals/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = output_dir / f"ragas-{stamp}"
    prefix.with_name(prefix.name + "-inputs.jsonl").write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    client = openai.AsyncOpenAI(
        api_key=connection.api_key,
        base_url=connection.base_url,
        timeout=args.timeout_seconds,
        max_retries=1,
    )
    embedding_client: Any = None
    try:
        kwargs: dict[str, Any] = {"max_tokens": args.ragas_max_tokens, "max_retries": 1}
        if model.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "low"
        llm = llms.llm_factory(model, client=client, **kwargs)
        types = {
            "faithfulness": metrics.Faithfulness,
            "factual_correctness": metrics.FactualCorrectness,
            "answer_accuracy": metrics.AnswerAccuracy,
            "context_precision": metrics.ContextPrecision,
            "context_recall": metrics.ContextRecall,
        }
        scorers = {name: types[name](llm=llm) for name in args.ragas_metrics if name in types}
        if "answer_relevancy" in args.ragas_metrics:
            embedding_connection = settings.embedding_connection
            if embedding_connection.provider != "openai" or not embedding_connection.api_key:
                raise ValueError("Answer relevancy requires configured OpenAI embeddings")
            embedding_client = openai.AsyncOpenAI(
                api_key=embedding_connection.api_key,
                base_url=embedding_connection.base_url,
                timeout=args.timeout_seconds,
                max_retries=1,
            )
            embedder = embeddings.embedding_factory(
                "openai", model=settings.embedding_model, client=embedding_client
            )
            scorers["answer_relevancy"] = metrics.AnswerRelevancy(llm=llm, embeddings=embedder)
        results: list[dict[str, Any]] = []
        gate = asyncio.Semaphore(args.ragas_concurrency)

        async def evaluate(case: RagasCase) -> None:
            async with gate:
                result = await score_case(case, scorers)
            results.append(result)
            # Checkpoint results even if the evaluator process is interrupted.
            prefix.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "ragas_version": version("ragas"),
                        "judge_model": model,
                        "judge_reasoning_effort": kwargs.get("reasoning_effort"),
                        "judge_max_tokens": args.ragas_max_tokens,
                        "source_report": args.ragas_report,
                        "metric_means": {
                            name: {
                                "measured_cases": len(values),
                                "mean": sum(values) / len(values) if values else None,
                            }
                            for name in scorers
                            for values in [
                                [
                                    row["scores"][name]
                                    for row in results
                                    if row["scores"][name] is not None
                                ]
                            ]
                        },
                        "metric_errors": sum(len(row["errors"]) for row in results),
                        "limitations": [
                            "LLM judge scores are estimates, not proof of correctness.",
                            "Unanswerable cases use abstention outcomes.",
                            "Faithfulness averages answered cases; false abstentions are separate.",
                            "Follow-ups use the recorded standalone query.",
                        ],
                        "cases": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[RAGAS {len(results)}/{len(cases)}] {case.case_id}: {result['scores']}",
                flush=True,
            )

        await asyncio.gather(*(evaluate(case) for case in cases))
    finally:
        await client.close()
        if embedding_client is not None:
            await embedding_client.close()
    print(f"RAGAS report: {prefix.with_suffix('.json')}")
    return int(any(row["errors"] for row in results))
