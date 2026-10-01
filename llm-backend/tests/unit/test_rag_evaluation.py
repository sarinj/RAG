from __future__ import annotations

import argparse
import json

import httpx
import pytest

from rag_app.evals.datasets import EvaluationCase
from rag_app.evals.retrieval import (
    failure_hint,
    load_cases,
    ranking_metrics,
    run_retrieval_evaluation,
)
from rag_app.evals.runner import _evaluate_case, _summarize


def test_retrieval_metrics_measure_recall_mrr_ndcg_without_duplicate_credit() -> None:
    metrics = ranking_metrics(["a", "b"], ["x", "a", "a", "b"])
    assert metrics["recall@5"] == metrics["recall@50"] == 1
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg@5"] < 1
    assert ranking_metrics(["a"], ["a", "a"])["ndcg@10"] == 1
    assert ranking_metrics(["missing"], ["a"])["mrr"] == 0
    with pytest.raises(ValueError, match="gold chunk"):
        ranking_metrics([], [])


def test_recall_cutoffs_and_first_loss_diagnostics() -> None:
    rankings = {"vector": ["a"], "lexical": [], "hybrid": ["a"], "reranked": [], "selected": []}
    assert failure_hint(["a"], rankings) == "reranker"
    assert failure_hint(["b"], rankings) == "retrieval"
    metrics = ranking_metrics(["gold"], [f"x{index}" for index in range(19)] + ["gold"])
    assert metrics["recall@5"] == metrics["recall@10"] == 0
    assert metrics["recall@20"] == metrics["recall@50"] == 1
    assert metrics["mrr"] == 0.05


async def test_offline_failure_dataset_runs_without_services(tmp_path) -> None:
    path = tmp_path / "failures.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Trial period?",
                "expected_chunk_ids": ["c182"],
                "actual_retrieved_chunk_ids": ["c91", "c182"],
                "actual_reranked_chunk_ids": ["c91"],
                "actual_selected_chunk_ids": ["c91"],
                "failure_category": "reranker",
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        retrieval_cases=path,
        limit=None,
        trace_file=None,
        offline=True,
        backend_url="http://unavailable.invalid",
        llm_url="http://unavailable.invalid",
        timeout_seconds=1,
        output_dir=tmp_path / "reports",
        failure_output=tmp_path / "saved.jsonl",
    )
    assert await run_retrieval_evaluation(args) == 0
    report = json.loads(next((tmp_path / "reports").glob("*.json")).read_text())
    assert report["metrics"]["stages"]["hybrid"]["recall@5"] == 1
    assert report["metrics"]["stages"]["selected"]["source_coverage"] == 0
    assert report["metrics"]["stages"]["vector"]["measured_cases"] == 0
    assert load_cases(tmp_path / "saved.jsonl")[0].failure_category == "reranker"


def test_benchmark_counts_false_abstentions_separately_from_request_failures() -> None:
    cases = [
        {
            "expectedAnswer": "two years",
            "answerStatus": "insufficient_evidence",
            "answerTokenF1": 0.0,
            "latencyMs": 10,
            "citationCount": 0,
            "error": None,
            "retrieval": None,
        },
        {
            "expectedAnswer": None,
            "answerStatus": "insufficient_evidence",
            "answerTokenF1": None,
            "correctAbstention": True,
            "latencyMs": 10,
            "citationCount": 0,
            "error": None,
            "retrieval": None,
        },
    ]
    metrics = _summarize(cases)
    assert metrics["failedCases"] == 0
    assert metrics["answerableFalseAbstentions"] == 1
    assert metrics["answerableFalseAbstentionRate"] == 1
    assert metrics["answerableAnswerRate"] == 0
    assert metrics["unanswerableAbstentionAccuracy"] == 1


async def test_benchmark_abstention_does_not_get_keyword_overlap_credit() -> None:
    payload = {
        "response": {
            "answer": "I could not find relevant warranty years in the selected documents.",
            "answerStatus": "insufficient_evidence",
            "citations": [],
        },
        "retrievalCandidates": [],
        "rerankedCandidates": [],
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(base_url="http://llm.test", transport=transport) as client:
        result = await _evaluate_case(
            client,
            {},
            {"id": "doc", "activeGenerationId": "gen"},
            EvaluationCase(
                case_id="warranty",
                category="direct",
                question="Warranty period?",
                expected_answer="two years warranty",
                history=(),
                evidence_spans=(),
            ),
            retrieval_mode="hybrid",
        )
    assert result["error"] is None
    assert result["answerTokenF1"] == 0
