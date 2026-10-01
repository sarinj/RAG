from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_app.evals.ragas import RagasCase, make_case, score_case, selected_ids
from rag_app.models import SourceGeneration


def test_ragas_uses_selected_original_sources_and_resolves_follow_up_question() -> None:
    scope = [SourceGeneration(document_id=uuid4(), generation_id=uuid4())]
    case = {
        "caseId": "follow-up",
        "question": "What about backups?",
        "answer": "Nightly. [S1]",
        "answerStatus": "answered",
        "expectedAnswer": "Nightly.",
        "queryDiagnostics": {"standaloneQuery": "What is the database backup schedule?"},
        "retrievalStages": {"selected": [{"chunkId": "chosen"}]},
        "retrieval": {"retrievedChunkIds": ["unselected", "chosen"]},
    }
    row = make_case(case, scope, {"chosen": "Backups run nightly.", "unselected": "Other text."})
    assert row.retrieved_contexts == ["Backups run nightly."]
    assert row.user_input == "What is the database backup schedule?"
    assert row.response == "Nightly." and row.source_scope == scope
    with pytest.raises(ValueError, match="pinned source generation"):
        make_case(case, scope, {"unselected": "Other text."})


def test_ragas_never_substitutes_reranking_for_selected_evidence() -> None:
    with pytest.raises(ValueError, match="selected evidence"):
        selected_ids({"retrieval": {"rerankedChunkIds": ["not-selected"]}}, None)
    assert selected_ids({}, {"selected_evidence": [{"chunk_id": "selected"}]}) == ["selected"]


def test_ragas_rejects_trace_evidence_outside_authorized_generation() -> None:
    scope = [SourceGeneration(document_id=uuid4(), generation_id=uuid4())]
    with pytest.raises(ValueError, match="outside"):
        make_case(
            {},
            scope,
            {"chunk": "source"},
            {
                "selected_evidence": [
                    {
                        "chunk_id": "chunk",
                        "document_id": str(scope[0].document_id),
                        "generation_id": str(uuid4()),
                    }
                ]
            },
        )


class FakeMetric:
    def __init__(self, value=1.0):
        self.value = value
        self.calls = []

    async def ascore(self, **data):
        self.calls.append(data)
        return SimpleNamespace(value=self.value)


async def test_ragas_separates_false_abstentions_from_faithfulness_and_context_recall() -> None:
    case = RagasCase(
        case_id="missing",
        question="Period?",
        user_input="Period?",
        response="No.",
        reference="30 days",
        answer_status="insufficient_evidence",
        retrieved_contexts=["30 days"],
        selected_chunk_ids=["gold"],
    )
    metrics = {name: FakeMetric() for name in ["faithfulness", "answer_accuracy", "context_recall"]}
    row = await score_case(case, metrics)
    assert row["scores"] == {"faithfulness": None, "answer_accuracy": 0.0, "context_recall": 1.0}
    assert not metrics["faithfulness"].calls and not metrics["answer_accuracy"].calls
    assert metrics["context_recall"].calls[0]["reference"] == "30 days"
    unknown = await score_case(case.model_copy(update={"reference": None}), metrics)
    assert unknown["correct_abstention"] is True
    assert all(value is None for value in unknown["scores"].values())


async def test_ragas_invalid_metric_is_reported_and_question_aware_accuracy_receives_question() -> (
    None
):
    case = RagasCase(
        case_id="normal",
        question="Period?",
        user_input="Period?",
        response="30 days.",
        reference="30 days",
        answer_status="answered",
        retrieved_contexts=["30 days"],
        selected_chunk_ids=["gold"],
    )
    metric = FakeMetric(float("nan"))
    row = await score_case(case, {"answer_accuracy": metric})
    assert metric.calls[0] == {
        "user_input": "Period?",
        "response": "30 days.",
        "reference": "30 days",
    }
    assert row["scores"]["answer_accuracy"] is None
    assert row["errors"] == {"answer_accuracy": "ValueError"}
