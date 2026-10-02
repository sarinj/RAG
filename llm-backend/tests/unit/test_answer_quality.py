from __future__ import annotations

import json

import pytest
from test_chat_follow_ups import ScriptedProvider
from test_query_routing import configuration, corpus_for, execute_with, plan_for, tokenizer_for

from rag_app.config import RagSettings
from rag_app.errors import ProviderTimeout
from rag_app.generation.answer_quality import review_answer
from rag_app.generation.grounding import AnswerClaim, GroundedAnswer
from rag_app.models import QueryRoute


def coverage_json() -> str:
    return '{"coverage":"complete","missing_aspects":[]}'


def draft_json(text: str) -> str:
    return json.dumps({"claims": [{"text": text, "source_labels": ["S1"]}]})


def review_json(quote: str | None = None, ids: tuple[int, ...] = ()) -> str:
    return json.dumps(
        {
            "missing_facts": [{"source_label": "S1", "quote": quote}] if quote else [],
            "irrelevant_claim_ids": list(ids),
        }
    )


async def test_review_validates_whitespace_normalized_original_quotes() -> None:
    text = "Procedure RESET-17 applies only after isolation.\nController approval is required."
    corpus = corpus_for([text])
    provider = ScriptedProvider([review_json(text.replace("\n", " "))])
    answer = GroundedAnswer(
        claims=[AnswerClaim(text="RESET-17.", source_chunk_ids=[corpus[0].chunk.id])]
    )
    review, status = await review_answer(
        provider,
        tokenizer_for(text),
        configuration(),
        question="Which reset procedure applies?",
        query="reset procedure",
        answer=answer,
        evidence=corpus,
    )
    assert status == "checked" and review is not None and review.needs_repair
    assert "Controller approval" in review.missing_facts[0].quote
    schema = provider.generations[0].response_schema
    assert schema is not None
    assert schema["$defs"]["SourceQuote"]["properties"]["source_label"]["enum"] == ["S1"]


async def test_review_rejects_unknown_source_labels() -> None:
    text = "The period is 30 days."
    corpus = corpus_for([text])
    provider = ScriptedProvider(
        [
            json.dumps(
                {
                    "missing_facts": [{"source_label": "S2", "quote": text}],
                    "irrelevant_claim_ids": [],
                }
            )
        ]
    )
    answer = GroundedAnswer(
        claims=[AnswerClaim(text="30 days.", source_chunk_ids=[corpus[0].chunk.id])]
    )
    review, status = await review_answer(
        provider,
        tokenizer_for(text),
        configuration(),
        question="What period?",
        query="period",
        answer=answer,
        evidence=corpus,
    )
    assert review is None and status == "invalid_source_quote"


async def test_review_is_opt_in(monkeypatch) -> None:
    text = "The desk number is 555-0109."
    corpus = corpus_for([text])
    provider = ScriptedProvider(
        [plan_for(QueryRoute.LOCAL_SEARCH), coverage_json(), draft_json(text)]
    )
    settings = RagSettings()
    assert not settings.rag_answer_review_enabled
    executor, request, _ = execute_with(monkeypatch, corpus, provider, settings=settings)
    result = await executor.answer(request)
    assert result.response.answer == f"{text} [S1]"
    assert len(provider.generations) == 3


async def test_review_context_limit_makes_no_provider_call() -> None:
    corpus = corpus_for(["A source fact. " * 300])
    provider = ScriptedProvider([])
    answer = GroundedAnswer(
        claims=[AnswerClaim(text="A source fact.", source_chunk_ids=[corpus[0].chunk.id])]
    )
    review, status = await review_answer(
        provider,
        tokenizer_for(corpus[0].chunk.original_text),
        configuration(512, 128),
        question="What fact?",
        query="fact",
        answer=answer,
        evidence=corpus,
    )
    assert review is None and status == "context_limit" and not provider.generations


async def test_missing_qualification_is_repaired_once_and_rechecked(monkeypatch) -> None:
    text = "A transfer is valid for 50 minutes. The overnight shuttle is excluded."
    corpus = corpus_for([text])
    provider = ScriptedProvider(
        [
            plan_for(QueryRoute.LOCAL_SEARCH, query="transfer validity"),
            coverage_json(),
            draft_json("A transfer is valid for 50 minutes."),
            '{"verdicts":[{"id":1,"status":"supported"}]}',
            review_json("The overnight shuttle is excluded."),
            draft_json(text),
            review_json(),
        ]
    )
    executor, request, _ = execute_with(
        monkeypatch, corpus, provider, settings=RagSettings(rag_answer_review_enabled=True)
    )
    result = await executor.answer(request.model_copy(update={"question": "Transfer validity?"}))
    assert result.response.answer == f"{text} [S1]"
    assert len(provider.generations) == 7
    assert "Evidence (original source text)" in provider.generations[2].prompt
    assert provider.generations[2].verbosity == "low"
    assert provider.generations[5].verbosity == "low"
    assert "COMPLETENESS AND RELEVANCE REPAIR" in provider.generations[5].prompt
    assert "overnight shuttle" in provider.generations[5].prompt
    assert result.diagnostics.timings_ms["answer_review"] > 0


@pytest.mark.parametrize(
    "review",
    [review_json("An invented fact."), review_json(ids=(2,)), review_json(ids=(1, 1)), "not JSON"],
)
async def test_invalid_review_cannot_discard_supported_answer(monkeypatch, review: str) -> None:
    text = "The desk number is 555-0109."
    corpus = corpus_for([text])
    provider = ScriptedProvider(
        [plan_for(QueryRoute.LOCAL_SEARCH), coverage_json(), draft_json(text), review]
    )
    executor, request, _ = execute_with(
        monkeypatch, corpus, provider, settings=RagSettings(rag_answer_review_enabled=True)
    )
    result = await executor.answer(request)
    assert result.response.answer == f"{text} [S1]"
    assert len(provider.generations) == 4


@pytest.mark.parametrize("repair", [draft_json("The period is 99 days."), '{"claims":[]}'])
async def test_failed_quality_repair_retains_the_supported_draft(monkeypatch, repair: str) -> None:
    text = "The period is 30 days. Renewal is excluded."
    corpus = corpus_for([text])
    outputs = [
        plan_for(QueryRoute.LOCAL_SEARCH),
        coverage_json(),
        draft_json("The period is 30 days."),
        '{"verdicts":[{"id":1,"status":"supported"}]}',
        review_json("Renewal is excluded."),
        repair,
    ]
    if "99" in repair:
        outputs.append('{"verdicts":[{"id":1,"status":"unsupported"}]}')
    provider = ScriptedProvider(outputs)
    executor, request, _ = execute_with(
        monkeypatch, corpus, provider, settings=RagSettings(rag_answer_review_enabled=True)
    )
    result = await executor.answer(request)
    assert result.response.answer == "The period is 30 days. [S1]"
    assert len(provider.generations) == len(outputs)


async def test_insufficient_coverage_stops_answer_generation(monkeypatch) -> None:
    corpus = corpus_for(["An unrelated fact."])
    provider = ScriptedProvider(
        [
            plan_for(QueryRoute.LOCAL_SEARCH),
            '{"coverage":"insufficient","missing_aspects":[]}',
        ]
    )
    executor, request, _ = execute_with(
        monkeypatch, corpus, provider, settings=RagSettings(rag_answer_review_enabled=True)
    )
    result = await executor.answer(request)
    assert result.response.answer_status == "insufficient_evidence"
    assert len(provider.generations) == 2


async def test_quality_model_is_used_for_answer_steps_and_recorded(monkeypatch) -> None:
    text = "The renewal period is 17 months."
    corpus = corpus_for([text])
    provider = ScriptedProvider(
        [plan_for(QueryRoute.LOCAL_SEARCH), coverage_json(), draft_json(text), review_json()]
    )
    executor, request, _ = execute_with(
        monkeypatch,
        corpus,
        provider,
        settings=RagSettings(rag_quality_model="quality-model", rag_answer_review_enabled=True),
    )
    result = await executor.answer(request)
    assert result.response.answer == f"{text} [S1]"
    assert provider.generations[0].model is None
    assert all(call.model == "quality-model" for call in provider.generations[1:])
    assert result.diagnostics.configuration["rag_quality_model"] == "quality-model"
    assert "Original source excerpts" in provider.generations[-1].prompt


async def test_writer_verifier_and_reviewer_use_original_sources(
    monkeypatch,
) -> None:
    requested = "The renewal period is 17 months."
    unrelated = "The staff cafeteria opens at 06:45."
    corpus = corpus_for([f"{requested} {unrelated}"])
    provider = ScriptedProvider(
        [
            plan_for(QueryRoute.LOCAL_SEARCH, query="renewal period"),
            coverage_json(),
            draft_json("Renewal period: 17 months."),
            '{"verdicts":[{"id":1,"status":"supported"}]}',
            review_json(),
        ]
    )
    executor, request, _ = execute_with(
        monkeypatch, corpus, provider, settings=RagSettings(rag_answer_review_enabled=True)
    )
    result = await executor.answer(request)
    assert result.response.answer == "Renewal period: 17 months. [S1]"
    assert requested in provider.generations[2].prompt
    assert unrelated in provider.generations[2].prompt
    assert unrelated in provider.generations[3].prompt
    assert unrelated in provider.generations[4].prompt


async def test_optional_review_provider_failure_preserves_a_supported_answer(monkeypatch) -> None:
    text = "The desk number is 555-0109."
    corpus = corpus_for([text])
    provider = ScriptedProvider(
        [plan_for(QueryRoute.LOCAL_SEARCH), coverage_json(), draft_json(text)]
    )

    async def unavailable(*args, **kwargs):
        raise ProviderTimeout("Temporary provider failure")

    monkeypatch.setattr("rag_app.generation.chat.review_answer", unavailable)
    executor, request, _ = execute_with(
        monkeypatch, corpus, provider, settings=RagSettings(rag_answer_review_enabled=True)
    )
    result = await executor.answer(request)
    assert result.response.answer == f"{text} [S1]"
    assert len(provider.generations) == 3
