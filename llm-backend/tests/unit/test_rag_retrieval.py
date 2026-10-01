from __future__ import annotations

import json
from uuid import uuid4

import pytest
from test_query_routing import RoutingProvider, corpus_for, execute_with, plan_for, tokenizer_for

from rag_app.config import RagSettings
from rag_app.errors import ServiceError
from rag_app.generation.intents import exhaustive_route
from rag_app.generation.prompts import ChatPromptBuilder
from rag_app.generation.query_rewriting import retrieval_queries, understand_request
from rag_app.models import QueryHit, QueryRoute, RetrievalMetadata, SourceGeneration
from rag_app.providers.models import EmbeddingOutput
from rag_app.retrieval.evidence import expand_neighbors
from rag_app.retrieval.hybrid import bm25_candidates, fuse_rankings


async def test_follow_up_query_preserves_technical_identifiers() -> None:
    provider = RoutingProvider(plan_for(QueryRoute.LOCAL_SEARCH, query="refund period for invoice"))
    plan = await understand_request(
        provider,
        tokenizer_for("INV-0382 refund"),
        question="Refund period for INV-0382?",
        history=[],
        context_window_tokens=8192,
        max_output_tokens=512,
    )
    assert "INV-0382" in plan.query
    assert "exact technical terms" in provider.generations[0].prompt


async def test_standalone_query_and_contextual_query() -> None:
    question = "What is the PostgreSQL backup policy?"
    provider = RoutingProvider(plan_for(QueryRoute.LOCAL_SEARCH, query=question))
    plan = await understand_request(
        provider,
        tokenizer_for(question),
        question=question,
        history=[],
        context_window_tokens=8192,
        max_output_tokens=512,
    )
    assert plan.query == question
    provider.plan = plan_for(QueryRoute.LOCAL_SEARCH, query=question, topic_message_index=0)
    follow_up = await understand_request(
        provider,
        tokenizer_for(question),
        question="What about its backup policy?",
        history=[
            {"role": "user", "content": "What database does the application use?"},
            {"role": "assistant", "content": "PostgreSQL"},
        ],
        context_window_tokens=8192,
        max_output_tokens=512,
    )
    assert "PostgreSQL" in follow_up.query


def test_multi_query_fusion_deduplicates_and_retains_all_channel_scores_and_ranks() -> None:
    a, b, c = corpus_for(["refund policy", "termination notice", "other policy"])
    a = a.model_copy(
        update={
            "retrieval": RetrievalMetadata(
                vector_score=0.9, query_hits=[QueryHit(query="refund", channel="vector", rank=1)]
            )
        }
    )
    lexical_a = a.model_copy(
        update={
            "retrieval": RetrievalMetadata(
                lexical_score=3.5,
                query_hits=[QueryHit(query="refund period", channel="lexical", rank=2)],
            )
        }
    )
    fused = fuse_rankings([[a, b, a], [c, lexical_a], [a, c]])
    assert len(fused) == 3
    first = next(item for item in fused if item.chunk.id == a.chunk.id)
    assert first.retrieval.vector_score == 0.9
    assert first.retrieval.lexical_score == 3.5
    assert first.retrieval.rrf_score == pytest.approx(2 / 61 + 1 / 62)
    assert len(first.retrieval.query_hits) == 2
    assert all(item.retrieval.fused_rank == index for index, item in enumerate(fused, 1))


def test_lexical_retrieval_keeps_original_text_and_scores() -> None:
    corpus = corpus_for(["refund period is thirty days", "other topic", "refund policy"])
    ranked = bm25_candidates(
        "refund period", corpus, tokenizer_for("refund period is thirty days other topic policy")
    )
    assert ranked[0].chunk.original_text == corpus[0].chunk.original_text
    assert ranked[0].retrieval.lexical_score is not None
    assert ranked[0].retrieval.lexical_rank == 1


def test_neighbors_cannot_cross_document_generation_or_scope() -> None:
    corpus = corpus_for(["previous context", "anchor fact", "next context", "far away"])
    source = SourceGeneration(
        document_id=corpus[0].document_id, generation_id=corpus[0].chunk.generation_id
    )
    foreign_doc = corpus[0].model_copy(update={"document_id": uuid4()})
    foreign_generation = corpus[0].model_copy(
        update={"chunk": corpus[0].chunk.model_copy(update={"generation_id": uuid4()})}
    )
    expanded = expand_neighbors(
        [corpus[1]], [*corpus, foreign_doc, foreign_generation], [source], window=1, top_n=1
    )
    assert {item.chunk.ordinal for item in expanded} == {0, 1, 2}
    assert len(expanded) == 3
    assert all(
        (item.document_id, item.chunk.generation_id) == (source.document_id, source.generation_id)
        for item in expanded
    )
    assert expanded[1].retrieval.neighbor_of == [corpus[1].chunk.id]


def test_evidence_selection_deduplicates_enforces_budget_and_covers_subquestions() -> None:
    corpus = corpus_for(
        ["A founder named Ada", "A founder named Ada", "Founded in 2010", "Funding was ten million"]
    )
    questions = ["founder", "funding"]
    corpus[0].retrieval.query_hits = [
        QueryHit(query="founder", subquestion="founder", channel="vector", rank=1)
    ]
    corpus[3].retrieval.query_hits = [
        QueryHit(query="funding", subquestion="funding", channel="vector", rank=1)
    ]
    tokenizer = tokenizer_for(" ".join(item.chunk.original_text for item in corpus))
    prompts = ChatPromptBuilder(tokenizer, 8192)
    diagnostics = []
    selected, prompt = prompts.select_evidence(
        "Founder and funding?",
        [],
        corpus,
        subquestions=questions,
        evidence_limit=3,
        evidence_token_budget=12,
        diagnostics=diagnostics,
    )
    assert corpus[0] in selected and corpus[3] in selected
    assert corpus[1] not in selected
    assert sum(prompts.token_count(item.chunk.original_text) for item in selected) <= 12
    assert prompts.token_count(prompt) <= 8192
    assert any(item["decision"] == "near_duplicate" for item in diagnostics)


@pytest.mark.parametrize(
    "question, route",
    [
        ("List every occurrence of the company", QueryRoute.AGGREGATION_LIST),
        ("Extract all patient IDs", QueryRoute.AGGREGATION_LIST),
        ("Summarize the entire document", QueryRoute.GLOBAL_SUMMARY),
        ("แสดงรายการยาทั้งหมด", QueryRoute.AGGREGATION_LIST),
        ("สรุปเอกสารทั้งหมด", QueryRoute.GLOBAL_SUMMARY),
        ("列出所有公司", QueryRoute.AGGREGATION_LIST),
        ("List all accessible files", None),
        ("All I need is the refund date", None),
        ("Who founded the company?", None),
    ],
)
def test_exhaustive_routing_guardrails(question, route) -> None:
    assert exhaustive_route(question) is route


async def test_guardrails_override_a_local_plan_and_expansion_is_bounded() -> None:
    provider = RoutingProvider(
        plan_for(
            QueryRoute.LOCAL_SEARCH,
            query="companies",
            retrieval_queries=["company names", "business names", "organizations"],
            subquestions=["founder", "funding"],
        )
    )
    plan = await understand_request(
        provider,
        tokenizer_for("companies"),
        question="List every company",
        history=[],
        context_window_tokens=8192,
        max_output_tokens=512,
        settings=RagSettings(rag_multi_query_enabled=True, rag_multi_query_count=2),
    )
    assert plan.query_route is QueryRoute.AGGREGATION_LIST
    # Local query execution obeys the configured variant bound.
    queries = retrieval_queries(
        plan, RagSettings(rag_multi_query_enabled=True, rag_multi_query_count=2)
    )
    assert len(queries) <= 4
    assert len({query.text for query in queries}) == len(queries)
    assert json.loads(provider.plan)["subquestions"] == ["founder", "funding"]


class MultiQueryProvider(RoutingProvider):
    async def embed(self, request):
        self.embeddings.append(request)
        return EmbeddingOutput(
            model="test",
            vectors=[[float(index + 1), 0, 0] for index in range(len(request.texts))],
            dimension=3,
        )


async def test_multi_query_graph_batches_embeddings_and_merges_scored_candidates(
    monkeypatch,
) -> None:
    corpus = corpus_for(["Refund policy is thirty days.", "Termination notice is sixty days."])
    provider = MultiQueryProvider(
        plan_for(
            QueryRoute.LOCAL_SEARCH,
            query="cancellation policy",
            retrieval_queries=["termination policy", "cancellation notice", "right to terminate"],
        )
    )
    settings = RagSettings(
        rag_multi_query_enabled=True,
        rag_multi_query_count=2,
        rag_claim_validation_enabled=False,
        rag_sufficiency_check_enabled=False,
    )
    executor, request, retrieval = execute_with(
        monkeypatch, corpus, provider, hops=[[corpus[0]], [corpus[0], corpus[1]]], settings=settings
    )
    result = await executor.retrieve_only(request)
    assert len(provider.embeddings) == 1 and len(provider.embeddings[0].texts) == 2
    assert retrieval.vector_reads == 2
    assert len(result.retrieved_candidates) == len(result.vector_candidates) == 2
    first = next(
        item for item in result.retrieved_candidates if item.chunk.id == corpus[0].chunk.id
    )
    assert len(first.retrieval.query_hits) == 2
    assert len(result.diagnostics.searches) == 2


async def test_independent_subquestions_stay_local_and_are_retrieved_separately(
    monkeypatch,
) -> None:
    corpus = corpus_for(["Ada founded the company.", "It raised ten million."])
    provider = MultiQueryProvider(
        plan_for(
            QueryRoute.LOCAL_SEARCH,
            query="company founders and funding",
            subquestions=["Who founded the company?", "How much did it raise?"],
        )
    )
    executor, request, retrieval = execute_with(
        monkeypatch, corpus, provider, hops=[[corpus[0]], [corpus[0]], [corpus[1]]]
    )
    result = await executor.retrieve_only(request)
    assert result.diagnostics.route is QueryRoute.LOCAL_SEARCH
    assert retrieval.vector_reads == 3
    assert {item.chunk.id for item in result.selected_evidence} == {
        item.chunk.id for item in corpus
    }
    assert not any(
        item.prompt.startswith("Plan the next retrieval step") for item in provider.generations
    )


async def test_neighbor_graph_keeps_source_order_and_token_budget(monkeypatch) -> None:
    corpus = corpus_for(["previous context", "anchor fact", "next context", "far away"])
    provider = RoutingProvider(plan_for(QueryRoute.LOCAL_SEARCH))
    settings = RagSettings(
        rag_neighbor_expansion_enabled=True,
        rag_evidence_token_budget=6,
        rag_final_evidence_limit=3,
        rag_claim_validation_enabled=False,
        rag_sufficiency_check_enabled=False,
    )
    executor, request, retrieval = execute_with(
        monkeypatch, corpus, provider, hops=[[corpus[1]]], settings=settings
    )

    async def neighbors(sources, anchors, config, *, window):
        assert len(anchors) == 1 and anchors[0].chunk.id == corpus[1].chunk.id
        assert window == 1
        return corpus

    monkeypatch.setattr(retrieval, "retrieve_neighbors", neighbors, raising=False)
    result = await executor.retrieve_only(request)
    assert [item.chunk.ordinal for item in result.selected_evidence] == [0, 1, 2]
    assert {item.chunk.ordinal for item in result.neighbor_chunks} == {0, 2}
    assert sum(item.chunk.token_count for item in result.selected_evidence) <= 6


async def test_out_of_scope_retrieval_never_reaches_reranking_or_generation(monkeypatch) -> None:
    corpus = corpus_for(["Authorized fact."])
    outside = corpus_for(["Forbidden fact."])
    provider = RoutingProvider(plan_for(QueryRoute.LOCAL_SEARCH))
    executor, request, _ = execute_with(monkeypatch, corpus, provider, hops=[outside])
    with pytest.raises(ServiceError, match="authorized source snapshot"):
        await executor.answer(request)
    assert len(provider.generations) == 1


def test_new_facts_and_negation_are_not_removed_as_near_duplicates() -> None:
    from rag_app.retrieval.evidence import near_duplicate

    repeated = "The service provides documented customer support. " * 50
    assert not near_duplicate(
        repeated + "Refunds are allowed.", repeated + "Refunds are not allowed."
    )
    assert not near_duplicate(repeated + "Thirty days.", repeated + "Forty days.")
