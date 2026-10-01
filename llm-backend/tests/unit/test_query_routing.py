from __future__ import annotations

import json
from collections.abc import Callable
from uuid import uuid4

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from rag_app.config import IndexConfiguration, RagSettings
from rag_app.errors import ProviderProtocolError, ServiceError
from rag_app.generation.chat import BaselineChatExecutor
from rag_app.generation.corpus import CorpusProcessor
from rag_app.generation.intents import ChatIntent, ChatPlan
from rag_app.generation.prompts import ChatPromptBuilder
from rag_app.generation.query_rewriting import understand_request
from rag_app.models import (
    ChatRequest,
    DocumentMetadata,
    EvidenceChunk,
    EvidenceRecord,
    QueryRoute,
)
from rag_app.providers.models import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)


def tokenizer_for(text: str) -> Tokenizer:
    splitter = Whitespace()
    words = dict.fromkeys(word for word, _ in splitter.pre_tokenize_str(text))
    tokenizer = Tokenizer(
        WordLevel(
            vocab={word: index for index, word in enumerate(["[UNK]", *words])},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = splitter
    return tokenizer


def configuration(context: int = 8192, output: int = 512) -> IndexConfiguration:
    return IndexConfiguration(
        embedding_model="test",
        embedding_digest="a" * 64,
        embedding_dimension=3,
        chunk_target_tokens=512,
        chunk_overlap_tokens=64,
        chunker_version="test",
        tokenizer_version="test",
        context_prompt_version=None,
        context_model=None,
        context_model_digest=None,
        context_window_tokens=context,
        max_output_tokens=output,
    )


def corpus_for(texts: list[str], name: str = "report.txt") -> list[EvidenceRecord]:
    document_id, generation_id = uuid4(), uuid4()
    offset = 0
    records = []
    for ordinal, text in enumerate(texts):
        records.append(
            EvidenceRecord(
                document_id=document_id,
                document_name=name,
                chunk=EvidenceChunk(
                    id=uuid4(),
                    generation_id=generation_id,
                    ordinal=ordinal,
                    original_text=text,
                    searchable_text=text,
                    context_text=None,
                    page_start=None,
                    page_end=None,
                    heading=f"Section {ordinal}",
                    source_start=offset,
                    source_end=offset + len(text),
                    token_count=len(text.split()),
                ),
            )
        )
        offset += len(text)
    return records


def plan_for(route: QueryRoute, **updates: object) -> str:
    return json.dumps(
        {
            "intent": "source_question",
            "query": "report contents",
            "max_words": None,
            "definition_subject": None,
            "topic_message_index": None,
            "direct_answer": None,
            "previous_intent": None,
            "query_route": route.value,
            "document_ids": [],
            "subqueries": [],
            "shorten": False,
            **updates,
        }
    )


class RoutingProvider:
    def __init__(
        self,
        plan: str,
        *,
        answer: str = "A supported answer [S1].",
        searches: list[str | None] | None = None,
        extraction: Callable[[list[dict]], dict] | None = None,
        reduction: Callable[[list[dict]], dict] | None = None,
        answer_reason: str | None = None,
    ) -> None:
        self.plan = plan
        self.answer = answer
        self.searches = iter(searches or [None])
        self.extraction = extraction
        self.reduction = reduction
        self.answer_reason = answer_reason
        self.generations: list[GenerationInput] = []
        self.embeddings: list[EmbeddingInput] = []
        self.passages: list[dict] = []

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput:
        self.embeddings.append(request)
        return EmbeddingOutput(
            model="test", vectors=[[float(len(self.embeddings)), 0.0, 0.0]], dimension=3
        )

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        self.generations.append(request)
        reason = None
        if request.prompt.startswith("Interpret the current user message"):
            response = self.plan
        elif "\nPassages: " in request.prompt:
            passages = json.loads(
                request.prompt.split("\nPassages: ", 1)[1].split("\nREPAIR:", 1)[0]
            )
            self.passages.extend(passages)
            response = json.dumps(
                self.extraction(passages)
                if self.extraction
                else {
                    "facts": [{"text": item["text"], "sources": [item["id"]]} for item in passages],
                    "complete": True,
                }
            )
        elif "\nNotes: " in request.prompt:
            notes = json.loads(request.prompt.split("\nNotes: ", 1)[1])
            response = json.dumps(
                self.reduction(notes)
                if self.reduction
                else {
                    "facts": notes,
                    "complete": True,
                }
            )
        elif request.prompt.startswith("Plan the next retrieval step"):
            response = json.dumps({"query": next(self.searches)})
        elif request.prompt.startswith("Rank the candidate passages"):
            response = '{"ranking":[1,2,3,4,5]}'
        else:
            response = json.dumps({"answer": self.answer})
            reason = self.answer_reason
        return GenerationOutput(model="test", response=response, done=True, done_reason=reason)


class ScopedCorpus:
    def __init__(
        self, corpus: list[EvidenceRecord], hops: list[list[EvidenceRecord]] | None = None
    ):
        self.corpus = corpus
        self.hops = hops
        self.vector_reads = 0
        self.full_reads = 0

    def scoped(self, sources) -> list[EvidenceRecord]:
        allowed = {(source.document_id, source.generation_id) for source in sources}
        return [
            item for item in self.corpus if (item.document_id, item.chunk.generation_id) in allowed
        ]

    async def list_accessible_documents(self, sources) -> list[DocumentMetadata]:
        documents = {item.document_id: item.document_name for item in self.scoped(sources)}
        return [
            DocumentMetadata(document_id=document_id, name=name)
            for document_id, name in documents.items()
        ]

    async def list_accessible_document_names(self, sources) -> list[str]:
        return [document.name for document in await self.list_accessible_documents(sources)]

    async def sources_match_index(self, sources, _config) -> bool:
        return bool(self.scoped(sources))

    async def retrieve(self, sources, _embedding, _config, **_kwargs) -> list[EvidenceRecord]:
        index = self.vector_reads
        self.vector_reads += 1
        return self.hops[index] if self.hops else self.scoped(sources)[:1]

    async def retrieve_active(self, sources, _config) -> list[EvidenceRecord]:
        return self.scoped(sources)

    async def retrieve_all(self, sources, _config) -> list[EvidenceRecord]:
        self.full_reads += 1
        return self.scoped(sources)


def execute_with(monkeypatch, corpus, provider, *, hops=None, settings=None):
    retrieval = ScopedCorpus(corpus, hops)
    monkeypatch.setattr("rag_app.generation.chat.RetrievalRepository", lambda _database: retrieval)
    tokenizer = tokenizer_for(" ".join(item.chunk.original_text for item in corpus))
    executor = BaselineChatExecutor(
        object(),
        provider,
        configuration(),
        tokenizer,
        settings
        or RagSettings(rag_sufficiency_check_enabled=False, rag_claim_validation_enabled=False),
    )
    documents = {item.document_id: item.chunk.generation_id for item in corpus}
    request = ChatRequest.model_validate(
        {
            "contractVersion": "1.1.0",
            "requestId": str(uuid4()),
            "question": "Read the report",
            "history": [],
            "retrievalMode": "vector",
            "sources": [
                {"documentId": str(document_id), "generationId": str(generation_id)}
                for document_id, generation_id in documents.items()
            ],
        }
    )
    return executor, request, retrieval


@pytest.mark.parametrize("route", [QueryRoute.GLOBAL_SUMMARY, QueryRoute.AGGREGATION_LIST])
async def test_whole_document_routes_cover_chunks_beyond_top_k_and_keep_original_citations(
    monkeypatch,
    route: QueryRoute,
) -> None:
    corpus = corpus_for([f"Company {index} operates in region {index}." for index in range(12)])
    provider = RoutingProvider(plan_for(route), answer="Company 11 operates in region 11 [S12].")
    executor, request, retrieval = execute_with(monkeypatch, corpus, provider)

    result = await executor.answer(request)

    assert retrieval.full_reads == 1 and retrieval.vector_reads == 0
    assert not provider.embeddings
    assert {item["ordinal"] for item in provider.passages} == set(range(12))
    assert result.diagnostics.route is route
    assert result.diagnostics.corpus_chunk_count == result.diagnostics.processed_chunk_count == 12
    assert result.response.citations[0].chunk_id == corpus[-1].chunk.id
    assert result.response.answer_status == "answered"
    final_prompt = provider.generations[-1].prompt
    assert "derived notes from the complete scoped corpus" in final_prompt
    assert "Company 11" in final_prompt


async def test_focused_question_keeps_normal_retrieval(monkeypatch) -> None:
    corpus = corpus_for(["The lead is Priya.", "The budget is ten dollars."])
    provider = RoutingProvider(plan_for(QueryRoute.LOCAL_SEARCH))
    executor, request, retrieval = execute_with(monkeypatch, corpus, provider)

    result = await executor.answer(request)

    assert retrieval.vector_reads == 1 and retrieval.full_reads == 0
    assert len(provider.embeddings) == 1
    assert result.diagnostics.route is QueryRoute.LOCAL_SEARCH
    assert result.response.citations[0].chunk_id == corpus[0].chunk.id


async def test_named_document_selection_cannot_include_other_documents(monkeypatch) -> None:
    requested = corpus_for(["Selected company is Orion."], "selected.txt")
    other = corpus_for(["Private company is Echo."], "other.txt")
    provider = RoutingProvider(
        plan_for(QueryRoute.GLOBAL_SUMMARY, document_ids=[str(requested[0].document_id)])
    )
    executor, request, _retrieval = execute_with(monkeypatch, [*requested, *other], provider)

    result = await executor.answer(request)

    assert result.diagnostics.processed_chunk_count == 1
    assert {item["document"] for item in provider.passages} == {"selected.txt"}
    assert "Private company" not in provider.generations[-1].prompt


async def test_global_route_rejects_a_partially_changed_source_snapshot(monkeypatch) -> None:
    first = corpus_for(["First document."], "first.txt")
    second = corpus_for(["Second document."], "second.txt")
    provider = RoutingProvider(plan_for(QueryRoute.GLOBAL_SUMMARY))
    executor, request, retrieval = execute_with(monkeypatch, [*first, *second], provider)

    async def partial_read(_sources, _config):
        return first

    monkeypatch.setattr(retrieval, "retrieve_all", partial_read)

    with pytest.raises(ServiceError) as failure:
        await executor.answer(request)

    assert failure.value.code == "SCOPE_CHANGED"
    assert not provider.passages


async def test_list_route_merges_overlap_duplicates_with_both_source_ids(monkeypatch) -> None:
    corpus = corpus_for(["Orion", "Orion", "Echo"])
    provider = RoutingProvider(
        plan_for(QueryRoute.AGGREGATION_LIST), answer="Orion [S1] and Echo [S3]."
    )
    executor, request, _retrieval = execute_with(monkeypatch, corpus, provider)

    result = await executor.answer(request)

    notes = json.loads(
        provider.generations[-1]
        .prompt.split("refer to original chunks): ", 1)[1]
        .split("\nAnswer:", 1)[0]
    )
    assert notes == [
        {"text": "Orion", "sources": ["S1", "S2"]},
        {"text": "Echo", "sources": ["S3"]},
    ]
    assert len(result.evidence_ids) == 3
    assert result.response.citations[-1].chunk_id == corpus[-1].chunk.id


async def test_brevity_does_not_trigger_lossy_shortening_of_an_exhaustive_list(monkeypatch) -> None:
    corpus = corpus_for(["Orion", "Echo", "Solaris"])
    answer = "Orion [S1], Echo [S2], and Solaris [S3]."
    provider = RoutingProvider(
        plan_for(QueryRoute.AGGREGATION_LIST, max_words=2),
        answer=answer,
    )
    executor, request, _retrieval = execute_with(monkeypatch, corpus, provider)

    result = await executor.answer(request)

    assert result.response.answer == answer
    assert len(provider.generations) == 5  # planning, three extractions, final answer
    assert "Do not remove items to meet a word limit" in provider.generations[-1].prompt


async def test_iterative_search_resolves_second_hop_from_first_hop_evidence(monkeypatch) -> None:
    corpus = corpus_for(["Project Orion is led by Priya.", "Priya works for Harborlight."])
    provider = RoutingProvider(
        plan_for(
            QueryRoute.MULTI_HOP,
            query="Orion leader",
            subqueries=["Project Orion leader"],
        ),
        searches=["Priya employer", None],
        answer="Orion's lead is Priya [S1], who works for Harborlight [S2].",
    )
    executor, request, retrieval = execute_with(
        monkeypatch, corpus, provider, hops=[[corpus[0]], [corpus[1]]]
    )
    question = "Which organization employs Orion's lead?"
    request = request.model_copy(update={"question": question})

    result = await executor.answer(request)

    assert result.diagnostics.searches == ["Project Orion leader", "Priya employer"]
    assert [item.texts[0] for item in provider.embeddings] == result.diagnostics.searches
    planner_prompts = [
        item.prompt
        for item in provider.generations
        if item.prompt.startswith("Plan the next retrieval step")
    ]
    assert "Project Orion is led by Priya" in planner_prompts[0]
    assert question in planner_prompts[0]
    assert retrieval.vector_reads == 2
    assert {citation.chunk_id for citation in result.response.citations} == {
        item.chunk.id for item in corpus
    }


async def test_multi_hop_stops_at_three_rounds(monkeypatch) -> None:
    corpus = corpus_for(["First link.", "Second link.", "Third link."])
    provider = RoutingProvider(
        plan_for(QueryRoute.MULTI_HOP), searches=["second search", "third search"]
    )
    executor, request, retrieval = execute_with(
        monkeypatch, corpus, provider, hops=[[item] for item in corpus]
    )

    result = await executor.answer(request)

    assert retrieval.vector_reads == len(result.diagnostics.searches) == 3


async def test_multi_hop_does_not_repeat_an_identical_search(monkeypatch) -> None:
    corpus = corpus_for(["The lead is Priya."])
    provider = RoutingProvider(
        plan_for(QueryRoute.MULTI_HOP, query="Orion leader"), searches=["ORION LEADER"]
    )
    executor, request, retrieval = execute_with(monkeypatch, corpus, provider)

    await executor.answer(request)

    assert retrieval.vector_reads == 1


async def test_planner_rejects_out_of_scope_document_selection() -> None:
    corpus = corpus_for(["A fact."])
    provider = RoutingProvider(plan_for(QueryRoute.GLOBAL_SUMMARY, document_ids=[str(uuid4())]))

    plan = await understand_request(
        provider,
        tokenizer_for("A fact."),
        question="Summarize the report",
        history=[],
        context_window_tokens=8192,
        max_output_tokens=512,
        documents=[DocumentMetadata(document_id=corpus[0].document_id, name="report.txt")],
    )

    assert not plan.document_ids
    assert len(provider.generations) == 2


async def test_brevity_and_global_summary_are_independent() -> None:
    provider = RoutingProvider(plan_for(QueryRoute.GLOBAL_SUMMARY, max_words=50))
    plan = await understand_request(
        provider,
        tokenizer_for("Summary of the report"),
        question="Summarize the entire report briefly",
        history=[],
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert plan.intent is ChatIntent.SOURCE_QUESTION
    assert plan.query_route is QueryRoute.GLOBAL_SUMMARY
    assert plan.max_words == 50 and not plan.shorten


async def test_shortening_a_global_summary_keeps_whole_document_route() -> None:
    provider = RoutingProvider(
        plan_for(
            QueryRoute.GLOBAL_SUMMARY,
            intent="answer_revision",
            max_words=25,
            shorten=True,
            topic_message_index=0,
        )
    )
    plan = await understand_request(
        provider,
        tokenizer_for("report summary shorter"),
        question="Make it shorter",
        history=[
            {"role": "user", "content": "Summarize the whole report"},
            {"role": "assistant", "content": "A report summary [S1]"},
        ],
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert plan.intent is ChatIntent.ANSWER_REVISION and plan.shorten
    assert plan.query_route is QueryRoute.GLOBAL_SUMMARY
    assert plan.max_words < 25


@pytest.mark.parametrize("intent", ["conversation", "source_catalog"])
async def test_non_factual_intents_bypass_retrieval(monkeypatch, intent: str) -> None:
    corpus = corpus_for(["A source fact."])
    provider = RoutingProvider(
        plan_for(QueryRoute.GLOBAL_SUMMARY, intent=intent, direct_answer="Hello!")
    )
    executor, request, retrieval = execute_with(monkeypatch, corpus, provider)

    result = await executor.answer(request)

    assert result.response.answer_status == "direct_answer"
    assert result.diagnostics.route is None
    assert retrieval.vector_reads == retrieval.full_reads == 0


async def test_incomplete_extraction_splits_a_passage_without_losing_original_provenance() -> None:
    corpus = corpus_for(["Orion Echo"])
    provider = RoutingProvider(
        "",
        extraction=lambda passages: {
            "facts": [{"text": item["text"].strip(), "sources": [item["id"]]} for item in passages],
            "complete": len(passages[0]["text"].split()) == 1,
        },
    )
    tokenizer = tokenizer_for("Orion Echo")
    processor = CorpusProcessor(
        provider, tokenizer, configuration(), ChatPromptBuilder(tokenizer, 7552)
    )

    result = await processor.prepare(
        "List all companies",
        "companies",
        QueryRoute.AGGREGATION_LIST,
        corpus,
        max_words=None,
        intent=ChatIntent.SOURCE_QUESTION,
    )

    assert [note["text"] for note in result.notes] == ["Orion", "Echo"]
    assert result.selected == tuple(corpus)
    assert len(provider.generations) == 3


async def test_invalid_map_citations_fail_after_one_repair() -> None:
    corpus = corpus_for(["Orion exists."])
    provider = RoutingProvider(
        "",
        extraction=lambda _passages: {
            "facts": [{"text": "Orion", "sources": [99]}],
            "complete": True,
        },
    )
    tokenizer = tokenizer_for("Orion exists.")
    processor = CorpusProcessor(
        provider, tokenizer, configuration(), ChatPromptBuilder(tokenizer, 7552)
    )

    with pytest.raises(ProviderProtocolError, match="Invalid cited document extraction"):
        await processor.prepare(
            "List companies",
            "companies",
            QueryRoute.AGGREGATION_LIST,
            corpus,
            max_words=None,
            intent=ChatIntent.SOURCE_QUESTION,
        )

    assert len(provider.generations) == 2


async def test_oversized_lists_fail_instead_of_being_reduced_or_sampled() -> None:
    corpus = corpus_for(
        [f"Company {index} " + "has recorded detailed activity " * 10 for index in range(12)]
    )
    provider = RoutingProvider("")
    tokenizer = tokenizer_for(" ".join(item.chunk.original_text for item in corpus))
    processor = CorpusProcessor(
        provider, tokenizer, configuration(1536), ChatPromptBuilder(tokenizer, 896)
    )

    with pytest.raises(ServiceError, match="complete extracted list") as failure:
        await processor.prepare(
            "List every record",
            "records",
            QueryRoute.AGGREGATION_LIST,
            corpus,
            max_words=None,
            intent=ChatIntent.SOURCE_QUESTION,
        )

    assert failure.value.code == "CONTEXT_LIMIT_EXCEEDED"
    assert len(provider.passages) == 12
    assert not any("\nNotes: " in request.prompt for request in provider.generations)


async def test_document_summaries_are_reduced_hierarchically_under_budget() -> None:
    corpus = corpus_for(
        [f"Topic {index} " + "has recorded detailed activity " * 10 for index in range(12)]
    )
    provider = RoutingProvider(
        "",
        reduction=lambda notes: {
            "facts": [
                {
                    "text": "Topics include "
                    + ", ".join(note["text"].split(" has ")[0] for note in notes),
                    "sources": list(
                        dict.fromkeys(source for note in notes for source in note["sources"])
                    ),
                }
            ],
            "complete": True,
        },
    )
    tokenizer = tokenizer_for(" ".join(item.chunk.original_text for item in corpus))
    prompts = ChatPromptBuilder(tokenizer, 1024)
    processor = CorpusProcessor(provider, tokenizer, configuration(1664), prompts)

    result = await processor.prepare(
        "Summarize all sections",
        "report",
        QueryRoute.GLOBAL_SUMMARY,
        corpus,
        max_words=None,
        intent=ChatIntent.SOURCE_QUESTION,
    )

    assert result.reduction_rounds > 0
    assert prompts.token_count(result.prompt) <= prompts.max_prompt_tokens
    assert "Topic 11" in result.prompt
    assert {item.chunk.id for item in result.selected} == {item.chunk.id for item in corpus}
    assert all(
        prompts.token_count(request.prompt) <= prompts.max_prompt_tokens
        for request in provider.generations
    )


async def test_truncated_final_list_is_rejected(monkeypatch) -> None:
    corpus = corpus_for(["Orion", "Echo"])
    provider = RoutingProvider(plan_for(QueryRoute.AGGREGATION_LIST), answer_reason="length")
    executor, request, _retrieval = execute_with(monkeypatch, corpus, provider)

    with pytest.raises(ServiceError, match="complete list or aggregation"):
        await executor.answer(request)


def test_factual_plan_discards_direct_answer_and_keeps_route_independent() -> None:
    plan = ChatPlan.model_validate_json(
        plan_for(
            QueryRoute.GLOBAL_SUMMARY,
            direct_answer="An uncited factual draft",
            definition_subject="Orion",
        )
    )

    assert plan.direct_answer is None
    assert plan.query_route is QueryRoute.GLOBAL_SUMMARY
