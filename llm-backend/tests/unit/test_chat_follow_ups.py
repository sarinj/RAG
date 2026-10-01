from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import uuid4

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from rag_app.config import IndexConfiguration, RagSettings
from rag_app.generation.answers import INSUFFICIENT_EVIDENCE_ANSWER
from rag_app.generation.chat import BaselineChatExecutor
from rag_app.generation.intents import ChatIntent
from rag_app.generation.prompts import ChatPromptBuilder
from rag_app.generation.query_rewriting import understand_request
from rag_app.generation.reranking import rerank_candidates
from rag_app.models import ChatRequest, DocumentMetadata, EvidenceChunk, EvidenceRecord
from rag_app.providers.models import (
    EmbeddingInput,
    EmbeddingOutput,
    GenerationInput,
    GenerationOutput,
)

SOLARIS = (
    "Project Solaris is HCA's municipal energy optimization program. "
    "It forecasts building energy demand and optimizes battery dispatch. "
    "Rafael Mendes leads the project."
)
UNRELATED = "Street lighting maintenance covers municipal lamps and electricity meters. "
LONG_ANSWER = f"{SOLARIS} {'It reduces peak electricity demand and energy costs. ' * 10}[S1]"
SHORT_ANSWER = (
    "Project Solaris helps municipal buildings use less energy and cut electricity costs. [S1]"
)
HISTORY = [
    {"role": "user", "content": "Who leads Project Solaris?"},
    {"role": "assistant", "content": "Rafael Mendes [S9]"},
    {"role": "user", "content": "what is that project about?"},
    {"role": "assistant", "content": f"{SOLARIS} [S9]"},
]


def plan_response(
    query: str = "Project Solaris purpose",
    *,
    intent: str = "source_question",
    max_words: int | None = None,
    definition_subject: str | None = None,
    topic_message_index: int | None = None,
    shorten: bool = False,
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "query": query,
            "max_words": max_words,
            "definition_subject": definition_subject,
            "topic_message_index": topic_message_index,
            "direct_answer": None,
            "shorten": shorten,
        }
    )


def tokenizer_for(text: str) -> Tokenizer:
    pre_tokenizer = Whitespace()
    words = dict.fromkeys(word for word, _ in pre_tokenizer.pre_tokenize_str(text))
    vocab = {word: index for index, word in enumerate(["[UNK]", *words])}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizer
    return tokenizer


def evidence_for(text: str) -> EvidenceRecord:
    return EvidenceRecord(
        document_id=uuid4(),
        document_name="helios.txt",
        chunk=EvidenceChunk(
            id=uuid4(),
            generation_id=uuid4(),
            ordinal=0,
            original_text=text,
            context_text=None,
            searchable_text=text,
            page_start=None,
            page_end=None,
            heading=None,
            source_start=0,
            source_end=len(text),
            token_count=len(text.split()),
        ),
    )


class ScriptedProvider:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = iter(responses)
        self.generations: list[GenerationInput] = []
        self.embeddings: list[EmbeddingInput] = []

    async def generate(self, request: GenerationInput) -> GenerationOutput:
        self.generations.append(request)
        return GenerationOutput(model="test", response=next(self.responses), done=True)

    async def embed(self, request: EmbeddingInput) -> EmbeddingOutput:
        self.embeddings.append(request)
        return EmbeddingOutput(model="test", vectors=[[0.1, 0.2, 0.3]], dimension=3)


class ScopedRetrieval:
    def __init__(self, evidence: EvidenceRecord) -> None:
        self.evidence = evidence

    async def list_accessible_documents(self, _sources) -> list[DocumentMetadata]:
        return [
            DocumentMetadata(
                document_id=self.evidence.document_id, name=self.evidence.document_name
            )
        ]

    async def sources_match_index(self, sources, _config) -> bool:
        return (
            len(sources) == 1
            and sources[0].document_id == self.evidence.document_id
            and sources[0].generation_id == self.evidence.chunk.generation_id
        )

    async def retrieve(self, _sources, _embedding, _config, **_kwargs) -> list[EvidenceRecord]:
        return [self.evidence]

    async def retrieve_active(self, _sources, _config) -> list[EvidenceRecord]:
        return [self.evidence]


async def test_reranker_can_read_solaris_after_an_unrelated_opening_section() -> None:
    text = UNRELATED * 60 + SOLARIS
    relevant = evidence_for(text)
    other = evidence_for(UNRELATED)
    provider = ScriptedProvider(['{"ranking":[2,1]}'])
    tokenizer = tokenizer_for(text)

    ranked = await rerank_candidates(
        provider,
        tokenizer,
        query="what is solaris?",
        candidates=[other, relevant],
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    passages = json.loads(provider.generations[0].prompt.split("Candidates: ", 1)[1])
    assert "Project Solaris" in passages[1]["text"]
    assert "municipal energy optimization program" in passages[1]["text"]
    assert relevant.chunk.id in {item.chunk.id for item in ranked}


async def test_reranking_cannot_discard_the_strongest_retrieved_source() -> None:
    relevant = evidence_for(SOLARIS)
    candidates = [relevant, *[evidence_for(UNRELATED) for _ in range(6)]]
    provider = ScriptedProvider(['{"ranking":[2,3,4,5,6]}'])

    ranked = await rerank_candidates(
        provider,
        tokenizer_for(SOLARIS + UNRELATED),
        query="what is solaris?",
        candidates=candidates,
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert relevant.chunk.id in {item.chunk.id for item in ranked}
    assert len(ranked) == 5
    assert len({record.chunk.id for record in ranked}) == 5


@pytest.mark.parametrize("response", ["not JSON", '{"query":"too long"}', '{"query":""}'])
async def test_failed_rewrite_keeps_the_subject_of_a_referential_question(response: str) -> None:
    provider = ScriptedProvider([response, response])
    tokenizer = tokenizer_for(SOLARIS + " Who leads Project Solaris? too long")

    plan = await understand_request(
        provider,
        tokenizer,
        question="what is that project about?",
        history=HISTORY,
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert "Solaris" in plan.query
    assert plan.intent is ChatIntent.SOURCE_QUESTION


@pytest.mark.parametrize("question", ["too long", "summarize it shortly", "make it shorter"])
async def test_presentation_request_uses_the_llm_to_identify_topic_and_brevity(
    question: str,
) -> None:
    word_limit = 50 if question == "summarize it shortly" else 25
    provider = ScriptedProvider(
        [
            plan_response(
                intent="answer_revision", max_words=word_limit, topic_message_index=2, shorten=True
            )
        ]
    )
    plan = await understand_request(
        provider,
        tokenizer_for(SOLARIS + " Who leads Project Solaris?"),
        question=question,
        history=HISTORY,
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert "Solaris" in plan.query
    assert plan.intent is ChatIntent.ANSWER_REVISION
    assert plan.shorten
    assert plan.max_words == min(word_limit, len(HISTORY[-1]["content"].split()) - 1)
    assert len(provider.generations) == 1


async def test_failed_rewrite_preserves_a_question_that_changes_subject() -> None:
    provider = ScriptedProvider(["not JSON", "not JSON"])
    plan = await understand_request(
        provider,
        tokenizer_for(SOLARIS),
        question="Who leads Project Echo?",
        history=HISTORY,
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert plan.query.startswith("Who leads Project Echo?")
    assert plan.intent is ChatIntent.SOURCE_QUESTION


def test_old_history_cannot_crowd_the_best_source_out_of_a_follow_up_prompt() -> None:
    evidence = evidence_for(SOLARIS)
    tokenizer = tokenizer_for(SOLARIS + " Earlier topic old context")
    history = [
        {"role": "user", "content": "Earlier topic"},
        {"role": "assistant", "content": "old context " * 300},
    ]
    counter = ChatPromptBuilder(tokenizer, 100_000)
    budget = (
        counter.token_count(
            counter.render("too long", history, [], retrieval_query="Project Solaris")
        )
        + 5
    )
    prompts = ChatPromptBuilder(tokenizer, budget)

    selected, prompt = prompts.select_evidence(
        "too long", history, [evidence], retrieval_query="Project Solaris"
    )

    assert selected == [evidence]
    assert prompts.token_count(prompt) <= budget
    assert "Project Solaris" in prompt


def executor_for(monkeypatch, evidence: EvidenceRecord, provider: ScriptedProvider):
    monkeypatch.setattr(
        "rag_app.generation.chat.RetrievalRepository", lambda _database: ScopedRetrieval(evidence)
    )
    config = IndexConfiguration.model_validate(
        {
            "embeddingModel": "test",
            "embeddingDigest": "a" * 64,
            "embeddingDimension": 3,
            "chunkTargetTokens": 512,
            "chunkOverlapTokens": 64,
            "chunkerVersion": "test",
            "tokenizerVersion": "test",
            "contextPromptVersion": None,
            "contextModel": None,
            "contextModelDigest": None,
            "contextWindowTokens": 8192,
            "maxOutputTokens": 512,
        }
    )
    tokenizer = tokenizer_for(SOLARIS + SHORT_ANSWER + " Who leads Project Solaris?")
    return BaselineChatExecutor(
        object(),
        provider,
        config,
        tokenizer,
        RagSettings(rag_sufficiency_check_enabled=False, rag_claim_validation_enabled=False),
    )  # type: ignore[arg-type]


def request_for(evidence: EvidenceRecord, question: str) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "contractVersion": "1.1.0",
            "requestId": str(uuid4()),
            "question": question,
            "history": HISTORY,
            "sources": [
                {
                    "documentId": str(evidence.document_id),
                    "generationId": str(evidence.chunk.generation_id),
                }
            ],
            "retrievalMode": "hybrid",
        }
    )


@pytest.mark.parametrize("question,word_limit", [("too long", 25), ("summarize it shortly", 50)])
async def test_shortening_follow_up_retrieves_the_topic_and_repairs_a_verbose_answer(
    monkeypatch, question: str, word_limit: int
) -> None:
    evidence = evidence_for(SOLARIS)
    provider = ScriptedProvider(
        [
            plan_response(intent="answer_revision", max_words=word_limit, topic_message_index=2),
            json.dumps({"answer": LONG_ANSWER}),
            json.dumps({"answer": SHORT_ANSWER}),
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)

    result = await executor.answer(request_for(evidence, question))

    assert "Solaris" in provider.embeddings[0].texts[0]
    assert len(provider.generations) == 3
    assert result.response.answer_status == "answered"
    assert len(result.response.answer.split()) <= word_limit
    assert result.response.citations[0].chunk_id == evidence.chunk.id
    assert result.evidence_ids == frozenset(
        {(evidence.document_id, evidence.chunk.generation_id, evidence.chunk.id)}
    )
    assert "[S9]" not in result.response.answer


@pytest.mark.parametrize("repair", ["not JSON", '{"answer":"INSUFFICIENT_EVIDENCE"}'])
async def test_failed_length_repair_does_not_turn_a_supported_answer_into_an_abstention(
    monkeypatch, repair: str
) -> None:
    evidence = evidence_for(SOLARIS)
    provider = ScriptedProvider(
        [
            plan_response(intent="answer_revision", max_words=25, topic_message_index=2),
            json.dumps({"answer": LONG_ANSWER}),
            repair,
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)

    result = await executor.answer(request_for(evidence, "too long"))

    assert result.response.answer_status == "answered"
    assert result.response.answer == f"{SOLARIS} [S1]"
    assert len(provider.generations) == 3


@pytest.mark.parametrize("repair", [json.dumps({"answer": LONG_ANSWER}), "not JSON"])
async def test_shortening_recovers_a_complete_source_statement_if_the_retry_is_still_long(
    monkeypatch, repair: str
) -> None:
    evidence = evidence_for(SOLARIS.replace(". ", ".\n\n"))
    provider = ScriptedProvider(
        [
            plan_response(intent="answer_revision", max_words=25, topic_message_index=2),
            json.dumps({"answer": LONG_ANSWER}),
            repair,
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)

    result = await executor.answer(request_for(evidence, "too long"))

    assert result.response.answer == (
        "Project Solaris is HCA's municipal energy optimization program. [S1]"
    )
    assert len(result.response.answer.split()) <= 25
    assert result.response.citations[0].chunk_id == evidence.chunk.id
    assert len(provider.generations) == 3


async def test_failed_shortening_turns_do_not_replace_the_last_supported_topic(monkeypatch) -> None:
    evidence = evidence_for(SOLARIS)
    provider = ScriptedProvider(
        [
            plan_response(intent="answer_revision", max_words=25, topic_message_index=2),
            json.dumps({"answer": SHORT_ANSWER}),
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)
    failed_pair = [
        {"role": "user", "content": "too long"},
        {"role": "assistant", "content": INSUFFICIENT_EVIDENCE_ANSWER},
    ]
    request = request_for(evidence, "too long").model_copy(
        update={
            "history": ChatRequest.model_validate(
                {
                    **request_for(evidence, "too long").model_dump(by_alias=True),
                    "history": [*HISTORY, *failed_pair, *failed_pair],
                }
            ).history,
        }
    )

    result = await executor.answer(request)

    assert result.response.answer_status == "answered"
    assert "Solaris" in provider.embeddings[0].texts[0]
    assert INSUFFICIENT_EVIDENCE_ANSWER not in provider.generations[1].prompt
    assert len(provider.generations) == 2


async def test_unanswered_question_that_names_a_new_subject_remains_the_topic() -> None:
    provider = ScriptedProvider(
        [
            plan_response(
                "Who leads Project Echo?",
                intent="answer_revision",
                max_words=25,
                topic_message_index=4,
            )
        ]
    )
    history = [
        *HISTORY,
        {"role": "user", "content": "Who leads Project Echo?"},
        {"role": "assistant", "content": INSUFFICIENT_EVIDENCE_ANSWER},
    ]
    plan = await understand_request(
        provider,
        tokenizer_for(SOLARIS + " Who leads Project Echo?"),
        question="too long",
        history=history,
        context_window_tokens=8192,
        max_output_tokens=512,
    )

    assert "Echo" in plan.query
    assert "Solaris" not in plan.query
    assert plan.topic_message_index == 4


async def test_definition_abstention_gets_one_grounded_retry(monkeypatch) -> None:
    evidence = evidence_for(SOLARIS)
    provider = ScriptedProvider(
        [
            plan_response(),
            '{"answer":"INSUFFICIENT_EVIDENCE"}',
            json.dumps({"answer": SHORT_ANSWER}),
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)
    request = request_for(evidence, "Explain Project Solaris").model_copy(update={"history": []})

    result = await executor.answer(request)

    assert result.response.answer_status == "answered"
    assert result.response.citations[0].chunk_id == evidence.chunk.id
    assert len(provider.generations) == 3


async def test_informal_definition_uses_the_source_if_generation_incorrectly_abstains(
    monkeypatch,
) -> None:
    evidence = evidence_for(SOLARIS)
    provider = ScriptedProvider(
        [
            plan_response(definition_subject="Solaris"),
            '{"answer":"INSUFFICIENT_EVIDENCE"}',
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)
    request = request_for(evidence, "what is solaris?").model_copy(update={"history": []})

    result = await executor.answer(request)

    assert result.response.answer == f"{SOLARIS} [S1]"
    assert result.response.answer_status == "answered"
    assert result.response.citations[0].chunk_id == evidence.chunk.id
    assert len(provider.generations) == 2


@pytest.mark.parametrize(
    "source,question",
    [
        (SOLARIS, "What is Solaris's source code URL?"),
        (SOLARIS + "\nSolaris is an operating system.", "what is solaris?"),
        (SOLARIS + "\nIt is led by Rafael Mendes.", "what is it?"),
    ],
)
async def test_unanswerable_or_ambiguous_question_still_abstains_after_one_retry(
    monkeypatch, source: str, question: str
) -> None:
    evidence = evidence_for(source)
    provider = ScriptedProvider(
        [
            plan_response(
                question,
                definition_subject="Solaris" if question == "what is solaris?" else None,
            ),
            *['{"answer":"INSUFFICIENT_EVIDENCE"}'] * 2,
        ]
    )
    executor = executor_for(monkeypatch, evidence, provider)
    request = request_for(evidence, question).model_copy(update={"history": []})

    result = await executor.answer(request)

    assert result.response.answer_status == "insufficient_evidence"
    assert result.response.citations == []
    assert len(provider.generations) == 3
