from __future__ import annotations

import asyncio
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from tokenizers import Tokenizer

from rag_app.api.chat import ChatExecutionResult
from rag_app.config import IndexConfiguration
from rag_app.errors import ProviderProtocolError, ServiceError
from rag_app.generation.answers import insufficient_evidence_response, parse_answer
from rag_app.generation.prompts import ChatPromptBuilder
from rag_app.generation.query_rewriting import rewrite_query
from rag_app.generation.reranking import rerank_candidates
from rag_app.models import ChatRequest, ChatResponse, EvidenceRecord
from rag_app.providers.models import EmbeddingInput, GenerationInput
from rag_app.providers.protocols import LLMProvider
from rag_app.repositories.database import Database
from rag_app.repositories.retrieval import RetrievalRepository, evidence_key
from rag_app.retrieval.hybrid import bm25_candidates, reciprocal_rank_fusion

REPAIR_PROMPT_TOKEN_RESERVE = 128
REPAIR_MAX_OUTPUT_TOKENS = 64


class ChatGraphState(TypedDict):
    request: ChatRequest
    history: list[dict[str, str]]
    retrieval_query: str
    candidates: list[EvidenceRecord]
    reranked_candidates: list[EvidenceRecord]
    selected: list[EvidenceRecord]
    prompt: str
    response: ChatResponse | None
    result: ChatExecutionResult | None


class BaselineChatExecutor:
    def __init__(
        self,
        database: Database,
        provider: LLMProvider,
        config: IndexConfiguration,
        tokenizer: Tokenizer,
    ) -> None:
        self._retrieval = RetrievalRepository(database)
        self._provider = provider
        self._config = config
        self._tokenizer = tokenizer
        prompt_limit = (
            config.context_window_tokens
            - config.max_output_tokens
            - REPAIR_PROMPT_TOKEN_RESERVE
        )
        self._prompts = ChatPromptBuilder(tokenizer, prompt_limit)
        graph = cast(Any, StateGraph)(ChatGraphState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("select_evidence", self._select_evidence_node)
        graph.add_node("insufficient_evidence", self._insufficient_evidence_node)
        graph.add_node("generate", self._generate)
        graph.add_node("repair", self._repair)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(cast(Any, START), "prepare")
        graph.add_edge("prepare", "rewrite_query")
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "rerank": "rerank",
                "insufficient_evidence": "insufficient_evidence",
            },
        )
        graph.add_edge("rerank", "select_evidence")
        graph.add_edge("select_evidence", "generate")
        graph.add_conditional_edges(
            "generate",
            self._route_after_generation,
            {"repair": "repair", "finalize": "finalize"},
        )
        graph.add_edge("repair", "finalize")
        graph.add_edge("insufficient_evidence", "finalize")
        graph.add_edge("finalize", cast(Any, END))
        # Chat and job persistence belong to the application database, not graph checkpoints.
        self._graph: Any = graph.compile()

    async def answer(self, request: ChatRequest) -> ChatExecutionResult:
        initial_state: ChatGraphState = {
            "request": request,
            "history": [],
            "retrieval_query": request.question,
            "candidates": [],
            "reranked_candidates": [],
            "selected": [],
            "prompt": "",
            "response": None,
            "result": None,
        }
        final_state: ChatGraphState = await self._graph.ainvoke(initial_state)
        result = final_state["result"]
        if result is None:
            raise RuntimeError("Chat graph completed without an execution result")
        return result

    async def _prepare(self, state: ChatGraphState) -> dict[str, Any]:
        request = state["request"]
        if self._prompts.max_prompt_tokens <= 0:
            raise ServiceError(
                422,
                "CONTEXT_LIMIT_EXCEEDED",
                "The configured model context is too small for chat generation.",
            )
        if not await self._retrieval.sources_match_index(request.sources, self._config):
            raise ServiceError(
                409,
                "INDEX_CONFIG_MISMATCH",
                "The selected documents use an incompatible embedding configuration.",
            )

        return {"history": self._prompts.fit_history(request)}

    async def _rewrite_query(self, state: ChatGraphState) -> dict[str, Any]:
        request = state["request"]
        retrieval_query = await rewrite_query(
            self._provider,
            self._tokenizer,
            question=request.question,
            history=state["history"],
            context_window_tokens=self._config.context_window_tokens,
            max_output_tokens=self._config.max_output_tokens,
        )
        return {"retrieval_query": retrieval_query}

    async def _retrieve(self, state: ChatGraphState) -> dict[str, Any]:
        request = state["request"]
        retrieval_query = state["retrieval_query"]
        query_embedding = await self._provider.embed(EmbeddingInput(texts=[retrieval_query]))
        if query_embedding.dimension != self._config.embedding_dimension:
            raise ProviderProtocolError("Query embedding dimension differs from the active index")

        vector_candidates = await self._retrieval.retrieve(
            request.sources,
            query_embedding.vectors[0],
            self._config,
        )
        if request.retrieval_mode == "hybrid":
            scoped_chunks = await self._retrieval.retrieve_active(request.sources, self._config)
            lexical_candidates = await asyncio.to_thread(
                bm25_candidates,
                retrieval_query,
                scoped_chunks,
                self._tokenizer,
            )
            candidates = reciprocal_rank_fusion(vector_candidates, lexical_candidates)
        else:
            candidates = vector_candidates
        return {"candidates": candidates}

    @staticmethod
    def _route_after_retrieval(state: ChatGraphState) -> str:
        return "rerank" if state.get("candidates") else "insufficient_evidence"

    async def _rerank(self, state: ChatGraphState) -> dict[str, Any]:
        ranked = await rerank_candidates(
            self._provider,
            self._tokenizer,
            query=state["retrieval_query"],
            candidates=state["candidates"],
            context_window_tokens=self._config.context_window_tokens,
            max_output_tokens=self._config.max_output_tokens,
        )
        return {"reranked_candidates": ranked}

    def _select_evidence_node(self, state: ChatGraphState) -> dict[str, Any]:
        request = state["request"]
        selected, prompt = self._prompts.select_evidence(
            request.question,
            state["history"],
            state["reranked_candidates"],
        )
        if not selected:
            raise ServiceError(
                422,
                "CONTEXT_LIMIT_EXCEEDED",
                "The selected evidence does not fit in the model context.",
            )

        return {"selected": selected, "prompt": prompt}

    async def _generate(self, state: ChatGraphState) -> dict[str, Any]:
        request = state["request"]
        selected = state["selected"]
        generation = await self._provider.generate(
            GenerationInput(
                prompt=state["prompt"],
                context_window_tokens=self._config.context_window_tokens,
                max_output_tokens=self._config.max_output_tokens,
                temperature=0,
                think=False,
                response_format="json",
            )
        )
        parsed = parse_answer(generation.response, request, selected)
        return {"response": parsed} if parsed is not None else {}

    @staticmethod
    def _route_after_generation(state: ChatGraphState) -> str:
        return "finalize" if state.get("response") is not None else "repair"

    async def _repair(self, state: ChatGraphState) -> dict[str, Any]:
        request = state["request"]
        selected = state["selected"]
        repair_prompt = (
            f"{state['prompt']}\n\n"
            'REPAIR: Return only a JSON object with exactly one string field named "answer". '
            "The answer must be one short sentence, no more than 30 words, grounded in the "
            "evidence above, and use only exact citation markers from the allowed labels. If "
            "the evidence cannot answer the question, return "
            '{"answer":"INSUFFICIENT_EVIDENCE"}. '
            "Do not include reasoning or commentary."
        )
        if (
            self._prompts.token_count(repair_prompt)
            > self._prompts.max_prompt_tokens + REPAIR_PROMPT_TOKEN_RESERVE
        ):
            raise ProviderProtocolError("Answer repair prompt exceeds the model context")
        generation = await self._provider.generate(
            GenerationInput(
                prompt=repair_prompt,
                context_window_tokens=self._config.context_window_tokens,
                max_output_tokens=min(self._config.max_output_tokens, REPAIR_MAX_OUTPUT_TOKENS),
                temperature=0,
                think=False,
                response_format="json",
            )
        )
        parsed = parse_answer(generation.response, request, selected)
        if parsed is None:
            raise ServiceError(
                422,
                "INVALID_CITATIONS",
                "The generated answer did not contain valid source citations.",
            )
        return {"response": parsed}

    @staticmethod
    def _insufficient_evidence_node(state: ChatGraphState) -> dict[str, Any]:
        return {"response": insufficient_evidence_response(state["request"])}

    @staticmethod
    def _finalize(state: ChatGraphState) -> dict[str, Any]:
        response = state.get("response")
        if response is None:
            raise RuntimeError("Chat graph reached finalization without a response")
        selected = state.get("selected", [])
        return {
            "result": ChatExecutionResult(
                response=response,
                evidence_ids=frozenset(evidence_key(item) for item in selected),
                retrieved_candidates=tuple(state.get("candidates", [])),
                reranked_candidates=tuple(state.get("reranked_candidates", [])),
            )
        }
