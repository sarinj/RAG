"""Request-scoped public progress and draft text; never expose model reasoning."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import UUID

from pydantic_core import from_json

from rag_app.errors import ProviderProtocolError
from rag_app.providers.models import GenerationInput, GenerationOutput
from rag_app.providers.protocols import LLMProvider, StreamingGenerationProvider

StreamSink = Callable[[dict[str, Any]], Awaitable[None]]

# These are descriptions of observable work, not a model's private deliberation.
STAGES: dict[str, tuple[str, str]] = {
    "planner": ("Understanding your question", "Question understood"),
    "retrieval": ("Finding relevant passages", "Source search complete"),
    "multi_hop": ("Connecting evidence across sources", "Evidence search complete"),
    "corpus_processing": ("Reading the selected documents", "Document review complete"),
    "reranking": ("Comparing relevant passages", "Relevant passages ranked"),
    "evidence_selection": ("Choosing sources for the answer", "Sources selected"),
    "evidence_sufficiency": (
        "Checking whether the sources answer your question",
        "Evidence checked",
    ),
    "generation": ("Writing the answer", "Draft prepared"),
    "claim_validation": ("Checking the answer against its sources", "Source checks complete"),
    "answer_review": ("Checking completeness and relevance", "Answer review complete"),
    "repair": ("Refining the answer", "Answer refined"),
    "finalization": ("Finalizing the answer and citations", "Answer checks complete"),
}


@dataclass(frozen=True)
class ChatStream:
    request_id: UUID
    sink: StreamSink

    async def emit(self, event_type: str, **data: Any) -> None:
        await self.sink({"type": event_type, "requestId": str(self.request_id), **data})


active_stream: ContextVar[ChatStream | None] = ContextVar("chat_stream", default=None)


async def progress(
    stage: str,
    status: Literal["active", "complete", "skipped"],
    message: str | None = None,
) -> None:
    stream = active_stream.get()
    if stream is not None and stage in STAGES:
        label = message or STAGES[stage][0 if status == "active" else 1]
        await stream.emit("progress", stage=stage, status=status, message=label)


class AnswerPreview:
    """Decode only public answer fields, including incomplete JSON strings."""

    def __init__(self) -> None:
        self._raw = ""
        self._text = ""

    def append(self, fragment: str) -> str:
        self._raw += fragment
        if len(self._raw) > 262_144:
            raise ProviderProtocolError("The streamed answer exceeds the text limit")
        try:
            value = from_json(self._raw, allow_partial="trailing-strings", cache_strings=False)
        except ValueError:
            return ""
        if not isinstance(value, dict):
            return ""
        value = cast(dict[str, Any], value)
        claims = value.get("claims")
        if isinstance(claims, list):
            texts: list[str] = []
            for claim in cast(list[Any], claims):
                if isinstance(claim, dict):
                    claim_text = cast(dict[str, Any], claim).get("text")
                    if isinstance(claim_text, str):
                        texts.append(claim_text)
            text = "\n\n".join(texts)
        elif isinstance(value.get("answer"), str):
            text = value["answer"]
        else:
            return ""
        # Incomplete escape sequences can make a partial parse temporarily regress.
        # Emit only a monotone prefix; complete validation still uses the entire JSON.
        if not text.startswith(self._text):
            return ""
        delta = text[len(self._text) :]
        self._text = text
        return delta


async def generate_answer(provider: LLMProvider, request: GenerationInput) -> GenerationOutput:
    stream = active_stream.get()
    method = getattr(provider, "generate_stream", None)
    if stream is None or not callable(method):
        return await provider.generate(request)
    await stream.emit("answer_reset")
    preview = AnswerPreview()

    async def on_text(fragment: str) -> None:
        delta = preview.append(fragment)
        # Keep even escaped supplementary Unicode well inside the SSE frame limit.
        for offset in range(0, len(delta), 8_192):
            await stream.emit("answer_delta", delta=delta[offset : offset + 8_192])

    return await cast(StreamingGenerationProvider, provider).generate_stream(request, on_text)
