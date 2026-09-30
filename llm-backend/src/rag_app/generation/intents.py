from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


class ChatIntent(StrEnum):
    SMALL_TALK = "small_talk"
    DOCUMENT_CATALOG = "document_catalog"
    DOCUMENT_QUESTION = "document_question"
    DOCUMENT_DEFINITION = "document_definition"
    ANSWER_REVISION = "answer_revision"
    ANSWER_SHORTENING = "answer_shortening"


class ChatPlan(BaseModel):
    """Validated interpretation of a message, produced by the configured LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: ChatIntent
    query: StrictStr = Field(min_length=1, max_length=12_000)
    max_words: StrictInt | None = Field(ge=1, le=10_000)
    definition_subject: StrictStr | None = Field(min_length=1, max_length=200)
    topic_message_index: StrictInt | None = Field(ge=0)
    direct_answer: StrictStr | None = Field(min_length=1, max_length=1000)
    previous_intent: ChatIntent | None = None

    @model_validator(mode="after")
    def validate_direct_answer(self) -> ChatPlan:
        if self.intent is ChatIntent.SMALL_TALK:
            if self.direct_answer is None:
                raise ValueError("Small talk requires a conversational response")
        else:
            # Some models draft a factual answer while planning. Discard it;
            # document answers must be generated from fresh, scoped evidence.
            self.direct_answer = None
        return self
