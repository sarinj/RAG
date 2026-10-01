from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator


class GenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: StrictStr = Field(min_length=1)
    context_window_tokens: StrictInt | None = Field(default=None, gt=0)
    max_output_tokens: StrictInt = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    think: StrictBool | None = None
    response_format: Literal["json"] | None = None
    response_schema: dict[str, Any] | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    reasoning_token_budget: StrictInt = Field(default=0, ge=0, le=4096)


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    response: str
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    reasoning_tokens: int | None = None


class EmbeddingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[StrictStr] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_texts(self) -> EmbeddingInput:
        if any(not text.strip() for text in self.texts):
            raise ValueError("embedding texts cannot be empty")
        return self


class EmbeddingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    vectors: list[list[float]] = Field(min_length=1)
    dimension: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_vectors(self) -> EmbeddingOutput:
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("embedding vectors have inconsistent dimensions")
        if any(not math.isfinite(value) for vector in self.vectors for value in vector):
            raise ValueError("embedding vectors must contain finite values")
        return self
