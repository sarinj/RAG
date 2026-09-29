from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: SecretStr | None = None
    internal_api_token: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    generation_model: NonEmptyString | None = None
    embedding_model: NonEmptyString | None = None
    index_config_path: Path | None = None
    tokenizer_path: Path | None = None
    upload_dir: Path | None = None
    max_upload_bytes: int = Field(default=52_428_800, gt=0)
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)

    database_pool_min_size: int = Field(default=1, ge=0)
    database_pool_max_size: int = Field(default=5, ge=1)
    database_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    ollama_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    ollama_read_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_write_timeout_seconds: float = Field(default=20.0, gt=0)
    ollama_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    ollama_generation_concurrency: int = Field(default=1, ge=1)
    ollama_embedding_concurrency: int = Field(default=2, ge=1)
    generation_timeout_seconds: float = Field(default=120.0, gt=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    readiness_cache_seconds: float = Field(default=15.0, ge=0)

    @model_validator(mode="after")
    def validate_runtime_limits(self) -> Settings:
        if self.database_pool_min_size > self.database_pool_max_size:
            raise ValueError("database pool minimum cannot exceed its maximum")
        return self

    @property
    def database_dsn(self) -> str | None:
        return self.database_url.get_secret_value() if self.database_url else None

    @property
    def internal_token(self) -> str | None:
        return self.internal_api_token.get_secret_value() if self.internal_api_token else None

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) URL with a host")
        if (
            parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            raise ValueError("OLLAMA_BASE_URL cannot include credentials, path, query, or fragment")
        return value.rstrip("/")


class IndexConfiguration(BaseModel):
    """Reviewed shared index settings plus the exact generation config stored per index."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )

    embedding_model: NonEmptyString
    embedding_digest: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    embedding_dimension: StrictInt = Field(gt=0)

    chunk_target_tokens: StrictInt = Field(gt=0)
    chunk_overlap_tokens: StrictInt = Field(ge=0)
    chunker_version: NonEmptyString
    tokenizer_version: NonEmptyString
    context_prompt_version: NonEmptyString | None
    context_model: NonEmptyString | None
    context_model_digest: str | None = Field(pattern=r"^[a-fA-F0-9]{64}$")
    context_window_tokens: StrictInt = Field(gt=0)
    max_output_tokens: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_consistency(self) -> IndexConfiguration:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk overlap must be smaller than the chunk target")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("maximum output tokens must be smaller than the context window")
        contextual_values = (
            self.context_prompt_version,
            self.context_model,
            self.context_model_digest,
        )
        if any(value is not None for value in contextual_values) and not all(
            value is not None for value in contextual_values
        ):
            raise ValueError("context prompt, model, and digest must be set together")
        return self


class IndexConfigurationError(ValueError):
    """An index config is absent, invalid, or inconsistent with the runtime environment."""


def load_index_configuration(path: Path | None) -> IndexConfiguration:
    if path is None:
        raise IndexConfigurationError("Index configuration is not configured")
    try:
        if path.stat().st_size > 64 * 1024:
            raise IndexConfigurationError("Index configuration is too large")
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = IndexConfiguration.model_validate(raw)
        return config
    except IndexConfigurationError:
        raise
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IndexConfigurationError("Index configuration is unavailable or invalid") from exc


RetrievalMode = Literal["vector", "hybrid"]
IndexMode = Literal["baseline", "contextual"]
