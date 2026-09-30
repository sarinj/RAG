from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
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
ProviderName = Literal["ollama", "openai", "claude", "gemini", "huggingface"]
EmbeddingProviderName = Literal["ollama", "openai", "gemini", "huggingface"]


@dataclass(frozen=True)
class ProviderConnection:
    provider: ProviderName
    base_url: str
    api_key: str | None = field(repr=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: SecretStr | None = None
    internal_api_token: SecretStr | None = None
    llm_provider: ProviderName = "ollama"
    llm_provider_base_url: str | None = None
    llm_provider_api_key: SecretStr | None = None
    embedding_provider: EmbeddingProviderName | None = None
    embedding_provider_base_url: str | None = None
    embedding_provider_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    generation_model: NonEmptyString | None = None
    embedding_model: NonEmptyString | None = None
    embedding_model_digest: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    index_config_path: Path | None = None
    tokenizer_path: Path | None = None
    upload_dir: Path | None = None
    max_upload_bytes: int = Field(default=52_428_800, gt=0)
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)

    database_pool_min_size: int = Field(default=1, ge=0)
    database_pool_max_size: int = Field(default=5, ge=1)
    database_pool_timeout_seconds: float = Field(default=5.0, gt=0)
    provider_connect_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias=AliasChoices(
            "PROVIDER_CONNECT_TIMEOUT_SECONDS", "OLLAMA_CONNECT_TIMEOUT_SECONDS"
        ),
    )
    provider_read_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        validation_alias=AliasChoices(
            "PROVIDER_READ_TIMEOUT_SECONDS", "OLLAMA_READ_TIMEOUT_SECONDS"
        ),
    )
    provider_write_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        validation_alias=AliasChoices(
            "PROVIDER_WRITE_TIMEOUT_SECONDS", "OLLAMA_WRITE_TIMEOUT_SECONDS"
        ),
    )
    provider_pool_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias=AliasChoices(
            "PROVIDER_POOL_TIMEOUT_SECONDS", "OLLAMA_POOL_TIMEOUT_SECONDS"
        ),
    )
    provider_generation_concurrency: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices(
            "PROVIDER_GENERATION_CONCURRENCY", "OLLAMA_GENERATION_CONCURRENCY"
        ),
    )
    provider_embedding_concurrency: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices(
            "PROVIDER_EMBEDDING_CONCURRENCY", "OLLAMA_EMBEDDING_CONCURRENCY"
        ),
    )
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

    @property
    def active_embedding_provider(self) -> EmbeddingProviderName:
        if self.embedding_provider:
            return self.embedding_provider
        if self.llm_provider == "claude":
            return "ollama"
        return self.llm_provider

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def normalize_embedding_provider(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def generation_connection(self) -> ProviderConnection:
        return ProviderConnection(
            provider=self.llm_provider,
            base_url=(
                self.llm_provider_base_url
                or self.default_provider_base_url(self.llm_provider)
            ),
            api_key=self.secret_value(self.llm_provider_api_key),
        )

    @property
    def embedding_connection(self) -> ProviderConnection:
        provider = self.active_embedding_provider
        same_provider = provider == self.llm_provider
        if self.embedding_provider_base_url:
            base_url = self.embedding_provider_base_url
        elif same_provider and self.llm_provider_base_url:
            base_url = self.llm_provider_base_url
        else:
            base_url = self.default_provider_base_url(provider)

        api_key = self.secret_value(self.embedding_provider_api_key)
        if api_key is None and same_provider:
            api_key = self.secret_value(self.llm_provider_api_key)
        return ProviderConnection(provider=provider, base_url=base_url, api_key=api_key)

    @staticmethod
    def secret_value(value: SecretStr | None) -> str | None:
        if value is None:
            return None
        return value.get_secret_value() or None

    def default_provider_base_url(self, provider: ProviderName) -> str:
        if provider == "ollama":
            return self.ollama_base_url
        if provider == "openai":
            return "https://api.openai.com/v1"
        if provider == "claude":
            return "https://api.anthropic.com/v1"
        if provider == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta"
        return "https://router.huggingface.co"

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

    @field_validator("llm_provider_base_url", "embedding_provider_base_url")
    @classmethod
    def validate_provider_base_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip()
        parts = urlsplit(normalized)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Provider base URL must be an HTTP(S) URL with a host")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Provider base URL cannot include credentials, query, or fragment")
        return normalized.rstrip("/")

    @field_validator("embedding_model_digest", mode="before")
    @classmethod
    def normalize_embedding_model_digest(cls, value: object) -> object:
        return None if value == "" else value


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
