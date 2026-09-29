from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag_app.config import IndexConfigurationError, Settings, load_index_configuration


def valid_index_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "embeddingModel": "embedding-test:latest",
        "embeddingDigest": "a" * 64,
        "embeddingDimension": 3,
        "chunkTargetTokens": 512,
        "chunkOverlapTokens": 64,
        "chunkerVersion": "chunker-v1",
        "tokenizerVersion": "tokenizer-v1",
        "contextPromptVersion": None,
        "contextModel": None,
        "contextModelDigest": None,
        "contextWindowTokens": 8192,
        "maxOutputTokens": 512,
    }
    config.update(overrides)
    return config


def test_loads_shared_index_config_with_camel_case_fields(tmp_path) -> None:
    config_path = tmp_path / "index.json"
    config_path.write_text(json.dumps(valid_index_config()), encoding="utf-8")

    config = load_index_configuration(config_path)

    assert config.embedding_model == "embedding-test:latest"
    assert config.embedding_dimension == 3
    assert config.chunk_target_tokens == 512


def test_rejects_inconsistent_chunk_overlap(tmp_path) -> None:
    config_path = tmp_path / "index.json"
    config_path.write_text(json.dumps(valid_index_config(chunkOverlapTokens=512)), encoding="utf-8")

    with pytest.raises(IndexConfigurationError):
        load_index_configuration(config_path)


def test_rejects_invalid_digest_and_unknown_config_fields(tmp_path) -> None:
    config_path = tmp_path / "index.json"
    config_path.write_text(
        json.dumps(valid_index_config(embeddingDigest="not-a-digest")), encoding="utf-8"
    )
    with pytest.raises(IndexConfigurationError):
        load_index_configuration(config_path)

    config_path.write_text(json.dumps(valid_index_config(extraField=True)), encoding="utf-8")
    with pytest.raises(IndexConfigurationError):
        load_index_configuration(config_path)


def test_context_settings_must_be_complete(tmp_path) -> None:
    config_path = tmp_path / "index.json"
    config_path.write_text(
        json.dumps(valid_index_config(contextModel="context-test:latest")), encoding="utf-8"
    )

    with pytest.raises(IndexConfigurationError):
        load_index_configuration(config_path)


def test_settings_validate_provider_url_pool_sizes_and_hide_secrets() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_base_url="http://ollama:11434/api")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_pool_min_size=2, database_pool_max_size=1)

    settings = Settings(_env_file=None, internal_api_token="secret-token")
    assert "secret-token" not in repr(settings)
