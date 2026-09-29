from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import httpx

from rag_app.api.chat import ChatExecutionResult
from rag_app.config import Settings
from rag_app.main import create_app
from rag_app.models import ChatRequest, ChatResponse

REQUEST_ID = "44444444-4444-4444-8444-444444444444"
DOCUMENT_ID = "11111111-1111-4111-8111-111111111111"
GENERATION_ID = "22222222-2222-4222-8222-222222222222"


class FakeDatabase:
    def __init__(self, active: bool = True) -> None:
        self.active = active

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def validate_sources(self, sources: list[tuple[UUID, UUID]]) -> bool:
        return self.active and len(sources) == 1


def request_body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "contractVersion": "1.0.0",
        "requestId": REQUEST_ID,
        "question": "What does the source say?",
        "history": [],
        "sources": [{"documentId": DOCUMENT_ID, "generationId": GENERATION_ID}],
        "retrievalMode": "vector",
    }
    payload.update(overrides)
    return payload


@asynccontextmanager
async def client_for(
    database: FakeDatabase,
    chat_executor: object | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        Settings(
            _env_file=None,
            database_url=None,
            internal_api_token="internal-test-token",
            ollama_base_url="http://localhost:11434",
            generation_model=None,
            embedding_model=None,
            index_config_path=None,
            tokenizer_path=None,
        ),
        database=database,  # type: ignore[arg-type]
        provider=object(),  # type: ignore[arg-type]
        chat_executor=chat_executor,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client


async def test_chat_rejects_missing_internal_token_with_contract_error() -> None:
    async with client_for(FakeDatabase()) as client:
        response = await client.post(
            "/internal/chat",
            json=request_body(),
            headers={"X-Request-Id": REQUEST_ID},
        )

    assert response.status_code == 401
    assert response.json() == {
        "code": "UNAUTHORIZED",
        "message": "A valid internal bearer token is required",
        "requestId": REQUEST_ID,
    }
    assert response.headers["X-Request-Id"] == REQUEST_ID


async def test_chat_rejects_contract_version_and_changed_source_scope() -> None:
    async with client_for(FakeDatabase()) as client:
        version_response = await client.post(
            "/internal/chat",
            json=request_body(contractVersion="2.0.0"),
            headers={"Authorization": "Bearer internal-test-token"},
        )
    assert version_response.status_code == 400
    assert version_response.json()["code"] == "CONTRACT_VERSION_MISMATCH"

    async with client_for(FakeDatabase(active=False)) as client:
        scope_response = await client.post(
            "/internal/chat",
            json=request_body(),
            headers={"Authorization": "Bearer internal-test-token"},
        )
    assert scope_response.status_code == 409
    assert scope_response.json()["code"] == "SCOPE_CHANGED"


async def test_chat_reports_not_ready_when_execution_pipeline_is_missing() -> None:
    async with client_for(FakeDatabase()) as client:
        response = await client.post(
            "/internal/chat",
            json=request_body(),
            headers={"Authorization": "Bearer internal-test-token"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "SERVICE_NOT_READY"


async def test_lifespan_wires_chat_executor_from_shared_runtime_config(
    tmp_path, monkeypatch
) -> None:
    from rag_app import main

    index_config = tmp_path / "index.json"
    index_config.write_text(
        """{
          "embeddingModel": "embedding-test:latest",
          "embeddingDigest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "embeddingDimension": 3,
          "chunkTargetTokens": 512,
          "chunkOverlapTokens": 64,
          "chunkerVersion": "chunker-v1",
          "tokenizerVersion": "tokenizer-v1",
          "contextPromptVersion": null,
          "contextModel": null,
          "contextModelDigest": null,
          "contextWindowTokens": 8192,
          "maxOutputTokens": 512
        }""",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("placeholder", encoding="utf-8")
    tokenizer = object()
    executor = object()
    monkeypatch.setattr(main, "load_tokenizer", lambda _settings, _config: tokenizer)
    monkeypatch.setattr(main, "BaselineChatExecutor", lambda *_args: executor)
    settings = Settings(
        _env_file=None,
        internal_api_token="internal-test-token",
        generation_model="generation-test:latest",
        embedding_model="embedding-test:latest",
        index_config_path=index_config,
        tokenizer_path=tokenizer_path,
    )
    database = FakeDatabase()
    provider = object()
    app = create_app(settings, database=database, provider=provider)  # type: ignore[arg-type]

    async with app.router.lifespan_context(app):
        assert app.state.chat_executor is executor


async def test_chat_returns_citations_from_retrieved_source_chunks() -> None:
    chunk_id = "33333333-3333-4333-8333-333333333333"
    chat_response = ChatResponse.model_validate(
        {
            "contractVersion": "1.0.0",
            "requestId": REQUEST_ID,
            "answer": "The source says this [S1].",
            "answerStatus": "answered",
            "citations": [
                {
                    "label": "S1",
                    "documentId": DOCUMENT_ID,
                    "generationId": GENERATION_ID,
                    "chunkId": chunk_id,
                    "documentName": "report.md",
                    "pageStart": None,
                    "pageEnd": None,
                    "heading": "Summary",
                }
            ],
        }
    )

    class FakeExecutor:
        async def answer(self, _request: ChatRequest) -> ChatExecutionResult:
            return ChatExecutionResult(
                response=chat_response,
                evidence_ids=frozenset(
                    {
                        (
                            UUID(DOCUMENT_ID),
                            UUID(GENERATION_ID),
                            UUID(chunk_id),
                        )
                    }
                ),
            )

    async with client_for(FakeDatabase(), chat_executor=FakeExecutor()) as client:
        response = await client.post(
            "/internal/chat",
            json=request_body(),
            headers={"Authorization": "Bearer internal-test-token"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "The source says this [S1]."


async def test_chat_rejects_citations_outside_retrieved_evidence() -> None:
    chat_response = ChatResponse.model_validate(
        {
            "contractVersion": "1.0.0",
            "requestId": REQUEST_ID,
            "answer": "The source says this [S1].",
            "answerStatus": "answered",
            "citations": [
                {
                    "label": "S1",
                    "documentId": DOCUMENT_ID,
                    "generationId": GENERATION_ID,
                    "chunkId": "33333333-3333-4333-8333-333333333333",
                    "documentName": "report.md",
                    "pageStart": None,
                    "pageEnd": None,
                    "heading": None,
                }
            ],
        }
    )

    class FakeExecutor:
        async def answer(self, _request: ChatRequest) -> ChatExecutionResult:
            return ChatExecutionResult(response=chat_response, evidence_ids=frozenset())

    async with client_for(FakeDatabase(), chat_executor=FakeExecutor()) as client:
        response = await client.post(
            "/internal/chat",
            json=request_body(),
            headers={"Authorization": "Bearer internal-test-token"},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_CITATIONS"


async def test_health_is_live_while_readiness_reports_missing_dependencies() -> None:
    async with client_for(FakeDatabase()) as client:
        live_response = await client.get("/health/live")
        ready_response = await client.get("/health/ready", headers={"X-Request-Id": REQUEST_ID})

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 503
    assert ready_response.json()["code"] == "SERVICE_NOT_READY"
    assert ready_response.json()["requestId"] == REQUEST_ID


def test_openapi_advertises_versioned_internal_contract() -> None:
    app = create_app(Settings(_env_file=None))

    schema = app.openapi()

    assert schema["info"]["version"] == "1.0.0"
    assert "/internal/chat" in schema["paths"]
    assert "/health/ready" in schema["paths"]
    assert schema["components"]["securitySchemes"]["InternalBearer"]["type"] == "http"
