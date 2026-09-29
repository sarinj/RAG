# LLM backend

FastAPI owns the internal chat contract, Ollama model access, and the ingestion worker. It reads the NestJS-owned PostgreSQL schema and never creates migrations.

## Current implementation slice

The service provides the internal FastAPI contract, model readiness checks, Ollama adapters, a durable baseline ingestion worker, and vector retrieval with cited answer generation. The worker claims queued or expired jobs with leases, parses PDF/TXT/Markdown, chunks with the pinned tokenizer, embeds and stages chunks, then publishes the generation and active pointer in one transaction. It is enabled by default in the root Compose stack.

Chat uses a LangGraph state graph to orchestrate source/config validation, scoped vector retrieval, evidence selection, answer generation, citation validation, and one bounded repair path. Ollama calls, PostgreSQL/pgvector retrieval, and durable application data remain in the existing provider and repository layers; the per-request graph does not duplicate application persistence. Contextual indexing and hybrid BM25 retrieval are not implemented; contextual jobs are failed with a sanitized unsupported-mode error rather than being indexed as baseline by mistake.

## Setup

Install [uv](https://docs.astral.sh/uv/), copy `.env.example` to `.env`, then set `DATABASE_URL`, `INTERNAL_API_TOKEN`, `OLLAMA_BASE_URL`, `GENERATION_MODEL`, `EMBEDDING_MODEL`, `INDEX_CONFIG_PATH`, `TOKENIZER_PATH`, and `UPLOAD_DIR`. The index configuration must be the reviewed artifact shared with NestJS; do not create independent Python defaults.

The tokenizer is a versioned runtime input. For the configured Qwen3 embedding model, fetch the tokenizer JSON from the [Qwen3-Embedding-0.6B model repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):

```sh
curl -L --fail https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/tokenizer.json \
  -o backend/config/tokenizer.json
shasum -a 256 backend/config/tokenizer.json
```

Set `tokenizerVersion` in the shared index config to `sha256:` followed by that digest. The worker checks it at startup and checks the recorded version for every generation. Compose mounts `backend/config/tokenizer.json` read-only at `/config/tokenizer.json`.

The Python bootstrap currently reads this flat JSON shape. It uses the shared generation config key names verbatim; the lead and NestJS owner must use the same artifact shape before integration:

```json
{
  "embeddingModel": "selected-embedding-model",
  "embeddingDigest": "64-character-model-digest",
  "embeddingDimension": 1024,
  "chunkTargetTokens": 512,
  "chunkOverlapTokens": 64,
  "chunkerVersion": "structure-token-offsets-v1",
  "tokenizerVersion": "sha256:<64-character-tokenizer-file-digest>",
  "contextPromptVersion": null,
  "contextModel": null,
  "contextModelDigest": null,
  "contextWindowTokens": 8192,
  "maxOutputTokens": 512
}
```

Embedding model, digest, and dimension must come from a local probe and review. The tokenizer digest must match the exact tokenizer file mounted for the worker.

## Commands

Run from this directory:

```sh
uv sync --locked
uv run uvicorn rag_app.main:app --app-dir src --host 0.0.0.0 --port 8000
uv run python scripts/export_openapi.py
uv run ruff check src scripts
uv run pyright src
uv run pytest
uv run rag-probe --sample-text "A representative sentence in each required language."
uv run rag-worker --check
uv run rag-worker
```

## RAG evaluation

`rag-eval` reads the bundled AstraRail and Helios City suites. It separates each question-and-answer section from the source corpus before ingestion, uploads a temporary evaluation document through the local backend, calls the internal chat endpoint for each case, and requests deletion of the temporary document when finished. AstraRail's expected answers are kept in `evals/datasets/astrarail_answers.json`; Helios answers are parsed from its answer-key section. Existing source documents are left untouched.

Run from the repository root after the Compose stack is ready:

```sh
docker compose build llm-backend
mkdir -p llm-backend/evals/reports
docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --no-sync rag-eval --dataset all
```

The runner writes JSON and Markdown reports under `llm-backend/evals/reports/`. Use `--dataset astrarail` or `--dataset helios_city` to run one suite, `--limit 3` for a short run, and `--keep-documents` to retain the temporary indexed corpus. It reports answer token F1, unanswerable abstention accuracy, citation presence, and latency with model/config and dataset hashes. The supplied suites do not label supporting source spans, so the report cannot claim retrieval recall or factual grounding; review answers and citations before drawing quality conclusions.

To evaluate an already-indexed document instead of uploading a temporary corpus, pass `--source astrarail=<document-uuid>` or `--source helios_city=<document-uuid>`.

`rag-probe` reports exact Ollama model tags/digests, a live embedding dimension, model context metadata when available, and a short generation response. It does not select a tokenizer or write the reviewed index config. Ollama's local model list, generation, embedding, and model-info endpoints are documented in its [API reference](https://docs.ollama.com/api).

Psycopg uses an async connection pool and registers pgvector for each connection. Readiness checks the shared `chunks.embedding` dimension against the config and live embedding model. The adapter uses explicit HTTP timeouts and separate per-process generation and embedding semaphores.

## API contract

`GET /health/live` is process liveness. `GET /health/ready` checks PostgreSQL schema, the reviewed config, local model availability, the embedding digest, and live vector dimension. Both are internal-network endpoints.

`POST /internal/chat` requires `Authorization: Bearer <INTERNAL_API_TOKEN>` and accepts/returns contract version `1.0.0`. The implementation validates request structure, history pairing, unique source documents, active generation scope, citation labels, and response markers.

OpenAPI is exported to `openapi.json` from the FastAPI app. No schema migrations or conversation persistence belong in this service.

## Operations and blockers

- Run API and worker from the same image with separate entrypoints. Mount uploads read-only in the worker and mount the same reviewed config in both services.
- API timeout defaults to 120 seconds, below the NestJS 130-second and browser 150-second deadlines. Rebenchmark before changing these values.
- Probe and record model digests whenever models are upgraded. Changing the embedding model or dimension requires a new full index generation.
- Worker leases last 90 seconds and are renewed every 20 seconds. Expired jobs are retried up to three automatic attempts; a document-level failure leaves an existing active generation in place.
- Upload storage is mounted read-only in the worker. Every write verifies the lease and deletion state; chunk batches reserve the shared 5,000-row capacity before insertion.
- The selected tokenizer must be pinned in `tokenizerVersion`; Ollama's REST API does not expose token counting.
- Contextualization and BM25/hybrid retrieval remain later LLM-04 through LLM-05 work. Use `rag-eval` to record the vector baseline before comparing retrieval changes.
