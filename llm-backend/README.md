# LLM backend

FastAPI owns the internal chat contract, model-provider adapters, and the ingestion worker. It reads the NestJS-owned PostgreSQL schema and never creates migrations.

## Current implementation slice

The service provides the internal FastAPI contract, model-provider adapters, a durable ingestion worker, contextual indexing, hybrid retrieval, and cited answer generation. Chat and ingestion depend on one `LLMProvider` port. The factory composes a generation adapter and an embedding adapter behind that port, so provider-specific HTTP formats stay outside the RAG workflow. The worker claims queued or expired jobs with leases, parses PDF/TXT/Markdown, chunks with the pinned tokenizer, optionally contextualizes each chunk against the full document, embeds and stages searchable text, then publishes the generation and active pointer in one transaction. It is enabled by default in the root Compose stack.

Chat uses one structured LLM call to interpret intent, resolve the search topic from history, and identify the requested answer length. The plan routes social messages to a conversational reply and document-library questions to active source metadata. Document questions and answer revisions use a broad document-QA path that supports explanations, summaries, comparisons, and synthesis across evidence. A LangGraph state graph orchestrates planning, source/config validation, scoped vector or hybrid BM25 plus vector retrieval, reciprocal-rank fusion, LLM reranking, evidence selection, answer generation, citation validation, and one bounded repair path. Hybrid search fetches up to 30 vector matches and 30 lexical matches from the same active source snapshot, fuses them, and reranks the best 30. The prompt receives up to five reranked chunks under the model token budget. Provider HTTP calls, PostgreSQL/pgvector retrieval, and durable application data stay behind adapters and repositories; the per-request graph does not duplicate application persistence.

Reranking uses excerpts around query terms and requests only the five best IDs. The strongest fused retrieval result stays first in the final evidence set so an unreliable model ranking cannot discard it. Follow-ups such as “summarize it shortly” and “too long” keep the previous topic searchable even if query rewriting returns invalid output. Answer generation receives the original user request and the resolved search topic separately, with explicit brevity targets and one retry for an overly long answer. A refusal also gets one evidence-grounded retry; a definition question can recover an unambiguous, verbatim definition from the selected source text with its citation. Conflicting definitions and missing information still require evidence-based generation or an insufficient-evidence response. Conversation history supplies context and never substitutes for current source evidence.

If the shortening retry remains too long, a complete source statement matching the draft's opening can supply a brief, cited answer. This fallback uses the current selected evidence and never cuts an answer in the middle of a sentence. Older conversation pairs are dropped when needed to fit the strongest source and resolved topic into the prompt.

Intent and length decisions use the LLM's validated JSON plan, with no phrase or language regex classification. The same planning call resolves follow-up topics and identifies the current and previous tasks. Failed replies supply no answer context for presentation requests; user questions remain available so an explicitly named new subject is preserved. A revision of an unanswered factual question preserves the lack of evidence instead of reverting to an older topic. New factual questions generate from the resolved query and current sources without repeating old answers in their prompt. Verbatim definition recovery is limited to the LLM's definition intent. When the LLM identifies shortening, the word limit is also capped below the previous supported answer's measured length. Factual drafts returned during planning are discarded; document answers require fresh scoped evidence. Invalid or unresolved plans get one bounded planning retry, then fall back to document QA with bounded recent context. Regex is limited to parsing citation markers, provider wrappers, source matches, and retrieval terms.

## Setup

Install [uv](https://docs.astral.sh/uv/), copy `.env.example` to `.env`, then set `DATABASE_URL`, `INTERNAL_API_TOKEN`, `LLM_PROVIDER`, `GENERATION_MODEL`, `EMBEDDING_MODEL`, `INDEX_CONFIG_PATH`, `TOKENIZER_PATH`, and `UPLOAD_DIR`. Supported generation providers are `ollama`, `openai`, `claude`, `gemini`, and `huggingface`. Embeddings can use `ollama`, `openai`, `gemini`, or `huggingface`. By default, the embedding provider follows `LLM_PROVIDER`; Claude defaults to Ollama because Anthropic does not offer an embedding model. Set `EMBEDDING_PROVIDER` to choose another embedding backend. The index configuration must be the reviewed artifact shared with NestJS; do not create independent Python defaults.

For Compose, use these settings to send both generation and embeddings to OpenAI:

```dotenv
LLM_PROVIDER=openai
LLM_PROVIDER_BASE_URL=
LLM_PROVIDER_API_KEY=<provider-key>
GENERATION_MODEL=gpt-5-nano
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_MODEL_DIGEST=d7facf4722679d2413040d7e0dec9e66ca3b6702d647ff97b5d452be42c37eee
```

To switch generation back to Ollama while retaining OpenAI embeddings, set:

```dotenv
LLM_PROVIDER=ollama
LLM_PROVIDER_BASE_URL=
EMBEDDING_PROVIDER_API_KEY=<your-openai-api-key>
GENERATION_MODEL=qwen3:4b
```

The database vector column is fixed at 1024 dimensions. The OpenAI adapter requests `dimensions=1024` from `text-embedding-3-small`, whose default output is 1536 dimensions. The shared index artifact uses the pinned fingerprint shown above: SHA-256 of `provider=openai;model=text-embedding-3-small;dimensions=1024`. OpenAI lists this embedding model at $0.02 per million input tokens. [`gpt-5-nano`](https://developers.openai.com/api/docs/models/gpt-5-nano) is a low-cost generation model at $0.05 per million input tokens and $0.40 per million output tokens. Both API services are billed by token usage. `openai` uses OpenAI's API by default; it also works with a local server that implements the OpenAI chat-completions, embeddings, and models endpoints.

After changing `.env`, apply the selection with `docker compose up -d --force-recreate llm-backend llm-worker`. Reindex each existing document from the document list before chatting with it; the old Ollama vectors are a different vector space and will not match OpenAI query embeddings. Reindexing calls the paid embeddings API. The OpenAI worker batches up to 32 chunks per request to reduce network overhead. Switching embeddings back to Ollama requires restoring that model's index config and reindexing again.

For Claude, set `LLM_PROVIDER=claude`, provide an Anthropic API key, and use a Claude model ID. Since Claude supplies generation only, embeddings default to Ollama. For example, set `EMBEDDING_PROVIDER=gemini` plus `EMBEDDING_PROVIDER_API_KEY` to use Gemini embeddings with Claude generation.

For Gemini, set `LLM_PROVIDER=gemini`, provide a Gemini API key, and set `GENERATION_MODEL` and `EMBEDDING_MODEL` to supported models. For Hugging Face Inference Providers, set `LLM_PROVIDER=huggingface`, use an access token with Inference Providers permission, and configure model IDs for chat and feature extraction. `LLM_PROVIDER_BASE_URL` defaults to `https://router.huggingface.co`.

When `EMBEDDING_PROVIDER` matches `LLM_PROVIDER`, the generation API key and base URL are reused unless the embedding-specific values are set. When they differ, use `EMBEDDING_PROVIDER_BASE_URL` and `EMBEDDING_PROVIDER_API_KEY` to configure the embedding connection separately. Hosted embedding adapters cannot report a stable weight digest, so `EMBEDDING_MODEL_DIGEST` must be an operator-pinned fingerprint that matches `embeddingDigest` in the shared index config. Keep the selected embedding model, fingerprint, and dimension aligned with the existing index.

The tokenizer is a versioned runtime input used for chunking and token budgets. This workspace keeps its existing Qwen3 tokenizer while sending embeddings to OpenAI. Fetch the tokenizer JSON from the [Qwen3-Embedding-0.6B model repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):

```sh
curl -L --fail https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/tokenizer.json \
  -o backend/config/tokenizer.json
shasum -a 256 backend/config/tokenizer.json
```

Set `tokenizerVersion` in the shared index config to `sha256:` followed by that digest. The worker checks it at startup and checks the recorded version for every generation. Compose mounts `backend/config/tokenizer.json` read-only at `/config/tokenizer.json`.

The Python bootstrap reads this flat JSON shape. For baseline indexing, leave the three context fields null. To enable full-document contextual indexing, set `INDEX_MODE=contextual` and provide `contextPromptVersion: "full-document-chunk-context-v1"`, the exact `contextModel` selected in `GENERATION_MODEL`, and an operator-pinned 64-character `contextModelDigest`. The worker rejects mismatched models/configuration and documents that exceed the complete-document context budget; it does not silently truncate the document. Reindex existing documents after changing the indexing mode or context settings.

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
  "maxOutputTokens": 512,
  "retrievalMode": "hybrid"
}
```

Embedding model, digest, and dimension must come from a probe and review. Ollama reports its embedding model digest. Hosted embedding APIs do not expose a weights digest, so set `EMBEDDING_MODEL_DIGEST` to an operator-pinned fingerprint and keep it equal to `embeddingDigest` in the shared config. The tokenizer digest must match the exact tokenizer file mounted for the worker.

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

`rag-eval` reads the standalone Harborlight pipeline benchmark in `evals/datasets/pipeline_v2.json` and its separate operations corpus. The benchmark includes exact-term, paraphrase, multi-hop, version-conflict, follow-up, and unanswerable cases; answerable cases carry source excerpts that are mapped to corpus offsets before the run. It finds reusable documents by stored SHA-256, verifies the selected document's content hash, and uploads a temporary source through the local backend when needed. It requests deletion only for documents created by that run (unless `--keep-documents` is set). Existing or explicitly selected documents are left untouched.

Run from the repository root after the Compose stack is ready:

```sh
docker compose build llm-backend
mkdir -p llm-backend/evals/reports
docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --no-sync rag-eval --dataset all
```

The runner writes JSON and Markdown reports under `llm-backend/evals/reports/`. Run `--dataset pipeline_v2` for the benchmark, `--limit 3` for a short run, and `--keep-documents` to retain a corpus uploaded during that run. Reports include answer token F1, abstention accuracy, citation presence, latency, candidate Recall@30, reranked Recall@5, and citation-evidence precision/recall. They record selected provider/model IDs, shared index settings, and dataset hashes. These are overlap and lexical metrics, not proof of semantic correctness; review answers and citations before drawing quality conclusions.

To evaluate an already-indexed benchmark corpus instead of uploading a temporary corpus, pass `--source pipeline_v2=<document-uuid>`.

`rag-probe` exercises the selected generation and embedding adapters with sample requests. For Ollama it also reports model digests and context metadata; hosted embedding providers report the configured embedding fingerprint. It does not select a tokenizer or write the reviewed index config. Ollama's local model list, generation, embedding, and model-info endpoints are documented in its [API reference](https://docs.ollama.com/api).

Psycopg uses an async connection pool and registers pgvector for each connection. Readiness checks the shared `chunks.embedding` dimension against the config and live embedding model. The adapter uses explicit HTTP timeouts and separate per-process generation and embedding semaphores.

## API contract

`GET /health/live` is process liveness. `GET /health/ready` checks PostgreSQL schema, the reviewed config, selected provider model availability, the embedding fingerprint, and live vector dimension. Both are internal-network endpoints.

`POST /internal/chat` requires `Authorization: Bearer <INTERNAL_API_TOKEN>` and accepts/returns contract version `1.1.0`. The implementation validates request structure, history pairing, unique source documents, active generation scope, citation labels, and response markers. `direct_answer` is reserved for greeting and document-catalog responses that need no chunk citations. All other questions remain eligible for document-grounded QA. `rag-eval` uses the equally authenticated `POST /internal/evaluation/chat`, which runs the same chat graph and additionally returns candidate chunk IDs and source offsets for labeled retrieval scoring.

OpenAPI is exported to `openapi.json` from the FastAPI app. No schema migrations or conversation persistence belong in this service.

## Operations and blockers

- Run API and worker from the same image with separate entrypoints. Mount uploads read-only in the worker and mount the same reviewed config in both services.
- API timeout defaults to 120 seconds, below the NestJS 130-second and browser 150-second deadlines. Rebenchmark before changing these values.
- Probe and record the embedding fingerprint whenever models are upgraded. Changing the embedding model or dimension requires a new full index generation.
- Worker leases last 90 seconds and are renewed every 20 seconds. Expired jobs are retried up to three automatic attempts; a document-level failure leaves an existing active generation in place.
- Upload storage is mounted read-only in the worker. Every write verifies the lease and deletion state; chunk batches reserve the shared 5,000-row capacity before insertion.
- The selected tokenizer must be pinned in `tokenizerVersion`; Ollama's REST API does not expose token counting.
- Contextual indexing requires a configured prompt version, model ID, model digest, and `INDEX_MODE=contextual`; baseline mode continues to embed original chunk text. Set `retrievalMode` to `vector` in the shared index configuration only when intentionally comparing the vector baseline. The default is scoped BM25 plus vector search, fusion, and reranking.
