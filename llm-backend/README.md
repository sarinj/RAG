# LLM backend

FastAPI owns the internal chat contract, model-provider adapters, and the ingestion worker. It reads the NestJS-owned PostgreSQL schema and never creates migrations.

## Current implementation slice

The service provides the internal FastAPI contract, model-provider adapters, a durable ingestion worker, contextual indexing, hybrid retrieval, and cited answer generation. Chat and ingestion depend on one `LLMProvider` port. The factory composes a generation adapter and an embedding adapter behind that port, so provider-specific HTTP formats stay outside the RAG workflow. The worker claims queued or expired jobs with leases, parses PDF/TXT/Markdown, chunks with the pinned tokenizer, optionally contextualizes each chunk against the full document, embeds and stages searchable text, then publishes the generation and active pointer in one transaction. It is enabled by default in the root Compose stack.

Chat uses one structured LLM call to interpret intent, resolve the topic from history, choose document scope and retrieval strategy, and identify answer preferences. `ChatIntent` expresses the task: `conversation`, `source_catalog`, `source_question`, or `answer_revision`. Definitions and shortening are preferences (`definition_subject`, `shorten`, and `max_words`), so a brief whole-document summary remains a source question with a global retrieval route. The planner receives accessible document names and IDs; it may narrow the server-authorized scope but cannot expand it.

The LangGraph routes document requests before normal retrieval:

| Route | Evidence strategy |
| --- | --- |
| `LOCAL_SEARCH` | Standalone query and optional independent subquestions/variants, scoped vector or hybrid retrieval, scored fusion, reranking, optional neighbors, and original evidence under count and token budgets (five chunks by default). |
| `GLOBAL_SUMMARY` | Read every scoped chunk in document/section order, generate cited section notes, and reduce them within documents and across documents when needed to fit the final prompt. |
| `AGGREGATION_LIST` | Scan every scoped chunk for matching entities, records, events, or numerical facts; merge exact normalized duplicates while retaining their original source IDs; generate the list/count/aggregation from the complete extracted facts. |
| `MULTI_HOP` | Perform at most three scoped retrieval rounds. Each later search can use names and relationships found in earlier evidence. Retain the strongest evidence from every successful round. |

Summaries and extraction run on demand against the selected active generations; no migration or reindex is needed. They bypass top-K retrieval and the five-chunk selection limit. Oversized passages are split with overlap, and incomplete or output-truncated extraction is retried on smaller inputs. Every map/reduce result validates its original source references. Derived notes remain separate from original chunk text, and final citations still resolve to original chunks. Lists are never compressed by the summary reducer; results exceeding the prompt or generation output limit return `CONTEXT_LIMIT_EXCEEDED` rather than silently sampling the corpus. The complete-corpus read also rejects overflow beyond the shared 5,000-chunk cap.

All routes retain source/config validation, evidence-based generation, citation validation, and one bounded answer repair. Local hybrid search fetches up to 30 vector matches and 30 lexical matches from the same active snapshot, fuses them, and reranks the best 30. Provider HTTP calls, PostgreSQL/pgvector retrieval, and durable application data stay behind adapters and repositories; the per-request graph does not duplicate application persistence.

Reranking uses excerpts around query terms and requests only the five best IDs. The strongest fused retrieval result stays first in the final evidence set so an unreliable model ranking cannot discard it. Follow-ups such as “summarize it shortly” and “too long” keep the previous topic searchable even if query rewriting returns invalid output. Answer generation receives the original user request and the resolved search topic separately, with explicit brevity targets and one retry for an overly long answer. A refusal also gets one evidence-grounded retry; a definition question can recover an unambiguous, verbatim definition from the selected source text with its citation. Conflicting definitions and missing information still require evidence-based generation or an insufficient-evidence response. Conversation history supplies context and never substitutes for current source evidence.

If the shortening retry remains too long, a complete source statement matching the draft's opening can supply a brief, cited answer. This fallback uses the current selected evidence and never cuts an answer in the middle of a sentence. Older conversation pairs are dropped when needed to fit the strongest source and resolved topic into the prompt.

Intent, ambiguous route, scope, and length decisions use the LLM's validated JSON plan. High-confidence multilingual exhaustive-request signals enforce complete corpus routes. The same planning call resolves follow-up topics, independent subquestions and optional query variants. Revisions retain the underlying retrieval strategy: shortening a focused answer uses local search, while shortening a document summary still processes the complete document. Failed replies supply no answer context for presentation requests; user questions remain available so an explicitly named new subject is preserved. A revision of an unanswered factual question preserves the lack of evidence instead of reverting to an older topic. New factual questions generate from the resolved query and current sources without repeating old answers in their prompt. Verbatim definition recovery requires the planner's definition-subject preference and local evidence. When the LLM identifies shortening, the word limit is also capped below the previous supported answer's measured length. Factual drafts returned during planning are discarded; document answers require fresh scoped evidence. Invalid or unresolved plans get one bounded planning retry, then fall back to document QA with bounded recent context and deterministic exhaustive routing.

Evidence coverage is checked before generation. Complete evidence supports a normal answer; partial evidence supports a cited answer with a reconciled missing-aspect note; insufficient evidence stops generation. By default, generated claims name short selected-source labels, which application code resolves to original chunk UUIDs before scope, support and citation validation. Legacy UUID drafts remain readable. One answer repair remains available. Structured `rag_trace` JSON events record scope, stage candidates/scores/ranks, fallback policies and timings, with text omitted by default. Query variants and neighbors are optional and off by default; retrieval depths retain the existing 30/30/30/5 baseline. See [RAG correctness and debugging](../docs/implementation/rag-correctness.md) for every setting, architecture/risk findings, evaluation commands and the debugging decision tree.

The sufficiency provider returns a coverage enum and missing aspects; application code derives
the `answerable` boolean. Sufficiency and claim checks receive explicit JSON schemas. Invalid
outputs remain rejected, and traces include schema error types/locations without source text.

Optional answer review checks completeness and relevance after claim support. The writer,
support checker and reviewer receive the selected original sources. Missing-fact feedback must
quote those originals exactly, allowing whitespace normalization, and name a selected source
label. Irrelevant-claim feedback must name valid claim IDs. Feedback can guide the existing
single repair; repaired answers still need valid citations and source support. An invalid,
unavailable or timed-out review keeps the supported draft. A repair with more reviewed issues,
or without a valid follow-up review, also retains that draft. Each review has a 20-second limit.

`RAG_ANSWER_REVIEW_ENABLED` defaults to `false`. The review's default completion budget is
3072 tokens, including hidden reasoning, with `RAG_ANSWER_REVIEW_REASONING_EFFORT=low`.
That budget is reserved in the context window when review is enabled. `RAG_QUALITY_MODEL`
optionally selects a model on the same provider connection for evidence assessment, grounded
answers, claim validation, review and repair. Planning, retrieval ranking and ingestion use
`GENERATION_MODEL`. Leave the quality model blank to share that model. Readiness checks both
configured models; traces record the actual model, and live reports include `qualityModel`.

The evaluated OpenAI quality preset is:

```dotenv
GENERATION_MODEL=gpt-5-nano
RAG_QUALITY_MODEL=gpt-5
RAG_ANSWER_REVIEW_ENABLED=true
RAG_ANSWER_REASONING_TOKEN_BUDGET=2048
```

This preset increases API cost and response latency. It uses the existing embeddings and
indexes. See the [answer-quality evaluation](../docs/implementation/rag-answer-quality-2026-10-02.md)
for the measured comparison and limitations. OpenAI GPT-5 answer and repair calls request low
verbosity. Answers retain material conditions and exceptions, use short sentences naming the
subject, and show a brief derivation for calculations; explicit requests for detail still apply.

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

Reports distinguish request/metric errors from false abstentions on answerable questions.
An insufficient-evidence reply gets zero answer token F1; boilerplate keyword overlap does
not count as an answer. Answerable answer rate measures response coverage, not semantic accuracy.

To evaluate an already-indexed benchmark corpus instead of uploading a temporary corpus, pass `--source pipeline_v2=<document-uuid>`.

### Optional RAGAS evaluation

RAGAS scores frozen answers against the **exact selected original chunks**. It supplements
the deterministic stage metrics; it does not change retrieval or production chat. Its
dependencies belong to the optional `eval` group and are excluded from the production image.

Keep a live benchmark's documents with `rag-eval --dataset pipeline_v2 --keep-documents`,
then run from the repository root:

```bash
docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --locked --no-default-groups --group eval rag-eval \
  --ragas-report /app/evals/reports/rag-eval-YYYYMMDDTHHMMSSZ.json \
  --ragas-judge-model gpt-5-mini
```

The defaults are [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/),
[answer accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/),
[answer relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/),
[context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
and [context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/).
Choose a subset with `--ragas-metrics`. Optional `factual_correctness` is also available,
but short reference fragments can confuse that metric because its comparison omits the
question. An independent judge is recommended; `--ragas-judge-model` does not change the
application model. This adapter requires configured OpenAI generation credentials and,
for answer relevancy, OpenAI embedding credentials.

Calls use `--ragas-max-tokens 2048`, `--ragas-concurrency 2` and one provider retry.
Use `--limit` for a small trial. Failed metrics are reported and make the command exit
nonzero. False abstentions get zero answer accuracy/relevancy and no faithfulness score;
faithfulness is conditional on returned answers. Unanswerable cases report abstention
separately instead of rewarding an empty response with a perfect score.

The command exports `ragas-*-inputs.jsonl`, which deliberately contains selected source
text and frozen answers. These files stay in the ignored reports directory. Re-score
with `--ragas-inputs <inputs.jsonl>` without needing the original database generation.
For older reports lacking selected stage IDs, supply `--ragas-trace-file` from exactly
that run. Never substitute retrieved or reranked candidates for the contexts actually
given to the answer model. LLM judges can make mistakes; review per-case results alongside
answers, citations and deterministic retrieval metrics.

`rag-probe` exercises the selected generation and embedding adapters with sample requests. For Ollama it also reports model digests and context metadata; hosted embedding providers report the configured embedding fingerprint. It does not select a tokenizer or write the reviewed index config. Ollama's local model list, generation, embedding, and model-info endpoints are documented in its [API reference](https://docs.ollama.com/api).

Psycopg uses an async connection pool and registers pgvector for each connection. Readiness checks the shared `chunks.embedding` dimension against the config and live embedding model. The adapter uses explicit HTTP timeouts and separate per-process generation and embedding semaphores.

## API contract

`GET /health/live` is process liveness. `GET /health/ready` checks PostgreSQL schema, the reviewed config, selected provider model availability, the embedding fingerprint, and live vector dimension. Both are internal-network endpoints.

`POST /internal/chat` requires `Authorization: Bearer <INTERNAL_API_TOKEN>` and accepts/returns contract version `1.1.0`. The implementation validates request structure, history pairing, unique source documents, active generation scope, citation labels, and response markers. `direct_answer` is reserved for conversational and source-catalog responses that need no chunk citations. All other questions remain eligible for document-grounded QA. `rag-eval` uses the equally authenticated `POST /internal/evaluation/chat`, which runs the same graph and additionally returns candidate chunk IDs, source offsets, and `diagnostics`: the route, searches, corpus chunk count (for complete-corpus routes), processed chunk count, and reduction rounds. Reports save these as `queryDiagnostics`; the regular chat response remains the v1.1.0 contract.

OpenAPI is exported to `openapi.json` from the FastAPI app. No schema migrations or conversation persistence belong in this service.

`conversationId` is an optional request field for trace correlation. Both internal evaluation endpoints expose vector, lexical, hybrid, reranked, neighbor and selected stages with scores and per-query hits. `POST /internal/evaluation/retrieval` uses the same token/snapshot checks and graph, skipping sufficiency, answer generation and claim checks. `rag-eval --retrieval-cases <cases.jsonl>` runs gold chunk-ID cases; `--offline` evaluates saved stage IDs, and `--trace-file <events.jsonl>` replays traces. Reports include Recall@5/10/20/50, MRR, NDCG for chunk-ID cases and final source coverage. The public chat response contains no trace or stage data.

## Operations and blockers

- Run API and worker from the same image with separate entrypoints. Mount uploads read-only in the worker and mount the same reviewed config in both services.
- API timeout defaults to 120 seconds, below the NestJS 130-second and browser 150-second deadlines. Rebenchmark before changing these values.
- Global summaries and exhaustive extraction require more model calls than local search and use the same request deadline. Narrow the selected scope for very large documents. Persistent generation-scoped summaries/entity indexes are a future optimization. LLM extraction and reduction still need real-provider semantic evaluation; complete chunk coverage and valid citations alone do not prove correct lists, arithmetic, or summaries.
- Probe and record the embedding fingerprint whenever models are upgraded. Changing the embedding model or dimension requires a new full index generation.
- Worker leases last 90 seconds and are renewed every 20 seconds. Expired jobs are retried up to three automatic attempts; a document-level failure leaves an existing active generation in place.
- Upload storage is mounted read-only in the worker. Every write verifies the lease and deletion state; chunk batches reserve the shared 5,000-row capacity before insertion.
- The selected tokenizer must be pinned in `tokenizerVersion`; Ollama's REST API does not expose token counting.
- Contextual indexing requires a configured prompt version, model ID, model digest, and `INDEX_MODE=contextual`; baseline mode continues to embed original chunk text. Set `retrievalMode` to `vector` in the shared index configuration only when intentionally comparing the vector baseline. The default is scoped BM25 plus vector search, fusion, and reranking.
