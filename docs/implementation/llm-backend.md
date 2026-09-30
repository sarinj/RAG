# LLM backend implementation plan

## Agent brief

Implement only `llm-backend/**` using Python, FastAPI, Pydantic, psycopg, pgvector, and Ollama. Read [v1 contract](../contracts/v1.md), [structure guide](../project-structure.md), [learning material](../../materials/rag-learning-guide.md), and [coordination guide](README.md).

You own model execution, ingestion worker, retrieval, and internal chat. You do not own migrations or conversation persistence. You are not alone in the repository: preserve others' work, request schema/API changes through the lead, do not edit outside your service, and do not spawn agents.

## Folder ownership

```text
llm-backend/
  src/rag_app/
    main.py, worker.py, config.py, models.py
    api/                           # routes, schemas, dependencies
    ingestion/                     # parsing, chunking, contextualization
    retrieval/                     # vector, tokenization, BM25, fusion
    generation/                    # answering, rewriting, citation checks, prompts
    providers/                     # narrow interfaces + Ollama adapter
    repositories/                  # shared-schema SQL, leases, publication
  tests/unit/, tests/integration/, tests/fixtures/
  evals/datasets/, evals/run.py
  openapi.json
  pyproject.toml, uv.lock, Dockerfile, README.md
```

Chat uses LangGraph `StateGraph` to make request stages and the bounded citation-repair branch explicit. Graph nodes still orchestrate independently testable algorithms, provider adapters, and repositories; Ollama access, SQL retrieval, and durable application persistence remain outside the graph framework.

## Dependencies and inputs

- Hardware/language/model selection requires user input; proposed models are candidates only.
- Lead provides runtime URLs, token, shared configuration and fixtures, corpus limits, and source volume.
- Backend provides authoritative migrations. Use the database schema exactly as documented, including job lease token, config JSON keys, and state transitions.

## LLM-01 — Probe, bootstrap, and internal contract

1. Scaffold Python src package, FastAPI app factory, worker entrypoint, settings validation, and health endpoints.
2. Add Ollama generation/embedding adapter with explicit timeouts, typed results, and controlled concurrency.
3. Probe model availability, digest, embedding dimension, tokenizer/input budget behavior, and a representative multilingual sample if required. Report usable configuration to the lead; do not freeze a model from download size alone.
4. Lead publishes reviewed index config; verify it on startup against live models and database dimension.
5. Implement Pydantic internal schemas from v1, token validation, error envelope, and OpenAPI export to `llm-backend/openapi.json` version 1.0.0.
6. Add DB connection management and typed internal document/chunk/evidence records.

Acceptance: Python and NestJS can generate/validate the same internal payloads; model-independent checks work without model downloads; a real probe supplies the exact dimension needed by BE-01. No migrations are added in Python.

## LLM-02 — Recoverable baseline ingestion

Depends on shared schema/migrations and config; integrated acceptance requires BE-02.

1. Claim queued/expired jobs with SKIP LOCKED and new lease tokens. Increment attempts, heartbeat, and bound retries exactly as v1.
2. Read source only through generated storage key under the mounted upload root. Reject traversal and missing/deleted source.
3. Parse text PDF/TXT/Markdown; normalize text consistently and preserve offsets/pages/headings. Reject empty/scanned/encrypted unsupported files with sanitized errors.
4. Chunk with configurable structural boundaries and overlap; enforce tokenizer budgets instead of assuming character count equals tokens.
5. Embed original chunks in baseline mode with the recorded embedding model/config. Verify vector length and finite values.
6. Stage chunks only under the current unexpired lease. On retry clear previous staging before rebuilding; serialize corpus-capacity checks using the shared locking rule.
7. Publish complete generation atomically with job success and active pointer swap. Never hold transactions across model calls. Handle lease loss/deletion without publication.
8. Implement graceful stop: stop claiming work, finish or abandon safely, and let lease recovery reclaim interrupted jobs.

Acceptance: a real upload becomes ready; restart reclaims work without duplicate chunks; stale workers cannot write/publish; page/heading metadata is correct; partial generations never appear in retrieval.

## LLM-03 — Baseline chat and evaluation

Depends on LLM-02 and reviewed internal contract; integrate with BE-03/04.

1. Validate nonempty source snapshot, non-deleted documents, active generations, and common embedding configuration.
2. Optionally rewrite follow-up question from bounded history; preserve the original question for answering. History is not citable evidence.
3. Embed query using the provider's required query format. Run exact cosine search with document/generation filtering in SQL.
4. Select original evidence under a model token budget, preserving source IDs and page/heading context. Assign deterministic per-answer labels S1, S2, etc.
5. Prompt for evidence-based answers and no-evidence response, treating document instructions as untrusted text.
6. Validate citation markers against supplied chunks. Permit one bounded repair; fail `INVALID_CITATIONS` if still invalid. Track insufficient-evidence as a normal response with empty citations.
7. Return only the v1 ChatResponse; do not write messages or expose debug retrieval results through the public contract.
8. Build 20–30 representative labeled evaluation questions and record baseline recall/hit rate, citation validity, human-reviewed grounding, ingestion time, and answer latency.

Acceptance: real requests return validated source IDs, selected scope cannot leak other documents, model outage/timeout maps to contract errors, and baseline report identifies model/config/dataset versions. Do not claim valid citations alone prove grounding.

## LLM-04 — Contextual indexing (implemented)

Depends on a saved baseline and BE-05 reindex protocol.

1. Add a versioned full-document + chunk contextualization prompt producing concise search context.
2. Budget full document, chunk, prompt, and output using the configured model. Reject over-budget documents with CONTEXT_LIMIT_EXCEEDED; never silently truncate or substitute section context.
3. Store original_text separately from context_text; searchable_text is context + original. Embed searchable_text.
4. Snapshot model/prompt/chunker/tokenizer settings for each generation and reuse that config through retries.
5. Publish replacement atomically, preserving old successful generation for citations. Respect lease ownership and storage/corpus cap during staging.
6. Compare contextual vector retrieval against baseline using the same labeled source spans. Map labels by source provenance if generation-specific chunk IDs change.

Acceptance: contextualization runs during ingestion only; answers/citation source bodies still use original text; failed replacement leaves old generation usable; contextual report records actual gains or regressions.

## LLM-05 — BM25 and fusion (implemented)

Depends on LLM-04 and language/tokenizer decision.

1. Tokenize both queries and searchable_text consistently; use appropriate segmentation for required languages.
2. Build BM25 over exactly the requested active source snapshot. Start with per-request rebuilding under the corpus cap; no unversioned cache.
3. Retrieve up to 30 lexical and 30 vector candidates. Remove lexical nonmatches, deduplicate by chunk ID, and apply equal-weight RRF with constant 60 initially.
4. Deterministically break ties; fuse the best 30 candidates and rerank them before selecting up to five chunks subject to the evidence token budget.
5. Evaluate vector baseline, contextual vector, and contextual hybrid with fixed datasets. Tune on tuning questions only and report held-out results separately.

Acceptance: exact terms and paraphrases are represented in evaluation, scope is identical for both retrieval paths, lexical misses do not fabricate BM25 candidates, and retrieval mode comes from the shared index configuration.

## LLM-06 — Final service readiness

1. Check API/worker deployment from the same image, model loading/readiness, configured deadlines, and packaged prompt resources.
2. Confirm async API handlers do not block on CPU parsing or sync I/O; ingestion runs in its separate worker.
3. Add sanitized structured logs with request/job/generation IDs and stage timings, excluding full document text/prompts by default.
4. Document recovery, model bootstrap, embedding-model reindex requirements, evaluation command, and known language/model limits.

## Verification and handoff

Unit checks cover parser spans, chunk boundaries, budgets, fusion, no-match lexical retrieval, and citation validation. PostgreSQL integration checks cover leases, expiry, duplicate staging, deletion/publication races, and scope filtering. Provider checks and evaluation require real selected models and must be reported separately from deterministic tests.

Provide lint/typecheck/test/OpenAPI-export commands in the service README; `uv run --project llm-backend pytest` is a planned test entrypoint. Return task ID, changed paths, schema/internal contract version, exact model/config identifiers, checks/evaluation results, remaining blockers, and requested cross-service changes.
