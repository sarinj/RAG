# Backend implementation plan

## Agent brief

Implement only `backend/**` with NestJS and TypeORM. Read [v1 contract](../contracts/v1.md), [structure guide](../project-structure.md), and [coordination guide](README.md). You own the public API and every database migration. Python uses your schema; coordinate changes through the lead before editing names, states, or column types.

You are not alone in the repository. Preserve other agents' edits; do not edit frontend, Python, root configuration/lockfiles, or shared contracts. Request cross-boundary changes from the lead. Do not spawn agents. Complete bounded assignments and report results.

## Folder ownership

```text
backend/src/
  main.ts, app.module.ts
  config/
  database/data-source.ts
  database/migrations/
  modules/documents/               # controller, DTOs, service, entity
  modules/ingestion/               # jobs, generations, chunk schema
  modules/conversations/           # turns, messages, citations
  modules/llm-client/              # internal HTTP + generated contract
  modules/storage/
  modules/health/
  common/                         # error filter, request IDs
backend/test/
backend/openapi.json
```

Entities live inside the owning module; migration ordering is centralized. Prefer straightforward services and TypeORM repositories, with custom SQL for locking/vector-related schema where necessary.

## Dependencies and inputs

- Lead confirms single-user/shared scope, limits, root infrastructure, and shared fixtures.
- LLM agent provides actual embedding dimension/digest and internal OpenAPI. Model-independent scaffolding can start before the probe; vector migration cannot be finalized before it.
- Lead supplies shared index configuration and internal token. Python and NestJS must read matching indexing settings.

## BE-01 — Bootstrap, public contract, schema

1. Scaffold NestJS, validation, configuration, error envelope, request IDs, and health endpoints.
2. Implement DTO schemas from v1; publish `backend/openapi.json` version 1.1.0. Include all error/202 variants so frontend generation does not guess them.
3. Add TypeORM entities and migrations for the exact shared table/column names. Disable synchronize.
4. Create pgvector extension, verified vector dimension, FKs, state checks, uniqueness/indexes, and document-to-active-generation ownership constraint.
5. Validate shared index configuration against the migrated embedding dimension at readiness.
6. Document runtime versus migration roles and provide migration commands to the lead.

Acceptance: migrations run on an empty PostgreSQL instance; schema matches the LLM repository contract; OpenAPI can generate frontend types; no model dimension is guessed.

## BE-02 — Upload, durable jobs, retries, deletion

Depends on BE-01; integrated acceptance requires LLM-02.

1. Implement local storage with generated storage keys, atomic temp-file move, size/type checks, and compensation if DB creation fails.
2. Enforce corpus/file limits and transaction-safe active content-hash uniqueness.
3. In one transaction create document, baseline generation/config snapshot, and queued job; return 202 Document.
4. Implement paginated list/detail with latestJob projection. Document ready state and latest job state are independent.
5. Implement retry: return an existing active job; otherwise create a new generation/job only for failed work, copying its configuration.
6. Implement deletion/tombstone and cancellation using the shared locking protocol; exclude tombstones from listing immediately.
7. Implement a recoverable backend cleanup sweep for file/index removal. Keep tombstone and citation provenance; remove jobs before deleting referenced generations. Cleanup must be idempotent after partial failure.

Acceptance: real ingestion reads the same storage volume/key; rejected upload leaves no live orphan document; concurrent retry creates one active job; worker cannot publish after deletion. Report known temporary file cleanup policy.

## BE-03 — Conversation and turn orchestration

Depends on BE-02 and internal chat OpenAPI; integrated acceptance requires LLM-03.

1. Create/list/get conversations with immutable all/selected scope. Validate ready selected IDs.
2. Resolve per-turn active generation snapshots; reject an empty corpus or invalid selected scope before inserting messages.
3. Atomically create chat_turn + user/assistant messages using idempotency/concurrency constraints. Implement same-ID replay and conflicting-content rejection exactly as v1.
4. Generate internal client types from Python OpenAPI; validate Python responses at runtime. Send bounded completed history, source snapshot, request ID, and server retrieval mode.
5. Apply ordered deadlines. On completion lock/recheck source documents and pending turn before storing the answer/citations. Discard stale-scope results.
6. Persist sanitized failure state on provider errors. Add expiry recovery for pending turns after crashes; prevent a late result overwriting a failed/completed turn.
7. Implement ascending paginated history and pending duplicate 202 response. RequestId in public messages comes from the turn record.

Acceptance: repeated requests cannot create duplicate turns; another pending turn is rejected; API restart recovers stuck state; source deletion/reindex during inference fails safely; no transaction remains open during model inference.

## BE-04 — Citation access and provenance

1. Validate internal citation IDs belong to the resolved document/generation/chunk set before persistence; public response must not leak arbitrary Python IDs.
2. Implement original-text chunk lookup, including old successful generations for historical citations.
3. Return 410 for deleted sources and 404 for nonexistent or unrelated chunks. Source payload omits answer-specific citation labels.
4. Keep generation/chunk provenance IDs after physical cleanup, per the no-FK citation design.

Acceptance: stored citations resolve before deletion and fail with the specified code afterward; old citations survive reindexing; generated chunk context is never returned as original source text.

## BE-05 — Reindex and operational readiness

Depends on a verified baseline and LLM-04.

1. Add reindex endpoint for ready documents using current index configuration. Deduplicate active jobs.
2. Retain active generation and ready status during replacement; expose latestJob failure without destroying old availability.
3. Ensure staging plus retained chunks count toward configured capacity; use the exact capacity advisory-lock protocol in v1, rather than independent race-prone checks.
4. Confirm errors/status enums match fixtures and refresh OpenAPI/client exports after coordinated changes.
5. Document schema setup, cleanup recovery, failed turn recovery, and environment configuration.

Acceptance: contextual and hybrid rollout changes server configuration, not public payload shapes. Reindex failures keep prior content usable and model-space mismatch gets an explicit error.

## Verification and handoff

Use real PostgreSQL for migration/locking/idempotency checks. Cover atomic upload, duplicate hash, repeated retry, lease/delete race with Python integration, pending-turn uniqueness, deadline recovery, and citation deletion. Unit tests can use a fake internal LLM client; integration acceptance must include the real Python service.

Provide scripts for lint/typecheck/build/unit/integration checks and OpenAPI export. Planned commands are not implemented yet. Return task ID, migration/contract versions, changed files, checks/results, environment additions, and the exact frontend/Python integration dependencies still outstanding.
