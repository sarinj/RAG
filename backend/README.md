# Backend

NestJS owns the public API, upload storage, PostgreSQL schema, document lifecycle, saved conversations, and citation references. Python owns parsing, chunk creation, model calls, and ingestion job execution.

## Local setup

Use Node.js 22.13+ or 24.11+. Install dependencies with `npm install`, copy `.env.example` to `.env`, then provide the same reviewed `index-config.json` used by the Python service at `INDEX_CONFIG_PATH`. That artifact must contain the model-probed embedding model, digest, and dimension plus the shared generation settings. The migration refuses to guess a vector dimension and fails if the artifact is missing or invalid.

Run `npm run build` before `npm run migration:run` after PostgreSQL with pgvector is available, then `npm run start:dev`. The public API listens on port 3001 by default. `GET /health/live` checks process liveness independently of PostgreSQL; `GET /health/ready` checks the complete migrated schema, vector dimension, fixed 5,000-chunk capacity, upload storage, shared index artifact, and internal Python readiness. Swagger is served at `/docs`, and the checked-in OpenAPI contract is generated with `npm run openapi:export`.

`npm run build` and `npm run typecheck` compile the service. Run `npm run llm-contract:generate` after the Python service exports its OpenAPI contract, and `npm run llm-contract:check` in CI to detect generated type drift. `npm run migration:run:build` and `npm run migration:revert:build` combine compilation with migration execution; the direct migration scripts run the already compiled files. Migrations are the only schema setup path; TypeORM `synchronize` remains disabled.

## Runtime behavior

- Uploads are limited to 50 MiB by default and stored under UUID keys. A failed database transaction removes the stored file. Startup removes abandoned temporary upload files older than one hour, and the cleanup sweep removes old unreferenced UUID uploads left by a process crash.
- Every worker chunk writer must lock and reserve capacity with `SELECT rag_reserve_chunk_capacity(batch_size)` after locking its document and job, then insert chunks in the same transaction. Database statement triggers serialize inserts and reject any write that would exceed the fixed 5,000-chunk contract cap.
- Upload, retry, and reindex create immutable generation and job records. The worker publishes a new generation only after it has finished writing its chunks.
- Deletion commits a tombstone and cancels active work before returning. A periodic cleanup pass removes the file, chunks, jobs, and generations while retaining the tombstone, conversation links, and citation identifiers.
- Chat calls Python synchronously with a 130-second default timeout. Turn recovery marks requests past their 180-second expiry failed. A citation is persisted only after the referenced original chunk is resolved inside that turn's document snapshot.

`POST /api/conversations/:id/messages/stream` accepts the same idempotent chat body
and sends SSE progress and provisional answer deltas. Its final `done` event contains
the saved turn after source checks and persistence. An accepted turn keeps running
if the viewer disconnects; reconcile through saved history. Duplicate requests replay
their saved status without starting another generation. See the
[streaming contract](../docs/contracts/chat-streaming.md) for events and error handling.

The LLM service, a real model configuration artifact, PostgreSQL with pgvector, root Compose wiring, and cross-service integration fixtures are separate dependencies. Until they are supplied, the backend can build and export OpenAPI, but migrations and readiness cannot complete.
