# Project structure and development guide

Use the [service implementation plans](implementation/README.md) for independent agent assignments and the [shared v1 contract](contracts/v1.md) for authoritative API/database details. The contract takes precedence over interface sketches below.

This guide defines the intended structure for the [RAG MVP plan](mvp-implementation-plan.md). The service roots are `frontend`, `backend`, and `llm-backend`. The latter replaces `ai-backend`.

The layouts below are blueprints. Add directories and abstractions when their first implementation needs them; do not generate empty layers for every feature.

For implementation with multiple agents, use the [agent execution plan](agent-execution-plan.md) alongside these ownership boundaries.

## Repository layout

```text
learning-RAG/
├── frontend/                     # Next.js browser application
├── backend/                      # NestJS product API and schema ownership
├── llm-backend/                  # Python RAG API and ingestion worker
├── materials/                    # Learning references
├── docs/
│   ├── mvp-implementation-plan.md
│   ├── project-structure.md
│   ├── decisions/                # Short architecture decision records
│   └── runbooks/                 # Setup, recovery, reindexing
├── infra/                        # Compose overrides and infrastructure config
├── scripts/                      # Repository-level developer commands
├── tests/e2e/                    # Cross-service user journeys
├── compose.yaml
├── package.json                  # Workspace commands; no application logic
├── pnpm-workspace.yaml           # frontend and backend
├── pnpm-lock.yaml
├── .env.example                  # Safe local configuration template
├── .gitignore
└── README.md                     # Setup, commands, architecture links
```

Use one repository so an API contract change can be reviewed with its consumers. Each service has its own dependencies, Dockerfile, README, configuration validation, and checks. The Python project uses `pyproject.toml` and `uv.lock`; it is not a pnpm package.

Avoid a shared business-logic package initially. Exchange JSON over versioned HTTP contracts; generate frontend API types from the NestJS OpenAPI schema. Keep Python schemas and the internal API contract checked together. Database entities must never become browser DTOs.

## Frontend: organize by user feature

```text
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── providers.tsx
│   │   ├── globals.css
│   │   ├── error.tsx
│   │   ├── not-found.tsx
│   │   ├── documents/
│   │   │   ├── page.tsx
│   │   │   └── loading.tsx
│   │   └── chat/[conversationId]/
│   │       ├── page.tsx
│   │       └── loading.tsx
│   ├── features/
│   │   ├── documents/
│   │   │   ├── api.ts
│   │   │   ├── queries.ts
│   │   │   ├── mutations.ts
│   │   │   ├── schemas.ts
│   │   │   ├── components/
│   │   │   │   ├── document-library.tsx
│   │   │   │   ├── upload-dropzone.tsx
│   │   │   │   └── ingestion-status.tsx
│   │   │   └── index.ts
│   │   └── chat/
│   │       ├── api.ts
│   │       ├── queries.ts
│   │       ├── mutations.ts
│   │       ├── components/
│   │       │   ├── chat-workspace.tsx
│   │       │   ├── message-list.tsx
│   │       │   ├── message-composer.tsx
│   │       │   ├── document-scope-picker.tsx
│   │       │   └── citation-drawer.tsx
│   │       └── index.ts
│   ├── components/
│   │   ├── ui/                    # shadcn primitives
│   │   └── layout/                # App shell and shared navigation
│   ├── lib/
│   │   ├── http.ts                # Axios instance and error normalization
│   │   ├── query-client.ts
│   │   └── utils.ts               # Small UI utilities only
│   ├── config/env.ts
│   └── generated/api.ts           # Generated; never manually edited
├── public/
├── components.json
├── package.json
├── tsconfig.json
├── next.config.ts
├── Dockerfile
└── README.md
```

### Frontend rules

- Route files compose screens and handle route concerns. Keep upload and chat behavior inside their features.
- Keep server components as the default for route/layout composition. Place `use client` at interactive boundaries such as the upload form and chat workspace.
- TanStack Query owns remote state: documents, job status, conversations, and messages. Component state owns drafts, open drawers, and temporary selections. Avoid copying query results into another global store.
- Feature `api.ts` modules call Axios. Query modules define stable keys and fetch options; mutation modules handle cache invalidation. Include conversation and scope identifiers where they affect cached results.
- Stop status polling at terminal states and pause unnecessary background polling. Upload percentage and ingestion progress are separate UI states.
- Use `components/ui` for reusable visual primitives. Components with document or chat rules belong to their feature.
- Export the small feature surface through `index.ts`. Other features should not import private component paths. When chat needs a reusable document selector, export it from documents and let chat compose it.
- Keep feature-specific types near their use. Use generated types for network payloads; use local schemas for form validation where needed.
- Only public configuration may use `NEXT_PUBLIC_*`. Never put model tokens, database credentials, or internal service tokens in browser code.
- Render model Markdown with raw HTML disabled or sanitized; allow safe link protocols. Support keyboard upload controls, readable status labels, and accessible error messages.
- Co-locate focused component/hook tests as `*.test.ts(x)`. Put complete upload-to-answer browser journeys in root `tests/e2e`.

Dependency direction: `app → features → shared UI/lib`. Shared UI must not import features. Avoid routing normal product operations through duplicate Next.js API endpoints when NestJS already owns them.

## Backend: organize by product capability

```text
backend/
├── src/
│   ├── main.ts
│   ├── app.module.ts
│   ├── config/
│   │   ├── env.validation.ts
│   │   └── configuration.ts
│   ├── database/
│   │   ├── data-source.ts
│   │   └── migrations/
│   ├── modules/
│   │   ├── documents/
│   │   │   ├── documents.module.ts
│   │   │   ├── documents.controller.ts
│   │   │   ├── documents.service.ts
│   │   │   ├── dto/
│   │   │   ├── entities/document.entity.ts
│   │   │   └── documents.repository.ts
│   │   ├── ingestion/
│   │   │   ├── ingestion.module.ts
│   │   │   ├── ingestion.service.ts
│   │   │   └── entities/           # Jobs, generations, chunk schema mappings
│   │   ├── conversations/
│   │   │   ├── conversations.module.ts
│   │   │   ├── conversations.controller.ts
│   │   │   ├── conversations.service.ts
│   │   │   ├── dto/
│   │   │   └── entities/           # Conversations, messages, citations
│   │   ├── llm-client/
│   │   │   ├── llm-client.module.ts
│   │   │   ├── llm-client.service.ts
│   │   │   └── contracts/
│   │   ├── storage/
│   │   │   ├── storage.module.ts
│   │   │   ├── storage.service.ts
│   │   │   └── local-file.storage.ts
│   │   └── health/
│   └── common/
│       ├── errors/
│       ├── filters/
│       └── interceptors/          # Request IDs and request logging
├── test/                         # Integration tests using real PostgreSQL
├── package.json
├── tsconfig.json
├── nest-cli.json
├── Dockerfile
└── README.md
```

### Backend rules

- Controllers validate and translate HTTP requests/responses. Services implement use cases, state transitions, and transaction boundaries.
- Keep TypeORM entities inside their owning module. The central data source discovers them; migrations stay centralized. This refines the simpler database folder in the MVP overview.
- Use TypeORM repositories directly for simple CRUD. Add a custom repository for repeated or complex queries; do not introduce generic base repositories and services.
- Export services through Nest modules. A module must not reach into another module's private repository. Keep dependency direction clear: documents may use ingestion and storage; conversations may use documents and llm-client. Avoid circular module dependencies.
- Keep public DTOs distinct from persisted entities. Do not leak storage paths, internal errors, or vector fields through automatic entity serialization.
- NestJS owns all migrations, including Python-used tables. Disable automatic schema synchronization. Review raw SQL for leases, vector operations, and extension setup alongside ordinary migrations.
- Use short transactions for related database state. Do not hold a transaction open during model inference or file parsing.
- External work needs idempotency and compensation: a failed DB insert cleans up its uploaded file; an expired worker lease can be retried without duplicate publication.
- Configure explicit timeouts for internal HTTP calls. Retry only safe operations; chat retries require a request ID and persisted state.
- Use validated configuration through dependency injection, not scattered environment reads. Return consistent errors with request IDs and log sanitized context.
- Add authentication/authorization as a deliberate feature if shared use is chosen. Service tokens between containers are not user authorization.
- Co-locate unit tests as `*.spec.ts`; put transaction, migration, and HTTP integration tests in `test/`.

Dependency direction: `controller → use-case service → repository or external adapter`. Add domain objects or separate application layers only when business rules become complex enough to justify them.

## LLM backend: organize around the RAG pipeline

```text
llm-backend/
├── src/rag_app/
│   ├── main.py                    # FastAPI app factory and wiring
│   ├── worker.py                  # Worker entrypoint and shutdown handling
│   ├── config.py                  # Validated settings
│   ├── api/
│   │   ├── routes/chat.py
│   │   ├── routes/health.py
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── dependencies.py        # Repositories and model clients
│   ├── ingestion/
│   │   ├── pipeline.py
│   │   ├── parsers/
│   │   │   ├── pdf.py
│   │   │   ├── text.py
│   │   │   └── markdown.py
│   │   ├── chunking.py
│   │   └── contextualization.py
│   ├── retrieval/
│   │   ├── pipeline.py
│   │   ├── vector.py
│   │   ├── bm25.py
│   │   ├── tokenization.py
│   │   └── fusion.py
│   ├── generation/
│   │   ├── answering.py
│   │   ├── query_rewrite.py
│   │   ├── citations.py
│   │   └── prompts/
│   │       ├── contextualize_v1.txt
│   │       ├── answer_v1.txt
│   │       └── rewrite_v1.txt
│   ├── providers/
│   │   ├── interfaces.py          # Narrow generation/embedding contracts
│   │   └── ollama.py
│   ├── repositories/
│   │   ├── connection.py
│   │   ├── jobs.py
│   │   ├── chunks.py
│   │   └── documents.py
│   └── models.py                  # Internal chunks/evidence/result types
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                  # Small non-sensitive sample documents
├── evals/
│   ├── datasets/
│   ├── run.py
│   └── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
└── README.md
```

### LLM backend rules

- Routes validate transport data and call pipeline functions. Pipelines orchestrate work; parsers, chunkers, rank fusion, and citation checks remain independently callable.
- Use typed internal records for parsed documents, chunks, retrieved evidence, and answers. Keep HTTP models, database rows, and model-provider responses separate.
- Keep provider-specific HTTP calls in `providers/ollama.py`. Define only the small generation and embedding interfaces actually needed, so changing the model runtime does not change retrieval logic.
- Version prompts and record their versions with ingestion generations/evaluation results. Package prompt files in the Python distribution and Docker image; avoid paths dependent on the current working directory.
- PostgreSQL is the source of truth. In-memory BM25 state is derived and disposable. Any later cache key must account for scope, active generations, and tokenizer configuration.
- Apply the same document/generation scope to lexical and vector retrieval. Never run a global search then filter only the final answer.
- Keep source text, generated context, and searchable text distinct. Citation evidence comes from original text and provenance.
- Model token budgets explicitly: contextualizer input, embedding input, retrieved evidence, history, and answer output. Disable silent truncation where it would invalidate ingestion or evidence.
- Run ingestion in the dedicated worker, not FastAPI background tasks. Bound concurrency, maintain leases, and verify lease ownership before publishing results. If deletion or lease loss occurs, the worker must not publish.
- Avoid blocking parsing or synchronous database calls on the async API event loop. Use consistent async I/O or explicit thread offloading. A separate worker keeps ingestion CPU work out of request handling.
- Python may update leased job progress and publish chunks according to the database contract. It does not create users, manage conversations, or define schema migrations.
- Start with one API process and one ingestion worker. Increase concurrency only after measuring model memory and queue behavior.
- Unit tests cover deterministic logic without requiring models. Integration tests cover PostgreSQL and provider contracts. Evaluations measure retrieval and answer quality using actual model outputs; keep them separate from fast checks.

Dependency direction: `API/worker → pipelines → algorithms, repositories, provider interfaces`. Parsing and rank fusion must not import FastAPI or know about HTTP requests.

## Cross-service contracts and operations

| Concern | Owner | Contract |
| --- | --- | --- |
| Public HTTP API | backend | OpenAPI; generated frontend types |
| Internal chat API | llm-backend | Pydantic/OpenAPI; checked NestJS client |
| Database schema | backend | Ordered TypeORM migrations |
| Chunk content and index publication | llm-backend | Generation/lease invariants documented with schema |
| Document lifecycle | backend | Python publishes readiness only under the agreed job protocol |
| Conversation persistence | backend | LLM backend returns an answer result |
| Upload bytes | backend | Generated storage key; read-only worker mount |
| Model and prompt configuration | llm-backend | Versioned generation metadata |

Use stable IDs, UTC timestamps, and a request ID across API and Python calls. Log stage timings and IDs, not raw document content or full prompts by default. Expose liveness separately from readiness; migration and model-bootstrap failures should be visible at startup.

Commit lockfiles and safe `.env.example` files. Keep secrets, uploaded documents, model weights, and local database files out of Git. Give each process only the database privileges and volume access its role needs. The migration role may change schema; runtime roles should not.

Run migrations once as an explicit startup/deployment step rather than racing across replicas. Document backup and restore of both PostgreSQL and uploads; they represent one logical data set.

## Naming and extension guidance

Use kebab-case for frontend filenames, Nest conventions such as `documents.service.ts`, and snake_case for Python modules. Prefer names that identify the capability over generic `manager`, `helper`, or `common-service` names. Shared folders should contain genuinely reused code, not code without a clear owner.

Examples of later changes:

- **DOCX support:** add a parser in the LLM service, update backend upload validation and UI messaging, and add a parsing fixture. Retrieval should remain unchanged.
- **Object storage:** replace the local storage implementation and define how workers access objects. Keep user-facing document IDs and API payloads stable.
- **Different model server:** add a provider adapter and run the same evaluation dataset. A new embedding model also creates a new indexing generation.
- **Reranking:** add a retrieval stage between fusion and final evidence selection, with latency/relevance evaluation.
- **Shared users:** introduce identity and ownership in NestJS, propagate authorized scopes to Python, and verify access for every file, message, and citation endpoint before exposing the service.
- **Streaming:** define an event contract for answer deltas, completion, errors, and cancellation; implement it across both services and persist only according to explicit final/failed state rules.

For each implementation slice: update the contract first when needed, implement a complete user-visible behavior, run focused checks, and record consequential architecture decisions in `docs/decisions/`. Keep speculative extensibility out of the MVP until a real requirement needs it.
