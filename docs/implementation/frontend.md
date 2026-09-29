# Frontend implementation plan

## Agent brief

Implement only `frontend/**` using Next.js App Router, TypeScript, Axios, shadcn/ui, and TanStack Query. Read [v1 contract](../contracts/v1.md), [structure guide](../project-structure.md), and [coordination guide](README.md) first. The v1 contract is authoritative; do not copy and independently modify its DTOs. Frontend calls only NestJS.

You are not alone in the repository. Preserve other agents' work. Request root dependency installation and contract changes from the lead. Do not edit root lockfiles, backend/Python files, shared contract fixtures, or infrastructure. Do not spawn agents. Complete one assigned task and report evidence before starting another.

## Folder ownership

```text
frontend/
  src/app/                         # routes, layouts, providers
  src/features/documents/          # api, query keys, mutations, components
  src/features/chat/               # conversations, turns, sources
  src/components/ui/               # shadcn primitives
  src/components/layout/           # shell/navigation
  src/lib/                         # Axios, QueryClient
  src/config/                      # validated public config
  src/generated/api.ts             # generated from backend/openapi.json
  public/
  package.json, Dockerfile, README.md
```

Use the detailed file layout in the structure guide. Add files for implemented behavior, not empty scaffolding layers.

## Dependencies and inputs

- Lead provides scope decisions, public API URL, shared v1 JSON fixtures, and pnpm workspace wiring.
- Backend provides public OpenAPI and readiness endpoint. Before it is available, use fixture adapters conforming to the reviewed contract; do not hand-maintain a second permanent interface package.
- Model settings, vector types, storage keys, and internal service credentials are not frontend concerns.

## FE-01 — Foundation and API boundary

1. Scaffold Next.js and shadcn, app layout/navigation, documents route, and chat route.
2. Add QueryClient provider at a client boundary. Keep static route/layout composition in server components.
3. Configure Axios with `NEXT_PUBLIC_API_BASE_URL`, request correlation, timeout, and one error-envelope parser. Do not automatically retry message/upload mutations.
4. Generate API types from backend OpenAPI; expose feature-level API functions using them. Keep query keys in each feature.
5. Add fixture-backed development transport only if the real API is pending; make its activation explicit and default it off.

Acceptance: routes render, config failures are clear, HTTP errors display safe messages, and no internal credentials enter the client bundle. Report fixture usage explicitly. Lead runs pnpm installation after manifest changes.

## FE-02 — Document library

Depends on FE-01 and reviewed document fixtures; real acceptance requires BE-02 + LLM-02.

1. Build single-file upload with configured type/size hints; backend remains validation authority.
2. Distinguish browser upload progress from queued/processing ingestion state.
3. Query paginated documents and display name, size, availability, latest stage, and retry action.
4. Poll every 2 seconds while a latestJob is queued/processing, with background polling disabled. Stop on terminal jobs; invalidate list/detail after mutations.
5. Implement retry and delete. On a ready document with a failed reindex, retain ready availability and show the job failure separately.
6. Handle duplicate, unsupported, oversized, extraction-failed, and deleted states. Deletion copy explains that saved chat text remains.

Acceptance: a real supported file appears as ready; a failed file exposes the backend error; retry does not duplicate list entries; deletion removes the file from the active list. No hardcoded fake readiness or progress.

## FE-03 — Conversations, scope, and message lifecycle

Depends on FE-01 and chat contract; real acceptance requires BE-03 + LLM-03.

1. Add all-files/selected-files picker. Only ready documents are selectable; never serialize empty selected scope as all.
2. Create a conversation using POST, then navigate to `/chat/:id`; load details/history with generated DTOs and pagination.
3. Scope is immutable in v1. Offer a new conversation to change selected documents.
4. Generate a UUID requestId once per intentional send and retain it for transport retries. POST content; show pending state; disable a different send while a turn is pending.
5. Handle 200 completed Turn and 202 pending Turn. Poll history for pending assistants. On network timeout or server failure, reconcile saved history before offering another attempt.
6. If a saved failed turn is deliberately retried, use a new request ID and explain its state through UI. Never silently resend with a new ID after a timeout.
7. On reload, recover pending/completed/failed messages from the API, not component state.

Acceptance: completed history survives reload; pending duplicates do not create duplicate messages; unavailable scope gets a useful error; insufficient evidence is a normal completed answer.

## FE-04 — Source citations and answer rendering

Depends on FE-03 and BE-04.

1. Render Markdown safely with raw HTML disabled/sanitized and safe URL protocols.
2. Associate `[S1]` markers with citation metadata in the answer. Do not infer citations from arbitrary model URLs.
3. Open a source drawer by document/chunk IDs; display original source text, document name, and page/heading from the source endpoint.
4. Handle 410 with “Source deleted”; handle missing source without breaking the conversation. Do not cache an excerpt indefinitely after a document deletion; invalidate relevant source queries.
5. Provide keyboard access, visible loading/error states, and accessible names for source controls.

Acceptance: clicking a real citation displays its exact original excerpt; deleting its document removes source access while historical chat remains understandable.

## FE-05 — Contextual reindex and final integration

Depends on BE-05/LLM-04.

1. Add a reindex action calling the server-configured endpoint, without exposing arbitrary model parameters.
2. Preserve chat availability for ready documents while latestJob processes a replacement.
3. Confirm terminal polling, first-ingestion failure, replacement failure, empty library, and model-timeout flows.
4. Remove accidental fixture dependencies from the real application path and document local startup/configuration.

Acceptance: contextualization and hybrid retrieval require no frontend DTO changes; only document/job state and generated answers differ.

## Verification and handoff

Create focused tests for scope serialization, request ID preservation, status polling, and source error rendering where they protect behavior. Lead owns the root browser E2E suite. Planned service scripts: `lint`, `typecheck`, `test`, `build`; run them once implemented. Commands are not present yet.

Return task ID, changed paths, OpenAPI version used, commands/results, screenshots or observed UI behavior, fixture dependencies, and outstanding integration issues. Do not call the service complete until the real upload → ready → ask → citation flow works with NestJS and Python.
