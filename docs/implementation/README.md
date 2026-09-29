# Independent implementation entrypoints

Give each agent its plan and the same [v1 contract](../contracts/v1.md). The plans partition writable files; they do not create separate versions of the API or database.

| Agent | Plan | Writable service |
| --- | --- | --- |
| Frontend | [Frontend plan](frontend.md) | `frontend/` |
| Backend | [Backend plan](backend.md) | `backend/` |
| LLM backend | [LLM backend plan](llm-backend.md) | `llm-backend/` |

The lead owns shared contracts, root infrastructure, fixtures, pnpm lockfile, and cross-service E2E tests. Each agent requests contract changes through the lead; agents do not edit other services or spawn further agents.

## Launch sequence

1. Confirm hardware, files/languages, and local/shared scope. The v1 draft assumes local single-user use, text PDF/TXT/Markdown, and non-streaming chat. Shared authentication would change the contract before implementation.
2. Read and review v1 with all three owners. Produce the shared JSON fixtures. LLM owner probes models; lead publishes the shared index configuration. Backend records the verified vector dimension before migrations.
3. Bootstrap all services in parallel; lead installs TS dependencies serially and wires Compose. Producers publish OpenAPI; consumers generate types. Each producer owns its OpenAPI export.
4. Implement upload concurrently: FE-02, BE-02, LLM-02. Integrate real upload → ready before baseline chat acceptance.
5. Implement baseline chat concurrently: FE-03/04, BE-03/04, LLM-03. Capture a real end-to-end baseline.
6. Add contextualization/reindexing, then BM25/RRF; compare quality before enabling hybrid defaults.
7. Fresh read-only tester/reviewer sessions check the integrated state. Lead assigns fixes to the owning builder.

Model-dependent work waits for the model configuration. UI and API scaffolding can proceed using reviewed fixtures. No agent may claim a completed integration using mocks alone.

## Contract checklist before each integration

- Producer OpenAPI agrees with v1; generated consumer types are current.
- Payloads use the same names, nullability, status enums, citation labels, and pagination.
- Both services use the same generation configuration and database names.
- Error/status handling, idempotency, timeout order, and source deletion behavior match.
- Service-focused checks and a real cross-service user flow pass.

Use the [structure guide](../project-structure.md) for folder detail and the [MVP plan](../mvp-implementation-plan.md) for product intent. The v1 contract resolves interface ambiguities in those higher-level documents.
