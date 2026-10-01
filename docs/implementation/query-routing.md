# Query routing implementation

Goal: Answer whole-document requests with complete scoped evidence instead of a few nearest chunks, while retaining ordinary RAG for focused questions.

Constraints: Extend the existing structured planner and LangGraph. Preserve the public chat contract, original chunk citations, active generation filtering, embedding checks, provider adapters, token budgets, and request deadline. No schema migration or reindex is required. Classify with the LLM rather than keyword rules.

The intent model has four general tasks: conversation, source catalog, source question, and answer revision. Retrieval strategy is independent. Definition subject, shortening, and answer length are preferences rather than additional intent categories.

Files likely involved: `llm-backend/src/rag_app/generation/`, the retrieval repository, internal evaluation schemas, unit tests, OpenAPI, and the service README.

Success criteria:

- `LOCAL_SEARCH` retains scoped vector/hybrid retrieval, reranking, and top-K evidence selection.
- `GLOBAL_SUMMARY` reads every chunk in the requested document scope, creates cited section summaries, and reduces them hierarchically to fit the final prompt.
- `AGGREGATION_LIST` extracts requested facts/items from every chunk and merges exact duplicates with their provenance. Complete extracted data must fit the prompt; oversized results fail explicitly instead of quietly losing items.
- `MULTI_HOP` performs at most three retrieval rounds, resolving later searches from earlier evidence and keeping evidence from each round.
- Follow-ups retain their resolved subject and evidence strategy. Planning cannot expand the server-authorized source scope.
- The evaluation endpoint reports the route, searches, and corpus coverage without changing public chat responses.

Open questions: None blocking. Summaries are computed on demand for the current source generation; persistent summary/entity indexes are a future optimization.

Ordered steps:

1. Extend the validated planner with strategy, bounded initial searches, and optional document selection from source metadata.
2. Add ordered complete-corpus retrieval and cited map/reduce processing with token and completeness checks.
3. Connect global and iterative retrieval branches to existing generation/citation validation; expose internal diagnostics.
4. Add regression tests for coverage beyond top-K, list deduplication, hop-dependent searches, invalid plans, and budgets.
5. Run Python lint, strict type checking, unit tests, and OpenAPI export; review the diff and document behavior and costs.

Verification commands: `uv run ruff check src scripts tests`, `uv run pyright src`, `uv run pytest`, `uv run python scripts/export_openapi.py` from `llm-backend/`.

Risks: LLM planning/extraction quality, lossy summary reduction, output/context limits for long lists, and additional latency. Preserve citations through every reduction, reject incomplete extraction, and keep the existing deadline. Deterministic tests establish orchestration and provenance; real-provider semantic evaluation remains separate.

Stop conditions: Do not silently sample whole-document requests, change the shared database schema, or claim real-provider evaluation without running it.

Verification results:

- Python: 69 unit tests pass; Ruff passes for source, scripts, and all tests; strict Pyright passes for the changed generation, models, API, and retrieval modules.
- TypeScript: regenerated internal contracts, backend type checking, and all 9 contract tests pass.
- Full Python type checking reports 122 existing errors in evaluation, provider, and BM25 modules. The original `HEAD` revision reports the same 122 errors with the same dependencies.
- OpenAPI was exported and the diff passes whitespace checks. No live-provider semantic benchmark was run.

Review refinements: exhaustive lists take priority over brevity; partial source snapshots fail with `SCOPE_CHANGED`; successful multi-hop rounds must all contribute evidence that fits the final context; hop planning receives the original question as well as the resolved search topic.
