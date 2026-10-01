# RAG correctness implementation packet

Goal: make evidence losses observable, improve scoped retrieval and grounding, and support
repeatable evaluation without changing the public chat response or ingestion ownership.

Constraints: preserve staged routing work, PostgreSQL/pgvector, generation snapshots, worker
leases, original citation text, the NestJS data boundary, and one bounded answer repair.
Planner scope is intersected with the server snapshot; no model output authorizes sources.

## Architecture inspected before implementation

- `api/chat.py`: bearer-token dependency, contract check, active-snapshot validation,
  deadline, executor invocation, and selected-evidence citation membership check.
- `generation/chat.py`: LangGraph prepare → understand → direct/local/corpus/multi-hop
  branches → rank/select → generate → optional repair → finalize.
- `generation/intents.py` and `query_rewriting.py`: typed `ChatPlan`, bounded recent
  history, standalone topic query, document narrowing, and two-attempt planner fallback.
- `repositories/retrieval.py`: active generation/document joins, embedding compatibility,
  vector cosine ordering (30), a per-request BM25 corpus (5,000), full-corpus overflow
  detection, and document catalog queries. `retrieval/hybrid.py`: tokenizer BM25 (30)
  and reciprocal rank fusion (k=60, output 30). Scores currently disappear.
- `generation/reranking.py`: generation-model ranking using query-centered excerpts;
  output five candidates, retain the strongest retrieval candidate, silent fallback.
- `generation/prompts.py`: rank-order selection of five original chunks, prompt budget,
  history trimming, untrusted-data instructions. No coverage or duplicate assessment.
- `generation/answers.py`: JSON answer and label membership validation; source definition
  and shortening recovery. Valid labels alone do not establish claim support.
- `generation/corpus.py`: document-ordered maps with source IDs, exact fact deduplication,
  complete extraction or explicit limits; hierarchical summary reduction with provenance.
- `worker.py`, `ingestion/*`, `repositories/ingestion.py`: leases/heartbeat, PDF/TXT/MD
  parsing, structural token/offset chunks, optional context + original search text,
  embedding, staged writes, atomic publication. Chunk metadata already supports neighbors.
- NestJS `conversations.service.ts`: resolves snapshot/history, owns turn persistence,
  validates chunk citations and rechecks active generations when completing the turn.
- `config.py`: environment runtime settings and immutable shared indexing configuration.
- `evals/*`: existing fixture upload/chat evaluation with source-span annotations;
  candidate Recall@30 and reranked Recall@5, but no independent stage rankings.

Objects: document UUID/name/status/active generation; generation UUID/document/status/
embedding config; chunk UUID/generation/ordinal/original/context/search text/pages/heading/
offsets/token count; snapshot list of document-generation pairs; conversation stored in
NestJS with bounded paired history; plan `ChatPlan`; candidate/evidence `EvidenceRecord`
containing `EvidenceChunk`; citations are current S-labels mapped to original chunk IDs.

## Root risks to measure

Wrong route or rewrite; silent lexical corpus sampling; missing raw scores/provenance;
reranker excerpt truncation or fallback; five-candidate truncation before selection;
repetitive evidence, missing boundary context or independent question coverage; evidence
presence mistaken for sufficiency; cited claims whose source does not support them.
These are hypotheses from code inspection, not measured answer-quality findings.

## Ordered slices and success criteria

1. Request trace and typed stage metadata; preserve baseline ranking/depths. Trace IDs,
   scores, ranks, scope, fallbacks, rejection reasons, and timings without source text.
2. Reuse the planner call for standalone rewriting, bounded optional variants, and
   independent subquestions. Preserve exact identifiers and authorized scope.
3. Configurable limits, scored multi-ranking RRF, reranker diagnostics, scoped neighbors,
   duplicate/coverage-aware token-budget selection. Optional retrieval changes stay off
   by default until compared on a labeled dataset.
4. Bounded sufficiency and typed claims, deterministic ID checks, semantic source support
   validation, one repair, deterministic partial/no-evidence responses.
5. Multilingual high-confidence corpus guardrails; preserve complete extraction and
   hierarchical summaries with original-source provenance.
6. Extend the internal evaluation surface and existing harness with every retrieval
   stage, Recall@5/10/20/50, MRR/NDCG, and manual failure JSONL cases.
7. Automated unit/graph/API/contract regressions, strict Python type/lint checks,
   generated OpenAPI/TypeScript contracts, and developer debugging documentation.

Verification: `GENERATION_MODEL=test EMBEDDING_MODEL=test .venv/bin/pytest -q`,
`.venv/bin/ruff check src tests`, `.venv/bin/pyright`, `npm run typecheck`,
`npm test` in backend. Use focused suites after each slice; complete regression at end.

Risks: additional bounded semantic calls increase latency; LLM support judgments remain
fallible; no real-service quality claim without labeled live evaluation. Keep timings,
configuration, and fallback outcomes in reports. Never silently sample exhaustive results.

Stop conditions: authorization expansion, citation provenance loss, ingestion/schema
ownership changes, unbounded model calls, or regressions that cannot be fixed in scope.
