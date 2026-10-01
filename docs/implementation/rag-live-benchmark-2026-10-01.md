# Live RAG benchmark — 2026-10-01

## Setup and scope

Ran the 28-question Harborlight `pipeline_v2` benchmark against the rebuilt local
NestJS/FastAPI/worker/PostgreSQL stack and the configured OpenAI APIs. Generation used
`gpt-5-nano`; embeddings used `text-embedding-3-small` at 1024 dimensions. The corpus
was verified by its SHA-256 and indexed through the actual upload/worker pipeline.
It contains 9,606 bytes and five chunks. There are 24 answerable and four unanswerable
questions, including follow-ups, arithmetic, exact identifiers and conflicting versions.

The runs used hybrid retrieval and the existing 30/30/30 candidate depths with five final
evidence chunks. Query expansion and neighbor expansion remained disabled. No retrieval
depth, embedding, chunking or model selection changed between these runs.

## Initial results before diagnosis fixes

Three complete live runs exposed and localized grounding regressions:

| Run | Answerable questions answered | False abstentions | Unanswerable questions correctly refused | HTTP/metric errors |
| --- | ---: | ---: | ---: | ---: |
| Initial implementation | 0/24 | 24 | 4/4 | 0 |
| Explicit sufficiency/claim JSON schemas | 9/24 | 15 | 4/4 | 0 |
| Coverage enum with derived answerability | 16/24 | 8 | 4/4 | 0 |

These answer counts measure response coverage, not correctness. Manual review found
contradictory missing-information notes in nine partial replies. For example, the Access
Desk reply gives `555-0148` and then says the phone number was not found. The flood-arrival
reply gives the 25-minute rule but does not calculate the requested `18:35` deadline.
Some answers also omit a gold qualification or add unrelated supported facts.

Pre-fix aggregate measurements (`rag-eval-20261001T063346Z.json`):

- Answerable mean lexical token F1: **0.2202**; 3/24 questions reached F1 ≥ 0.5.
- Answerable answer rate: **66.67%**; false abstention rate: **33.33%**.
- Unanswerable abstention accuracy: **100%**.
- All returned answers include valid citations to selected evidence.
- Citation evidence overlap precision/recall: **0.6458 / 0.6667**, averaged over the
  answerable questions, including zero citation coverage for refusals.
- Median latency: **8.26 seconds**; p95: **13.21 seconds**.

| Stage | Recall@5/10/20/50 | MRR | Source coverage |
| --- | ---: | ---: | ---: |
| Vector | 1.0 | 0.8507 | 1.0 |
| Lexical | 1.0 | 0.9792 | 1.0 |
| Hybrid/RRF | 1.0 | 0.8889 | 1.0 |
| Reranked | 1.0 | 0.8958 | 1.0 |
| Selected evidence | 1.0 | 0.8958 | 1.0 |

The entire corpus fits in the five-chunk evidence set. Recall is saturated on this
fixture; these results cannot establish that candidate depths, expansion, neighbors or
reranking improve retrieval over a larger corpus. F1 is lexical overlap, and citation
overlap is not semantic claim verification. Zero request errors is not a quality pass.

The previous saved 2026-09-30 report used the same corpus/model identifiers and reported
F1 0.4185, four request errors, 75% unanswerable abstention accuracy and 2.86-second median
latency. It is a historical reference, not a controlled code/configuration comparison.
Those initial results did **not** establish an overall answer-quality improvement over it.

## Initial schema corrections

The expected source facts reached selected evidence for all answerable cases. The failures
occurred during sufficiency, generation or claim validation.

1. Claim validation expected `{"verdicts":[{"id":1,"status":"supported"}]}`, but real
   provider outputs included `claims_verdicts` or a bare verdict. Added the exact typed
   response schema and explicit array/field requirements to the prompt. Invalid outputs
   remain rejected, and one bounded answer repair remains available.
2. The sufficiency model returned contradictory `answerable` and `coverage` values.
   It now returns coverage and missing aspects. Application code derives answerability
   deterministically from the enum, with the existing coverage invariants retained.
3. Added redacted schema error types/locations to traces. Unexpected field names and
   source/model text are not logged. Explicitly described supported negative answers
   and arithmetic in the grounding instructions.
4. Aligned answer-repair instructions with the structured claim format. Reports now
   count false abstentions separately from request/metric errors. Abstentions receive
   zero answer F1 rather than keyword-overlap credit from refusal boilerplate.

Eight answerable questions still abstained at that point:

| Cases | Observed blocking stage |
| --- | --- |
| `route-holiday-followup`, `procurement-mid-tier`, `camera-retention` | Sufficiency judged the selected evidence insufficient. |
| `maintenance-two-part`, `flood-close-threshold` | Sufficiency returned an additional field rejected by its schema. |
| `signal-checklist-code`, `signal-near-match`, `depot-h4-stock` | Claim support checks rejected one or more claims after bounded repair. |

The subsequent diagnosis inspected the original claim/source pairs and reproduced false
support judgments, including rejection of a verbatim source paragraph. Nine partial replies
also had missing-aspect metadata that contradicted supplied facts. This located the dominant
failures after evidence selection, before considering any retrieval tuning.

## Confirmed causes and targeted fixes

Frozen probes held prompts, original passages, claim IDs and selected chunks fixed while
varying one provider parameter at a time. Diagnostic captures use only the public synthetic
fixture and stay in the ignored reports directory.

1. **Reasoning effort was hardcoded to `minimal` for every `gpt-5-nano` call.** The
   sufficiency checker overlooked explicitly supplied phone numbers, policy durations,
   procurement requirements and supported negative answers. The verifier rejected correct
   cited claims. Setting grounding effort to `low` corrected the captured failures without
   changing the application model, embeddings or retrieval settings.
2. **JSON object mode did not enforce the typed schema.** Native Structured Outputs now
   enforce coverage, verdict and grounded-answer schemas. Local validation still rejects
   invalid UUIDs and out-of-evidence sources. This follows the distinction in OpenAI's
   [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
3. **Hidden reasoning exhausted small completion budgets.** Low-effort sufficiency calls
   could finish at the old 256-token cap without valid JSON; one answer exhausted 512
   reasoning tokens before producing visible content. Grounding now has a bounded 1024-token
   total completion budget. Answer generation gets a 512-token reasoning reserve while
   retaining the existing 512-token visible output cap. Both reserves count against the
   context window. Traces report usage and finish reasons.
4. **Unverified missing-aspect text was appended after claim validation.** A supported
   partial draft now gets one bounded coverage reconciliation. Only verified missing
   aspects produce a specific note, scoped to the supplied evidence. Incorrect absence
   notes no longer appear in the completed post-fix runs.
5. **Generation repeated calculation rules instead of applying them.** Low-effort answer
   reasoning and clearer instructions produce the requested deadline. The claim checker
   receives hypothetical inputs from the user question without treating them as document
   facts. Frozen controls accepted six correct deadline calculations and rejected three
   incorrect ones. Repeated captures also exposed a correct deadline claim citing a neighboring
   chunk without the 25-minute rule; the validator properly rejected that source assignment.
   Generation now uses short source labels mapped deterministically to selected UUIDs before
   all scope and support checks. Old UUID drafts remain readable. This reduces reference
   copying complexity; it does not make a wrong source choice acceptable.
6. **A factual discount question was routed to the source catalog.** Conservative guards
   now permit direct social/catalog replies only for matching requests. A factual question
   mislabeled as either direct intent returns to scoped QA. English and common multilingual
   social/catalog forms are covered; unfamiliar direct phrasing may conservatively use RAG.
7. **Repair lacked actionable rejection feedback.** Its single attempt now receives the
   rejected draft, claim IDs and source references privately. Full claim text stays out of
   normal trace logs. Review reproduced an unsafe literal-paragraph shortcut that could omit
   a surrounding supersession notice. Literal validation now accepts only a complete unchanged
   original passage; individual paragraphs, changed quantities and paraphrases still need
   semantic checking with their source context.

### Live fix stages

All rows use the same retained document/generation, model IDs, five indexed chunks and
retrieval depths. The context reservation changed to accommodate hidden reasoning; all
five chunks still fitted in selected evidence.

| Report time (UTC) | Change since prior stage | Answered / answerable | False abstentions | Wrong direct routes | Lexical F1 | Unanswerable abstention |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06:33:46 | Pre-fix reference | 16/24 | 8 | 0 | 0.2202 | 4/4 |
| 07:33:22 | Native schemas, grounding effort/budgets, coverage reconciliation | 24/24 | 0 | 0 | 0.4584 | 4/4 |
| 07:40:47 | Answer reasoning reserve, calculations and repair feedback | 23/24 | 0 | 1 | 0.5040 | 4/4 |
| 07:48:24 | Direct-intent guard | 23/24 | 1 | 0 | 0.5444 | 4/4 |
| 07:58:42 | Repeat of guard configuration, final output-cap checks | 24/24 | 0 | 0 | 0.5335 | 4/4 |
| 08:05:32 | Short-label claim references with deterministic UUID resolution | 24/24 | 0 | 0 | 0.5225 | 4/4 |
| 08:21:30 | Final combined patch, qualification checks and large-corpus schema guard | 24/24 | 0 | 0 | 0.5117 | 4/4 |

These post-guard runs had zero request errors and source Recall@5/10/20/50 of 1.0 at every stage.
The 07:48 run's median/p95 latency was 11.45/15.87 seconds, versus 8.26/13.21 before the fixes. The
extra reasoning improves these measured failures at a latency cost. Answered counts and
lexical F1 still do not establish semantic accuracy. Manual review found the requested
core facts in the 23 returned answers, but some policy qualifications were omitted and
the fire checklist answer included additional supported details.

The short-label run also had zero repairs and no false abstentions. Its median/p95 was
10.95/18.21 seconds. The previous guard repeat had median/p95 12.39/20.45 seconds and no
repairs either; these single-run timing differences do not establish a speed gain from
labels. The short-label run explicitly resolved the 60/75-minute version conflict and
computed 18:35. Its transfer answer still omitted the Airport Express exclusion, its
annual price omitted the calendar-year qualification, and several replies were verbose.

Qualification controls then exposed a literal-paragraph shortcut and a semantic version
ambiguity. The final checker reads surrounding source text, names and section metadata
and evaluates each claim in the question's time/version context. Live controls reject an
obsolete 15-day rule as current, accept its explicit 30-day replacement, and accept an
explicitly historical 15-day statement. Large corpus schemas also avoid the native
1000-value enum limit while retaining deterministic label/source validation, as specified
in the [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

The final combined run answered 24/24 answerable cases, refused all four unanswerable
cases and needed no answer repairs. Median/p95 latency was **11.81/16.71 seconds**.
Manual review confirmed the requested core facts and the 18:35 calculation, with no
contradictory missing-information notes. Remaining completeness issues include omission
of the Airport Express transfer exclusion, the annual pass's calendar-year qualification
and the signal-reset prerequisite that the train be removed from service. These are
saved as three generation review cases; an answered response with valid citations does
not establish that it retained every policy qualification.

## Optional RAGAS evaluation

Added an optional `eval` dependency group and CLI adapter. Production has no RAGAS imports
or new service. Evaluations use frozen answers and **exact selected original chunk text**,
never the larger candidate set. An explicit export produces ignored input JSONL for replay
without the database. Standard trace logging remains free of source text.

The full pre-fix run used RAGAS 0.4.3 and an independent `gpt-5-mini` judge with low reasoning,
2048 completion tokens, concurrency two and one provider retry. It completed all 28 cases
with zero metric errors:

The same judge/settings scored the 07:48:24 direct-guard run, also completing 28 cases with
zero metric errors. That run still contained one false abstention. The final combined
08:21:30 run was scored separately with the same five metrics, judge and completion budget;
it also completed all 28 cases with zero metric errors:

| Metric | Before | After direct guard | Final combined patch | Measured cases |
| --- | ---: | ---: | ---: | --- |
| Faithfulness | 0.7650 | 0.9638 | 0.9583 | 16 before / 23 guard / 24 final returned answers |
| Answer accuracy | 0.5729 | 0.9167 | 0.8958 | 24 answerable cases, with false abstentions scored zero |
| Answer relevancy | 0.4849 | 0.7410 | 0.8069 | 24 answerable cases |
| Context precision | 0.8958 | 0.8958 | 0.8958 | 24 answerable cases |
| Context recall | 1.0000 | 1.0000 | 1.0000 | 24 answerable cases |

Four unanswerable cases were correctly refused and scored separately. Faithfulness is
conditional on returned answers; it must be interpreted with false-abstention counts.
Question-aware answer accuracy is the default semantic comparison. Optional
`factual_correctness` can misjudge short gold fragments that omit the question's subject.
The judge also credited a baseline reply that quoted the 25-minute rule without computing
the requested deadline, so even semantic judge scores need manual checks.
The final accuracy mean is lower than the intermediate guard run despite eliminating its
false abstention. Qualification omissions, extra details and judge variability remain;
the sequence is not a monotonic quality gain. The final judge scored the transfer and
signal-checklist answers 0.5. It also scored the wheel-alarm answer 0.5 despite its correct
removal/inspection rule, illustrating why individual grades need source-aware review.
All 28 exported final inputs were checked against the report's selected chunk IDs.

Run from the repository root with the benchmark generation retained:

```sh
docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --locked --no-default-groups --group eval rag-eval \
  --ragas-report /app/evals/reports/rag-eval-20261001T082130Z.json \
  --ragas-judge-model gpt-5-mini
```

For the pre-fix report, also supply its exact `--ragas-trace-file`. New reports include
selected stage IDs. See [evaluation usage](../../llm-backend/README.md#optional-ragas-evaluation)
for metric selection, saved-input replay and API requirements. Metric errors cause a
nonzero exit; they are never silently converted to zero or perfect scores.

## Artifacts and replay

- [Initial JSON report](../../llm-backend/evals/reports/rag-eval-20261001T061153Z.json)
- [Schema correction JSON report](../../llm-backend/evals/reports/rag-eval-20261001T062612Z.json)
- [Pre-fix JSON report](../../llm-backend/evals/reports/rag-eval-20261001T063346Z.json)
- [Pre-fix Markdown report](../../llm-backend/evals/reports/rag-eval-20261001T063346Z.md)
- [28 redacted request traces](../../llm-backend/evals/reports/rag-eval-20261001T063346Z-traces.jsonl)
- [17 failure/review cases](../../llm-backend/evals/reports/rag-eval-20261001T063346Z-failures.jsonl)
- [Grounding fix report](../../llm-backend/evals/reports/rag-eval-20261001T073322Z.json)
- [Answer reasoning fix report](../../llm-backend/evals/reports/rag-eval-20261001T074047Z.json)
- [Direct-routing guard report](../../llm-backend/evals/reports/rag-eval-20261001T074824Z.json)
- [28 guard-run traces](../../llm-backend/evals/reports/rag-eval-20261001T074824Z-traces.jsonl)
- [Guard repeat report](../../llm-backend/evals/reports/rag-eval-20261001T075842Z.json)
- [Short-label report](../../llm-backend/evals/reports/rag-eval-20261001T080532Z.json)
- [Final combined live report](../../llm-backend/evals/reports/rag-eval-20261001T082130Z.json)
- [28 final redacted traces](../../llm-backend/evals/reports/rag-eval-20261001T082130Z-traces.jsonl)
- [Three remaining generation review cases](../../llm-backend/evals/reports/rag-eval-20261001T082130Z-failures.jsonl)
- [RAGAS pre-fix report](../../llm-backend/evals/reports/ragas-before-full/ragas-20261001T074454Z.json)
- [RAGAS direct-guard report](../../llm-backend/evals/reports/ragas-after-full/ragas-20261001T075842Z.json)
- [Final RAGAS report](../../llm-backend/evals/reports/ragas-final-full/ragas-20261001T082726Z.json)
- [Frozen final RAGAS inputs](../../llm-backend/evals/reports/ragas-final-full/ragas-20261001T082726Z-inputs.jsonl)

The failure file includes eight refusals and nine conflicting partial-coverage replies,
with gold chunk IDs, generation pins, history, observed stage IDs and investigation notes.
Replay confirmed that all gold chunks remain selected for these 17 cases. Automatic
retrieval failure hints return `unknown` because no retrieval-stage loss was observed.
The three final generation review cases also replayed with complete source coverage at
every stage. Their manual `generation` category records omitted qualifications; the
automatic retrieval hint remains `unknown`, which correctly avoids blaming retrieval.

The benchmark source created for diagnosis is kept ready for replay:
`c45de299-acda-452f-9481-3e7d04b1efe9`, generation
`c2694b2b-aa2f-48f7-92c3-869a1f04bc56`. Reports/traces are local generated artifacts
under the ignored `llm-backend/evals/reports/` directory.

Run the full live benchmark from the repository root with the stack ready:

```sh
docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --no-sync rag-eval --dataset pipeline_v2
```

Replay the saved retrieval rankings without providers:

```sh
cd llm-backend
uv run rag-eval \
  --retrieval-cases evals/reports/rag-eval-20261001T063346Z-failures.jsonl \
  --offline --output-dir evals/reports/failure-replay
```

## Changes and configuration

| Files | Responsibility |
| --- | --- |
| `generation/grounding.py`, `generation/chat.py`, `models.py` | Typed coverage/verdicts, bounded reasoning, partial reconciliation, qualification checks and repair feedback |
| `generation/answers.py`, `generation/prompts.py` | Short model-facing source references resolved to authorized original chunk IDs |
| `generation/intents.py`, `generation/query_rewriting.py` | Direct-intent guards for factual requests |
| `providers/models.py`, `providers/openai.py`, `retrieval/tracing.py` | Native response schemas, reasoning budgets/usage and redacted diagnostics |
| `evals/runner.py`, new `evals/ragas.py`, `pyproject.toml`, `uv.lock` | Exact stage/source export, optional RAGAS evaluation and explicit false-abstention metrics |
| `config.py`, `.env.example`, root `compose.yaml` | Four new bounded settings |
| `tests/unit/`, backend README and implementation documentation | Regression coverage and replay/debugging instructions |

Python module paths above are under `llm-backend/src/rag_app/`.

| New setting | Default | Meaning |
| --- | --- | --- |
| `RAG_GROUNDING_REASONING_EFFORT` | `low` | Sufficiency and semantic claim verification effort |
| `RAG_GROUNDING_MAX_OUTPUT_TOKENS` | `1024` | Grounding's total completion budget, including hidden reasoning |
| `RAG_ANSWER_REASONING_EFFORT` | `low` | Grounded answer and bounded repair effort |
| `RAG_ANSWER_REASONING_TOKEN_BUDGET` | `512` | Extra completion room for reasoning; the visible answer cap remains 512 |

The [RAG correctness guide](rag-correctness.md) documents the architecture, remaining
retrieval settings and debugging decision tree. For one failure, correlate the request ID
across its report and trace, check the route and rewritten query, then follow the expected
chunk through vector/lexical retrieval, fusion, reranking and selected evidence. If it
survives, inspect sufficiency, claim verdicts, model finish reasons and token usage.
Keep the failed case with its generation pins before changing one variable and replaying.

## Verification and limits

**148 Python tests passed**, Ruff passed, and regenerated FastAPI OpenAPI was unchanged.
The nine NestJS contract tests, TypeScript checks and build passed. Full Pyright still
fails with 112 pre-existing provider/evaluation diagnostics (118 before these fixes),
with no new diagnostic messages. Added tests cover native schemas, bounded reasoning
reserves, literal versus qualified claims, hypothetical calculation inputs, reconciled
missing aspects, rejection feedback, direct-routing guards, visible exhaustive-output
budgets, short-label authorization/normalization, surrounding version qualifications,
large-corpus schema limits and exact RAGAS context/abstention/error handling.
All 28 final traces were checked for source-text redaction and private feedback exclusion.
All 76 recorded grounding/generation completions ended normally, with no truncation.

Run the Python suite from `llm-backend`:

```sh
GENERATION_MODEL=generation-test EMBEDDING_MODEL=embedding-test uv run pytest -q
uv run ruff check src tests
```

This is a small synthetic **live integration benchmark**, not a representative production
dataset or a public benchmark ranking. All five chunks fit in context, so retrieval metrics
are saturated. Add a held-out dataset of actual failures and a larger corpus before tuning
candidate depths or claiming production-wide accuracy. Models and LLM judges can still
produce intermittent failures; preserve per-request traces and report repeated runs.
