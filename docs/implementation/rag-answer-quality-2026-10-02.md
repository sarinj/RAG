# RAG answer quality evaluation — 2026-10-02

The selected preset improved estimated accuracy from 0.8958 in the previous saved report
to 0.9479, and relevance from 0.8069 to 0.8297. The fresh original-code control scored
0.9375 accuracy and 0.8038 relevance, so the measured accuracy gain over that control is
smaller. All 24 answerable cases answered and all four unknown cases refused. Faithfulness
and lexical F1 did not improve; response latency and API usage increased.

## Selected approach

Use a stronger model for grounded answering and verification, with a bounded review for
missing qualifications and irrelevant details. The original selected sources remain available
to the writer, support checker and reviewer. Generation uses short complete sentences that
name the subject, retain conditions and units, and briefly derive calculated values.

The optional reviewer returns structured feedback, not a replacement answer. Missing facts
must be exact quotations from a selected original, allowing whitespace normalization. Source
labels and irrelevant claim IDs are validated locally. Feedback guides the existing single
repair; the revised answer still passes source membership, citation and claim support checks.
Invalid or unavailable review, an unsupported repair, a repair with more reviewed issues, or
an unavailable follow-up review retains the previously supported draft. Each review has a
20-second limit. Normal traces include counts, statuses, timing and provider usage without
source quotations or answer text.

## Fixed evaluation inputs

The control and candidate runs use the existing retained document and active generation.
No corpus, reference answers, embeddings, chunks or retrieval depths were changed.

| Input | Value |
| --- | --- |
| Baseline code | `9145e6e` (`feat: Improve performance`) |
| Dataset | Harborlight `pipeline_v2`, version 2.0; 24 answerable and four unknown cases |
| Source document | `c45de299-acda-452f-9481-3e7d04b1efe9` |
| Active generation | `c2694b2b-aa2f-48f7-92c3-869a1f04bc56` |
| Normalized corpus | 9,606 bytes, five chunks |
| Normalized corpus SHA-256 | `222847c8b0b8aa6c5c3089c6b338b47939c9b21553487bacf0835d283dc060ea` |
| Combined dataset/corpus fingerprint | `3678f18cec7a2ee714a1fabe2e5108842498b3e8bce72fe91778e5434d3f53ad` |
| Embeddings | `text-embedding-3-small`, 1024 dimensions |
| Chunking | 512-token target, 64-token overlap; existing tokenizer/config |
| Retrieval | Hybrid, 30 vector / 30 lexical / 30 fused candidates; five final chunks |
| Expansion | Query variants and neighbors disabled |
| Context / visible output cap | 8192 / 512 tokens |
| RAGAS judge | `gpt-5-mini`, low reasoning, 2048 completion tokens, concurrency two |
| RAGAS version | 0.4.3 |

The entire five-chunk corpus fits in selected evidence. Recall was 1.0 at every stage in
all comparisons. This isolated remaining failures in answer generation and review; it
does not demonstrate better retrieval on a large corpus.

## Comparison

The previous saved evaluation is the user's historical reference. A fresh control reruns
the original code/configuration before candidate changes. Its higher accuracy illustrates
model/judge variability; both references must remain visible.

| Measurement | Previous saved run, Oct 1 | Fresh original control, Oct 2 | Final quality preset |
| --- | ---: | ---: | ---: |
| RAGAS answer accuracy | 0.8958 | 0.9375 | 0.9479 |
| RAGAS faithfulness | 0.9583 | 1.0000 | 0.9514 |
| RAGAS answer relevance | 0.8069 | 0.8038 | 0.8297 |
| RAGAS context precision | 0.8958 | 0.9167 | 0.8958 |
| RAGAS context recall | 1.0000 | 1.0000 | 1.0000 |
| Answerable lexical token F1 | 0.5117 | 0.5289 | 0.5107 |
| Answerable cases answered | 24/24 | 24/24 | 24/24 |
| Unknown cases correctly refused | 4/4 | 4/4 | 4/4 |
| Request / metric errors | 0 / 0 | 0 / 0 | 0 / 0 |
| Median response latency | 11.81 s | 9.88 s | 17.99 s |
| p95 response latency | 16.71 s | 12.51 s | 56.27 s |

RAGAS grades the frozen answers against their actual selected original chunks. Unknown
cases use refusal outcomes separately; they receive no perfect empty-answer grade.
Answerable mean scores cover 24 cases. Coverage, lexical overlap and citation overlap
are distinct from semantic correctness. Judge scores are estimates, and single runs
cannot establish statistical significance.

## Candidate methods and rejected regressions

Every live candidate answered all 24 answerable cases, refused all four unknown cases
and had zero request errors. These outcomes alone were insufficient to select a method.

| Candidate | Lexical F1 | RAGAS accuracy | RAGAS relevance | Median / p95 |
| --- | ---: | ---: | ---: | --- |
| Fresh nano control | 0.5289 | 0.9375 | 0.8038 | 9.88 / 12.51 s |
| Nano evidence brief and review | 0.5186 | Not graded | Not graded | 17.04 / 33.38 s |
| Mini quality model, brief and full source generation | 0.5280 | 0.8958 | 0.7296 | 13.62 / 22.32 s |
| Mini generation from verified excerpts | 0.4813 | Not graded | Not graded | 13.88 / 21.61 s |
| GPT-5, originals, medium review, terse labeled fragments | 0.6321 | 0.9583 | 0.7245 | 22.20 / 54.21 s |
| Final GPT-5, originals, low review, concise prose/derivations | 0.5107 | 0.9479 | 0.8297 | 17.99 / 56.27 s |

Evidence briefs often classified adjacent discounts or logging instructions as requested
facts. Excerpt generation then repeated those details. The brief/excerpt implementation was
removed. GPT-5 review distinguished qualifications from adjacent details more reliably.
Its intermediate terse fragments lost subjects needed for relevance grading, and bare
computed values confused faithfulness grading. The final wording names the subject and
connects calculations to the original rule and question inputs. Eleven frozen review controls
supported low reasoning: the three old omissions were detected, and irrelevant discount/log
claims were identified without demanding additional neighboring facts.

## Source checks and remaining limitations

The three qualifications omitted from the previous saved run are present in the final answers:

| Case | Qualification verified against the original |
| --- | --- |
| Annual pass price | 420 Harborlight Credits **for one calendar year** |
| Bus-to-tram transfer | 75 minutes; **Airport Express excluded** |
| Signal reset | SIG-882 only after the train is **out of passenger service** and the controller authorizes the reset |

Both requested arithmetic answers include their derivation: two daily passes cost
`2 × 8.40 = 16.80`; an 18:10 alert plus the sourced 25-minute limit gives an 18:35 deadline.
The wheel-alarm answer retains removal until technician inspection. Six additional cases
returned four correct answers and two correct refusals, with no errors: exactly 22 percent
does not trigger the below-22 rule; return at 80 percent also requires a passed diagnostic;
controller approval alone does not permit SIG-882 while the train remains in passenger
service; the verified-student annual price is 336 credits for one calendar year; no depot
street address or Access Desk fax was invented.

Completeness is still imperfect. The normal camera-retention answer gives 30 days but omits
the incident-hold exception. The fire answer includes supervisor reporting details beyond
the reference's immediate actions. One tender answer still had a missing-fact flag after
repair, and the bounded pipeline did not attempt a further correction. Quote validation
proves provenance, not whether a review's suggested qualification is necessary. These are
observable limits of the selected method.

All 28 request-matched traces were captured. There were 26 valid review calls, no invalid or
timed-out reviews, and two repaired requests. Trace outputs identify the answer model as
`gpt-5-2025-08-07`. The six additional cases ran concurrently with part of the final benchmark;
latency is an observed result under that workload, not an isolated speed comparison. The
additional student-price request took 87.64 seconds, illustrating the tail-latency cost.

The final judge gave less than full accuracy to the version comparison (0.75), water
response (0.75), camera retention (0.75) and fire checklist (0.50). Its faithfulness scores
were below one for wheel temperature (0.667), rail maintenance (0.667) and the flood-watch
threshold (0.50). Manual comparison found the requested wheel scenario, maintenance
interval/run and rolling rainfall threshold supported by the original rules. That source
check does not replace the reported judge scores. The two calculations both received
faithfulness 1.0 after the wording change. Overall, this is a measured accuracy/relevance
improvement with remaining omissions and judge variability, not a perfect-quality result.

## Selected runtime settings

```dotenv
GENERATION_MODEL=gpt-5-nano
RAG_QUALITY_MODEL=gpt-5
RAG_ANSWER_REVIEW_ENABLED=true
RAG_ANSWER_REVIEW_MAX_OUTPUT_TOKENS=3072
RAG_ANSWER_REVIEW_REASONING_EFFORT=low
RAG_ANSWER_REASONING_EFFORT=low
RAG_ANSWER_REASONING_TOKEN_BUDGET=2048
```

Planning, reranking and ingestion retain `GENERATION_MODEL`. Sufficiency, grounded
generation, support checks, partial-coverage reconciliation, review and repair use the
quality model. OpenAI GPT-5 answer/repair calls request `verbosity=low`, as documented in
the official [GPT-5 parameter guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools).
Explicit requests for detail still guide the prompt. Readiness validates both configured
generation models. The public chat response and contract version remain unchanged; private
evaluation diagnostics now accept string configuration values for model settings.

Review defaults off and the quality model defaults blank in examples. The evaluated preset
is enabled in this workspace's ignored root `.env` and applied to the local Compose services.
This uses paid GPT-5 calls and adds latency. Review/output reservations also reduce the space
available for evidence on larger documents. Existing indexes require no reindexing for this
answer-stage change. The final traces record 106 answer-stage calls with 289,594 prompt
tokens and 26,351 completion tokens, including 22,720 reasoning tokens. These totals exclude
planner/reranker/embedding calls and RAGAS grading. Large-corpus effectiveness and exact
monetary cost were not measured.

## Verification and artifacts

- Python: 164 tests passed; Ruff passed on source/scripts/tests.
- Focused Pyright: zero errors in changed generation/config/model modules.
- OpenAPI export and generated TypeScript client updated; backend type checking passed.
- Backend contract suite: nine tests passed.
- Compose image rebuilt and local API/worker recreated with the selected settings.
- Final live benchmark: 24/24 answerable responses, 4/4 unknown refusals, no request errors;
  citation membership/overlap precision and recall both 1.0.
- Original-source qualification/arithmetic review and six additional edge cases completed.
- Final frozen-answer RAGAS grading: all 28 cases processed, 24 measured cases per semantic
  metric, zero metric errors; unknown refusals checked separately.

Reports are retained under the ignored `llm-backend/evals/reports/` directory:

- Previous reference: `rag-eval-20261001T082130Z.json` and
  `ragas-final-full/ragas-20261001T082726Z.json`.
- Fresh control: `quality-baseline/rag-eval-20261002T043406Z.json` and
  `quality-baseline-ragas/ragas-20261002T044053Z.json`.
- Candidate captures: `quality-nano/`, `quality-mini/`, `quality-focused/`,
  `quality-strong/` and their completed RAGAS directories.
- Final live capture: `quality-final/rag-eval-20261002T074720Z.json`, matching Markdown and
  `rag-eval-20261002T074720Z-traces.jsonl`.
- Final frozen grading: `quality-final-ragas/ragas-20261002T075809Z.json` and matching
  `ragas-20261002T075809Z-inputs.jsonl` with the actual selected original contexts.
- Additional boundary/refusal cases: `diagnostics/quality-holdout.json`.

Reproduce the live comparison from the repository root after selecting the settings:

```sh
docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --no-sync rag-eval \
  --dataset pipeline_v2 \
  --source pipeline_v2=c45de299-acda-452f-9481-3e7d04b1efe9 \
  --output-dir /app/evals/reports/quality-final

docker compose run --rm --no-deps \
  -v "$PWD/llm-backend/evals/reports:/app/evals/reports" \
  llm-backend uv run --locked --no-default-groups --group eval rag-eval \
  --ragas-report /app/evals/reports/quality-final/rag-eval-20261002T074720Z.json \
  --ragas-judge-model gpt-5-mini \
  --output-dir /app/evals/reports/quality-final-ragas
```

RAGAS dependencies are installed in the disposable evaluator container; production
dependencies stay unchanged. Capture request-matched `rag_trace` events before recreating
the API container to retain repair outcomes, actual model IDs and token usage.
