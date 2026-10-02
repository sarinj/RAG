# RAG answer quality implementation packet

Goal: Improve semantic accuracy, completeness and relevance in the latest Harborlight
evaluation. Target missing policy qualifications and unnecessary adjacent facts, then
measure against the existing live benchmark and the same RAGAS grading configuration.

Constraints: Retain the source document and active generation, embeddings, indexed chunks,
retrieval depths, public response contract and bounded request deadline. Keep original
sources available to generation and support validation. Do not edit reference answers.
Use general source reasoning and fictional calibration examples with different facts.

Files likely involved: Python generation/grounding, runtime settings, Compose pass-through,
evaluation reporting, focused regression tests and implementation documentation.

Success criteria:

- An optional quality model handles sufficiency, grounded answers, support, review and repair.
  Planning, reranking and ingestion retain the base model.
- A typed optional review catches missing supported qualifications and unrelated answer details.
  Exact original quotations and valid claim IDs are required for feedback.
- At most one answer repair; repaired answers still pass source membership and claim support.
- Invalid optional guidance cannot discard an already validated answer.
- The known transfer exclusion, calendar-year duration and signal-reset prerequisites appear.
- Full live evaluation preserves answer coverage and unanswerable refusal behavior; report
  semantic accuracy, faithfulness, relevance, latency and unresolved limitations honestly.

Investigation findings: Retrieval recall was already saturated on this five-chunk fixture.
Nano evidence briefs and mini quality passes did not improve the live comparison. Generating
from validated excerpts also reduced lexical overlap and encouraged extra details. Those
brief/excerpt paths were removed. GPT-5 using original sources improved completeness and
accuracy; overly terse labeled fragments then hurt relevance and calculation grading, so
the final wording uses concise prose and a small calculation derivation. The evaluated local
preset uses GPT-5 with low reasoning, a 3072-token review budget and a 2048-token answer
reasoning reserve. Each review is limited to 20 seconds; review defaults off.

Ordered steps:

1. Freeze the previous report and inspect source-aware failures.
2. Add optional typed review with deterministic original-source quote validation.
3. Add bounded answer review, model selection and repair feedback, preserving a supported draft.
4. Run focused regressions and the live benchmark; compare the same RAGAS judge's scores.
5. Review the diff and document final settings, measured results and tradeoffs in
   [the evaluation report](rag-answer-quality-2026-10-02.md).

Verification commands: focused pytest; Ruff on changed source/tests; Pyright on changed
modules; Compose rebuild; `rag-eval --dataset pipeline_v2`; RAGAS with gpt-5-mini and the
same metrics/settings as the prior run. Repeat only for a new change or unresolved issue.

Risks: More API calls, stronger-model cost, response latency, variable review judgments and
smaller evidence capacity from reserved review tokens. This fixture cannot establish retrieval
quality on a large corpus. Review feedback is guidance, never evidence. Trace statuses/counts
and provider usage; keep source text private.

Stop conditions: Do not claim improved quality without live measurement. If a change
reduces accuracy, coverage or unanswerable refusal, diagnose and revise it before shipping.
