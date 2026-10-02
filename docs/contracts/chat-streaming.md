# Chat streaming

Streaming extends the [v1 chat contract](v1.md) with two additive POST endpoints.
Existing REST chat and evaluation endpoints retain their request and response shapes.
The chat UI uses streaming automatically; no new environment setting or migration is needed.

## Visible behavior

Chat shows live source retrieval, evidence selection, writing, source checks, optional
answer review and repair. The expandable panel summarizes work that actually ran,
including selected passage/document counts. These are public process summaries.
Private model reasoning, prompts, planner output and reviewer feedback are never sent.

The configured OpenAI adapter streams answer content from chat-completion deltas.
Only public answer fields are decoded from partial structured JSON. Other adapters
retain buffered answer generation while reporting the same live stages. Direct replies
and insufficient-evidence replies can finish without an answer-writing stream.

Streamed answer text is provisional and marked **Draft · checking sources**. A repair
resets that draft. The final event replaces it with the saved answer and clickable
citations, including cases where a checked earlier draft is retained after review.
Only the checked answer is saved. Progress logs remain in the current chat session
for the latest 50 streamed requests; reloading keeps the answers and citations but
clears those logs.

## Public endpoint

`POST /api/conversations/:id/messages/stream` accepts the existing
`{requestId, content}` body. The request ID remains the durable idempotency key.
Use `Accept: text/event-stream` and `Content-Type: application/json`.

Successful transport uses HTTP 200 and UTF-8 SSE. Each data frame is one JSON
object with `type` and the body `requestId`. Comments provide 10-second keepalives.
Validation/scope failures before accepting a new turn use ordinary HTTP error JSON.
Failures after headers are sent use a terminal `error` frame.

| Type | Additional fields | Meaning |
| --- | --- | --- |
| `accepted` | none | The new pending turn has been persisted. |
| `progress` | `stage`, `status`, `message` | Public stage update; status is `active`, `complete` or `skipped`. |
| `answer_reset` | none | Clear the provisional answer before writing or repair. |
| `answer_delta` | `delta` | Append provisional answer text; no final citation metadata yet. |
| `done` | `turn` | Authoritative saved `Turn` in the existing public shape. |
| `error` | `code`, `message` | Terminal safe error; reconcile saved history. |

Stages are `planner`, `retrieval`, `multi_hop`, `corpus_processing`, `reranking`,
`evidence_selection`, `evidence_sufficiency`, `generation`, `claim_validation`,
`answer_review`, `repair` and `finalization`. Repeated stages update the summary.
Skipped optional stages need not appear.

A duplicate request replays its existing turn without starting generation again.
Its `done.turn.assistantMessage.status` can still be `pending` or `failed`; inspect
that status rather than treating every `done` as a completed answer. Poll ordinary
saved history for pending replays.

Disconnecting the viewer leaves accepted generation running within the existing
deadline so the answer can be saved. Browser navigation cancels local consumption.
A connection that cannot drain within 10 seconds is closed without cancelling the
accepted turn. Before sending another attempt after an interruption, reconcile with
`GET /api/conversations/:id/messages`. Streams do not reconnect or retry automatically.

## Internal endpoint

`POST /internal/chat/stream` accepts the existing authenticated `ChatRequest` with
contract version `1.1.0`. It emits the same progress/reset/delta events, followed by
`result {response: ChatResponse}` after Python's scope and citation checks, or
`error {code, message, status}`. It never persists chat history. NestJS validates the
result, original chunks and current source snapshot, then persists before public `done`.

The internal connection has NestJS's 130-second deadline, within the browser's
150-second deadline; Python retains its 120-second generation deadline. Internal
disconnects cancel request work and clean up its bounded queue and provider stream.
Consumers require an explicit terminal result; EOF is never implicit success.
Frames and provisional text are bounded to 262,144 characters, and deltas to 32,768
characters. UTF-8 decoding and SSE framing handle split network chunks.

Runtime schemas live in the backend's `stream.contract.ts` and frontend's
`streaming.ts`. OpenAPI records each SSE transport as a string and this document
defines its event payloads; normal JSON request/response types remain generated.

## Implementation checks

The implementation uses type checks, scoped Python lint, frontend lint and production
builds. No test suites or live answer-quality evaluations were run for this change.
See the [implementation packet](../implementation/chat-streaming-plan.md).
