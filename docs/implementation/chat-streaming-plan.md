# Chat streaming implementation packet

Goal: Stream answers into chat and show a concise, visible account of source retrieval,
writing and validation while the answer is being prepared.

Constraints: Keep the existing scoped, idempotent turn lifecycle, source/citation checks,
answer review and one repair. Save only the final checked reply. Expose public progress
summaries and answer text, never private reasoning, prompts, review payloads or raw JSON.
Keep ordinary REST chat and evaluation compatible. Do not add or run test suites unless asked.

Files: Python provider/composite, generation progress and internal SSE route; NestJS stream
reader, LLM gateway and conversation stream endpoint; React stream client, active reply and
progress panel; generated contracts and API documentation.

Success criteria:

- The configured OpenAI provider sends genuine content deltas during generation.
- Only answer text is decoded from partial structured JSON; hidden reasoning is ignored.
- Progress corresponds to actual pipeline stages, with source counts after selection.
- The provisional reply is marked as a draft and replaced by the final saved response.
- Repair can replace the draft without appending a second answer.
- Disconnects retain the accepted turn's bounded generation and history reconciliation;
  duplicates do not start another generation. Stream errors cannot masquerade as completion.
- Scope/citation validation and source changes still fail the saved turn safely.
- Chat displays the progress log during generation and can inspect it after completion
  within the current chat session.

Decisions: Add POST SSE endpoints through both services. Use native fetch streaming in the
browser (POST and custom request headers), not EventSource auto-retries. Retain the current
model API and structured schemas. Only answer-writing and repair calls stream; planning and
verification remain private. Other generation providers retain buffered answers and live
stage updates. Progress logs are transient UI state; final text/citations remain durable.

Steps: Implement scoped events and provider streaming; add Python/NestJS SSE transport;
connect frontend draft/progress UI; regenerate contracts; run type checks, lint and builds;
rebuild services and inspect operational readiness. No benchmark rerun is needed for this
transport/UI change unless requested.

Risks: Partial JSON/UTF-8 boundaries, truncated streams, slow consumers, duplicate sends,
disconnects, unmounts, repairs and draft/final race conditions. Bound event buffers and use
request IDs, explicit final events, existing durable history and cancellation cleanup.

References: [OpenAI streaming](https://developers.openai.com/api/docs/guides/streaming-responses),
[Pydantic partial JSON](https://docs.pydantic.dev/latest/concepts/json/),
[SSE framing](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).

## Completed implementation — 2026-10-02

Implemented the provider stream, both SSE routes, strict event consumers, live draft
and expandable progress panel. Added the [streaming contract](../contracts/chat-streaming.md)
and regenerated both OpenAPI exports and their consumer types. Disconnects retain
accepted turns; bounded drain waits release stalled viewers. Pending duplicate replays
cannot override later completed history. Final text remains visible while history refreshes.

Checks completed:

- Backend and frontend TypeScript checks passed.
- Pyright reported zero errors on all six changed Python source files.
- Ruff passed on those files; frontend ESLint passed.
- NestJS OpenAPI export/build and the production frontend/API/model Docker builds passed.
- Recreated backend, frontend, model API and worker; retained database and upload volumes.
- Python and NestJS readiness returned HTTP 200; frontend returned HTTP 200.
- Both running services publish the new SSE endpoints in their OpenAPI metadata.
- No test suites or live chat/answer-quality evaluations were run.

Implementation review covered source/privacy boundaries, provisional-to-final replacement,
request IDs, duplicate requests, partial/truncated frames, cancellation and backpressure.
Runtime stream behavior still needs a live chat exercise if end-to-end verification is requested.
