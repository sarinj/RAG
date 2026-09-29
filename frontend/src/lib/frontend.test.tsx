import assert from "node:assert/strict";
import { test } from "node:test";
import AxiosMockAdapter from "axios-mock-adapter";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ApiError, http } from "./http";
import { hasPendingAssistant } from "@/features/chat/polling";
import { createTurnRequest } from "@/features/chat/request-id";
import { serializeScope } from "@/features/chat/scope";
import { sendMessage } from "@/features/chat/api";
import { SourceErrorNotice } from "@/features/chat/components/source-error";
import { hasActiveIngestionJob } from "@/features/documents/polling";

test("selected document scope stays explicit and rejects empty or duplicate selections", () => {
  assert.deepEqual(serializeScope({ mode: "all" }), { scope: { mode: "all" } });
  assert.deepEqual(serializeScope({ mode: "selected", documentIds: ["doc-a", "doc-b"] }), {
    scope: { mode: "selected", documentIds: ["doc-a", "doc-b"] },
  });
  assert.throws(() => serializeScope({ mode: "selected", documentIds: [] }), /1 and 100 unique/);
  assert.throws(() => serializeScope({ mode: "selected", documentIds: ["doc-a", "doc-a"] }), /1 and 100 unique/);
});

test("each intentional send gets a UUID while a transport retry can reuse its original payload", () => {
  const request = createTurnRequest("What does the report say?");
  const transportRetry = request;
  assert.match(request.requestId, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  assert.equal(transportRetry.requestId, request.requestId);
  assert.equal(transportRetry.content, request.content);
  assert.notEqual(createTurnRequest(request.content).requestId, request.requestId);
});

test("message transport retries keep the same body request ID", async () => {
  const mock = new AxiosMockAdapter(http);
  const sentIds: string[] = [];
  const endpoint = "/conversations/conversation-id/messages";
  mock.onPost(endpoint).replyOnce((config) => {
    sentIds.push(JSON.parse(String(config.data)).requestId as string);
    return [503, { code: "MODEL_UNAVAILABLE", message: "Unavailable", requestId: "http-attempt-1" }];
  });
  mock.onPost(endpoint).reply((config) => {
    sentIds.push(JSON.parse(String(config.data)).requestId as string);
    return [200, { requestId: sentIds[1], userMessage: {}, assistantMessage: {} }];
  });

  const request = createTurnRequest("What does the report say?");
  try {
    await assert.rejects(sendMessage("conversation-id", request), ApiError);
    await sendMessage("conversation-id", request);
    assert.deepEqual(sentIds, [request.requestId, request.requestId]);
  } finally {
    mock.restore();
  }
});

test("document polling only continues for queued or processing jobs", () => {
  assert.equal(hasActiveIngestionJob([{ latestJob: { status: "queued" } }]), true);
  assert.equal(hasActiveIngestionJob([{ latestJob: { status: "processing" } }]), true);
  assert.equal(hasActiveIngestionJob([{ latestJob: { status: "succeeded" } }]), false);
  assert.equal(hasActiveIngestionJob([{ latestJob: { status: "failed" } }]), false);
  assert.equal(hasActiveIngestionJob([{ latestJob: null }]), false);
});

test("message polling only continues for a pending assistant", () => {
  assert.equal(hasPendingAssistant([{ role: "assistant", status: "pending" }]), true);
  assert.equal(hasPendingAssistant([{ role: "user", status: "pending" }]), false);
  assert.equal(hasPendingAssistant([{ role: "assistant", status: "completed" }]), false);
  assert.equal(hasPendingAssistant([{ role: "assistant", status: "failed" }]), false);
});

test("a deleted citation source renders an explicit retained-answer message", () => {
  const markup = renderToStaticMarkup(<SourceErrorNotice error={new ApiError("gone", "SOURCE_DELETED", 410, "req-1")} />);
  assert.match(markup, /Source deleted/);
  assert.match(markup, /saved answer remains in the conversation/);
  assert.match(markup, /role="alert"/);
});
