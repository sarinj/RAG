import { z } from "zod";
import { publicConfig } from "@/config/env";
import { ApiError } from "@/lib/http";
import { readServerEvents } from "@/lib/sse";
import type { Turn } from "./api";

const requestId = z.string().uuid();
const citation = z.object({
  label: z.string(), documentId: requestId, generationId: requestId, chunkId: requestId,
  documentName: z.string(), pageStart: z.number().nullable(), pageEnd: z.number().nullable(),
  heading: z.string().nullable(),
});
const message = z.object({
  id: requestId, conversationId: requestId, requestId,
  role: z.enum(["user", "assistant"]), status: z.enum(["pending", "completed", "failed"]),
  content: z.string(), answerStatus: z.enum(["answered", "insufficient_evidence", "direct_answer"]).nullable(),
  citations: z.array(citation), error: z.object({ code: z.string(), message: z.string() }).nullable(),
  createdAt: z.string(), updatedAt: z.string(),
});
const turn = z.object({ requestId, userMessage: message, assistantMessage: message }).refine((value) =>
  value.userMessage.role === "user" && value.assistantMessage.role === "assistant"
  && value.userMessage.requestId === value.requestId && value.assistantMessage.requestId === value.requestId,
);
const progressEvent = z.object({
  type: z.literal("progress"), requestId,
  stage: z.enum([
    "planner", "retrieval", "multi_hop", "corpus_processing", "reranking", "evidence_selection",
    "evidence_sufficiency", "generation", "claim_validation", "answer_review", "repair", "finalization",
  ]),
  status: z.enum(["active", "complete", "skipped"]), message: z.string().min(1).max(512),
}).strict();
const eventSchema = z.discriminatedUnion("type", [
  progressEvent,
  z.object({ type: z.literal("accepted"), requestId }).strict(),
  z.object({ type: z.literal("answer_delta"), requestId, delta: z.string().max(32_768) }).strict(),
  z.object({ type: z.literal("answer_reset"), requestId }).strict(),
  z.object({ type: z.literal("done"), requestId, turn }).strict(),
  z.object({ type: z.literal("error"), requestId, code: z.string(), message: z.string() }).strict(),
]);

export type ChatStreamEvent = z.infer<typeof eventSchema>;
export type ProgressStep = Omit<z.infer<typeof progressEvent>, "type" | "requestId">;
export type StreamReply = {
  status: "active" | "complete" | "error";
  draft: string;
  steps: ProgressStep[];
  latestMessage: string;
  turn: Turn | null;
};
export type StreamMessageInput = {
  requestId: string;
  content: string;
  onEvent?: (event: ChatStreamEvent) => void;
  signal?: AbortSignal;
};

export function newStreamReply(): StreamReply {
  return { status: "active", draft: "", steps: [], latestMessage: "Preparing your answer", turn: null };
}

export function updateStreamReply(previous: StreamReply, event: ChatStreamEvent): StreamReply {
  if (event.type === "answer_reset") return { ...previous, draft: "" };
  if (event.type === "answer_delta") return { ...previous, draft: previous.draft + event.delta };
  if (event.type === "done") {
    const assistant = event.turn.assistantMessage;
    return {
      ...previous,
      status: assistant.status === "pending" ? "active" : assistant.status === "failed" ? "error" : "complete",
      draft: "",
      // Pending replays must not override newer saved history when it completes.
      turn: assistant.status === "pending" ? null : event.turn,
      latestMessage: assistant.status === "pending" ? "Waiting for the saved answer"
        : assistant.status === "failed" ? assistant.error?.message ?? "The answer could not be completed."
          : "Answer saved",
    };
  }
  if (event.type === "error") return { ...previous, status: "error", draft: "", latestMessage: event.message };
  if (event.type === "progress") {
    const step: ProgressStep = { stage: event.stage, status: event.status, message: event.message };
    const exists = previous.steps.some((item) => item.stage === event.stage);
    return {
      ...previous,
      latestMessage: event.message,
      steps: exists ? previous.steps.map((item) => item.stage === step.stage ? step : item) : [...previous.steps, step],
    };
  }
  return previous;
}

export async function sendMessageStream(conversationId: string, input: StreamMessageInput): Promise<Turn> {
  if (!publicConfig.apiBaseUrl) {
    throw new ApiError(publicConfig.apiConfigError ?? "The API is not configured.", "FRONTEND_CONFIGURATION_ERROR", null, null);
  }
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (input.signal?.aborted) abort();
  else input.signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(abort, 150_000);
  let draftLength = 0;
  try {
    const response = await fetch(`${publicConfig.apiBaseUrl}/conversations/${encodeURIComponent(conversationId)}/messages/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream", "X-Request-Id": input.requestId },
      body: JSON.stringify({ requestId: input.requestId, content: input.content }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new ApiError(
        typeof error?.message === "string" ? error.message : "The message could not be sent.",
        typeof error?.code === "string" ? error.code : "REQUEST_FAILED",
        response.status, response.headers.get("X-Request-Id"),
      );
    }
    if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
      throw new ApiError("The API did not return an answer stream.", "REQUEST_FAILED", null, input.requestId);
    }
    for await (const raw of readServerEvents(response.body)) {
      const parsed = eventSchema.safeParse(raw);
      if (!parsed.success || parsed.data.requestId !== input.requestId) {
        throw new ApiError("The API returned an invalid stream event.", "REQUEST_FAILED", null, input.requestId);
      }
      const event = parsed.data;
      if (event.type === "answer_reset") draftLength = 0;
      if (event.type === "answer_delta") {
        draftLength += event.delta.length;
        if (draftLength > 262_144) throw new ApiError("The answer stream exceeded its limit.", "REQUEST_FAILED", null, input.requestId);
      }
      if (event.type === "done" && (event.turn.requestId !== input.requestId
        || event.turn.assistantMessage.conversationId !== conversationId
        || event.turn.userMessage.conversationId !== conversationId)) {
        throw new ApiError("The API returned a mismatched turn.", "REQUEST_FAILED", null, input.requestId);
      }
      input.onEvent?.(event);
      if (event.type === "error") throw new ApiError(event.message, event.code, null, event.requestId);
      if (event.type === "done") return event.turn;
    }
    throw new ApiError("The answer stream ended early. Check saved history before sending again.", "REQUEST_FAILED", null, input.requestId);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("The answer stream was interrupted. Check saved history before sending again.", "REQUEST_FAILED", null, input.requestId);
  } finally {
    clearTimeout(timer);
    controller.abort();
    input.signal?.removeEventListener("abort", abort);
  }
}
