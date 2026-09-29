import { http, type Page } from "@/lib/http";
import type { components } from "@/generated/api";
import { serializeScope } from "./scope";

type Schemas = components["schemas"];
type CitationLocation = { pageStart: number | null; pageEnd: number | null; heading: string | null };

export type Scope = Schemas["CreateConversationDto"]["scope"];
// The current OpenAPI schema marks nullable scalar citation fields as empty objects.
export type Citation = Omit<Schemas["CitationDto"], keyof CitationLocation> & CitationLocation;
export type ConversationRecord = Omit<Schemas["ConversationDto"], "scope"> & { scope: Scope };
export type MessageRecord = Omit<Schemas["MessageDto"], "citations"> & { citations: Citation[] };
export type Turn = Schemas["TurnDto"];
export type SourceRecord = Omit<Schemas["SourceDto"], keyof CitationLocation> & CitationLocation;

export async function listConversations(params: { limit: number; cursor: string | null }) {
  const response = await http.get<Page<ConversationRecord>>("/conversations", { params });
  return response.data;
}

export async function createConversation(scope: Scope) {
  const response = await http.post<ConversationRecord>("/conversations", serializeScope(scope));
  return response.data;
}

export async function getConversation(conversationId: string) {
  const response = await http.get<ConversationRecord>(`/conversations/${encodeURIComponent(conversationId)}`);
  return response.data;
}

export async function listMessages(conversationId: string, params: { limit: number; cursor: string | null }) {
  const response = await http.get<Page<MessageRecord>>(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    { params },
  );
  return response.data;
}

export async function sendMessage(conversationId: string, input: { requestId: string; content: string }) {
  const response = await http.post<Turn>(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    input,
  );
  return response.data;
}

export async function getSource(documentId: string, chunkId: string) {
  const response = await http.get<SourceRecord>(
    `/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}`,
  );
  return response.data;
}
