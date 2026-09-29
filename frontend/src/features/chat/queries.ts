import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { isApiConfigured } from "@/config/env";
import { getConversation, getSource, listConversations, listMessages } from "./api";
import { hasPendingAssistant } from "./polling";

export const chatKeys = {
  all: ["chat"] as const,
  conversations: () => [...chatKeys.all, "conversations"] as const,
  conversationList: () => [...chatKeys.conversations(), { limit: 20 }] as const,
  conversation: (conversationId: string) => [...chatKeys.conversations(), conversationId] as const,
  messages: (conversationId: string) => [...chatKeys.all, "messages", conversationId, { limit: 100 }] as const,
  source: (documentId: string, chunkId: string) => [...chatKeys.all, "source", documentId, chunkId] as const,
};

export function useConversations() {
  return useInfiniteQuery({
    queryKey: chatKeys.conversationList(),
    queryFn: ({ pageParam }) => listConversations({ limit: 20, cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    enabled: isApiConfigured(),
  });
}

export function useConversation(conversationId: string) {
  return useQuery({
    queryKey: chatKeys.conversation(conversationId),
    queryFn: () => getConversation(conversationId),
    enabled: isApiConfigured() && Boolean(conversationId),
  });
}

export function useConversationMessages(conversationId: string) {
  return useInfiniteQuery({
    queryKey: chatKeys.messages(conversationId),
    queryFn: ({ pageParam }) => listMessages(conversationId, { limit: 100, cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    enabled: isApiConfigured() && Boolean(conversationId),
    staleTime: 0,
    refetchInterval: (query) => {
      const messages = query.state.data?.pages.flatMap((page) => page.items) ?? [];
      return hasPendingAssistant(messages) ? 2_000 : false;
    },
    refetchIntervalInBackground: false,
  });
}

export function useSource(documentId: string | null, chunkId: string | null) {
  return useQuery({
    queryKey: documentId && chunkId ? chatKeys.source(documentId, chunkId) : [...chatKeys.all, "source", "closed"],
    queryFn: () => getSource(documentId!, chunkId!),
    enabled: isApiConfigured() && Boolean(documentId) && Boolean(chunkId),
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
