import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { isApiConfigured } from "@/config/env";
import { getDocument, listDocuments } from "./api";
import { hasActiveIngestionJob } from "./polling";

export const documentKeys = {
  all: ["documents"] as const,
  lists: () => [...documentKeys.all, "list"] as const,
  list: () => [...documentKeys.lists(), { limit: 20 }] as const,
  detail: (documentId: string) => [...documentKeys.all, "detail", documentId] as const,
};

export function useDocuments() {
  return useInfiniteQuery({
    queryKey: documentKeys.list(),
    queryFn: ({ pageParam }) => listDocuments({ limit: 20, cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    enabled: isApiConfigured(),
    staleTime: 0,
    refetchInterval: (query) => {
      const documents = query.state.data?.pages.flatMap((page) => page.items) ?? [];
      if (!hasActiveIngestionJob(documents)) return false;
      const embedding = documents.some(
        (document) => document.latestJob?.status === "processing" && document.latestJob.stage === "embedding",
      );
      return embedding ? 500 : 2_000;
    },
    refetchIntervalInBackground: false,
  });
}

export function useDocument(documentId: string) {
  return useQuery({
    queryKey: documentKeys.detail(documentId),
    queryFn: () => getDocument(documentId),
    enabled: isApiConfigured() && Boolean(documentId),
  });
}
