import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ApiError } from "@/lib/http";
import { deleteDocument, reindexDocument, retryDocument, uploadDocument } from "./api";
import { documentKeys } from "./queries";

export function useUploadDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ file, onProgress }: { file: File; onProgress: (percentage: number | null) => void }) =>
      uploadDocument(file, onProgress),
    onSuccess: async (document) => {
      queryClient.setQueryData(documentKeys.detail(document.id), document);
      await queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
    onError: async (error) => {
      if (error instanceof ApiError && error.code === "DUPLICATE_DOCUMENT") {
        await queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
      }
    },
  });
}

export function useRetryDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: retryDocument,
    onSuccess: async (document) => {
      queryClient.setQueryData(documentKeys.detail(document.id), document);
      await queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
  });
}

export function useReindexDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: reindexDocument,
    onSuccess: async (document) => {
      queryClient.setQueryData(documentKeys.detail(document.id), document);
      await queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: async (_result, documentId) => {
      queryClient.removeQueries({ queryKey: documentKeys.detail(documentId) });
      await queryClient.invalidateQueries({ queryKey: documentKeys.lists() });
    },
  });
}
