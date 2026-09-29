import { http, type Page } from "@/lib/http";
import type { components } from "@/generated/api";

type Schemas = components["schemas"];

export type JobStatus = Schemas["JobDto"]["status"];
export type JobStage = Schemas["JobDto"]["stage"];
export type Failure = Schemas["ApiFailureDto"];
// The current OpenAPI schema marks nullable scalar values as empty objects.
export type IngestionJob = Omit<Schemas["JobDto"], "totalChunks"> & { totalChunks: number | null };
export type DocumentRecord = Omit<Schemas["DocumentDto"], "activeGenerationId" | "latestJob"> & {
  activeGenerationId: string | null;
  latestJob: IngestionJob | null;
};

export type DocumentPage = Page<DocumentRecord>;

export async function listDocuments(params: { limit: number; cursor: string | null }): Promise<DocumentPage> {
  const response = await http.get<DocumentPage>("/documents", { params });
  return response.data;
}

export async function getDocument(documentId: string) {
  const response = await http.get<DocumentRecord>(`/documents/${encodeURIComponent(documentId)}`);
  return response.data;
}

export async function uploadDocument(file: File, onProgress: (percentage: number | null) => void) {
  const body = new FormData();
  body.append("file", file);
  const response = await http.post<DocumentRecord>("/documents", body, {
    onUploadProgress: (event) => {
      if (!event.total) {
        onProgress(null);
        return;
      }
      onProgress(Math.min(100, Math.round((event.loaded / event.total) * 100)));
    },
  });
  return response.data;
}

export async function retryDocument(documentId: string) {
  const response = await http.post<DocumentRecord>(`/documents/${encodeURIComponent(documentId)}/retry`, {});
  return response.data;
}

export async function reindexDocument(documentId: string) {
  const response = await http.post<DocumentRecord>(`/documents/${encodeURIComponent(documentId)}/reindex`, {});
  return response.data;
}

export async function deleteDocument(documentId: string) {
  await http.delete<void>(`/documents/${encodeURIComponent(documentId)}`);
}
