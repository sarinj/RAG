import axios, { AxiosError } from "axios";
import { publicConfig } from "@/config/env";

export type Page<T> = { items: T[]; nextCursor: string | null };

type ErrorEnvelope = { code?: unknown; message?: unknown; requestId?: unknown };

export class ApiError extends Error {
  readonly name = "ApiError";

  constructor(
    message: string,
    readonly code: string,
    readonly status: number | null,
    readonly requestId: string | null,
  ) {
    super(message);
  }
}

export const http = axios.create({
  baseURL: publicConfig.apiBaseUrl ?? undefined,
  timeout: 150_000,
  headers: { Accept: "application/json" },
});

function makeRequestId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
    const random = Math.random() * 16;
    const value = character === "x" ? random : (random % 4) + 8;
    return Math.floor(value).toString(16);
  });
}

http.interceptors.request.use((request) => {
  if (!publicConfig.apiBaseUrl) {
    throw new ApiError(
      publicConfig.apiConfigError ?? "The public API URL is not configured.",
      "FRONTEND_CONFIGURATION_ERROR",
      null,
      null,
    );
  }

  request.headers.set("X-Request-Id", request.headers.get("X-Request-Id") ?? makeRequestId());
  return request;
});

http.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (error instanceof ApiError) return Promise.reject(error);
    if (!axios.isAxiosError(error)) {
      return Promise.reject(new ApiError("Something went wrong. Please try again.", "UNKNOWN_ERROR", null, null));
    }

    const responseData = error.response?.data as ErrorEnvelope | undefined;
    const headerRequestId = error.response?.headers?.["x-request-id"];
    const requestId = typeof responseData?.requestId === "string"
      ? responseData.requestId
      : typeof headerRequestId === "string"
        ? headerRequestId
        : null;
    const status = error.response?.status ?? null;
    const code = typeof responseData?.code === "string" ? responseData.code : "REQUEST_FAILED";
    const message = typeof responseData?.message === "string"
      ? responseData.message
      : error.code === AxiosError.ERR_CANCELED
        ? "The request was cancelled."
        : error.code === AxiosError.ECONNABORTED
          ? "The request took too long. Refresh the conversation or library to check whether it was saved."
          : error.response
            ? "The request could not be completed. Please try again."
            : "The API could not be reached. Check that NestJS is running and the public API URL is correct.";

    return Promise.reject(new ApiError(message, code, status, requestId));
  },
);

export function apiErrorMessage(error: unknown, fallback = "Something went wrong. Please try again.") {
  if (typeof error === "string") return error;
  if (error instanceof ApiError) {
    const friendly: Record<string, string> = {
      DUPLICATE_DOCUMENT: "This file is already in your library. The list has been refreshed.",
      FILE_TOO_LARGE: "This file is larger than the server allows.",
      UNSUPPORTED_FILE_TYPE: "The server does not support this file type.",
      EMPTY_DOCUMENT: "No readable text was found in this file.",
      NO_READY_DOCUMENTS: "There are no ready documents to search yet.",
      DOCUMENT_NOT_READY: "One or more selected documents are no longer ready. Refresh the library and try again.",
      SCOPE_CHANGED: "The selected library changed during this answer. Your conversation is still saved.",
      TURN_IN_PROGRESS: "This conversation already has a reply in progress.",
      REQUEST_ID_CONFLICT: "This send ID was already used with different text. Start a fresh attempt.",
      SOURCE_DELETED: "This source was deleted. The saved answer is still available.",
      NOT_FOUND: "This item could not be found. It may have been deleted.",
      MODEL_UNAVAILABLE: "The answer service is unavailable right now. Your saved conversation is safe.",
      GENERATION_TIMEOUT: "The answer service timed out. Refresh the conversation to check its saved status.",
    };
    return friendly[error.code] ?? error.message;
  }
  return error instanceof Error ? error.message : fallback;
}
