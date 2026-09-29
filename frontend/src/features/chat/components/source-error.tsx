import React from "react";
import { TriangleAlert } from "lucide-react";
import { ApiError, apiErrorMessage } from "@/lib/http";

export function sourceErrorCopy(error: unknown) {
  if (error instanceof ApiError && error.code === "SOURCE_DELETED") {
    return {
      title: "Source deleted",
      message: "This document was removed, so its original text is no longer available. The saved answer remains in the conversation.",
    };
  }
  if (error instanceof ApiError && error.code === "NOT_FOUND") {
    return { title: "Source unavailable", message: "This source could not be found in the document." };
  }
  return { title: "Source unavailable", message: apiErrorMessage(error, "This source could not be opened.") };
}

export function SourceErrorNotice({ error }: { error: unknown }) {
  const copy = sourceErrorCopy(error);
  return (
    <div className="source-state source-state-warning" role="alert">
      <TriangleAlert size={19} aria-hidden="true" />
      <div><strong>{copy.title}</strong><p>{copy.message}</p></div>
    </div>
  );
}
