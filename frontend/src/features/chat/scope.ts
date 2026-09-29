import { ApiError } from "@/lib/http";
import type { Scope } from "./api";

export function serializeScope(scope: Scope): { scope: Scope } {
  if (scope.mode === "all") return { scope: { mode: "all" } };
  const uniqueIds = [...new Set(scope.documentIds)];
  if (uniqueIds.length === 0 || uniqueIds.length !== scope.documentIds.length || uniqueIds.length > 100) {
    throw new ApiError(
      "Choose between 1 and 100 unique ready documents for a selected scope.",
      "INVALID_SCOPE",
      null,
      null,
    );
  }
  return { scope: { mode: "selected", documentIds: uniqueIds } };
}
