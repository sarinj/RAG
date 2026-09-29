export function hasActiveIngestionJob(documents: readonly { latestJob: { status: string } | null }[]) {
  return documents.some((document) => document.latestJob?.status === "queued" || document.latestJob?.status === "processing");
}
