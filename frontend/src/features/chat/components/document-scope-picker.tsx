"use client";

import { Check, ChevronDown, FileText, LoaderCircle, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { ErrorNotice, LoadingState } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { formatBytes } from "@/lib/utils";
import { useDocuments, type DocumentRecord } from "@/features/documents";
import type { Scope } from "../api";

export function DocumentScopePicker({ value, onChange }: { value: Scope; onChange: (scope: Scope) => void }) {
  const documentsQuery = useDocuments();
  const [search, setSearch] = useState("");
  const readyDocuments = useMemo(
    () => (documentsQuery.data?.pages.flatMap((page) => page.items) ?? []).filter((document) => document.status === "ready"),
    [documentsQuery.data],
  );
  const visibleDocuments = readyDocuments.filter((document) => document.name.toLowerCase().includes(search.toLowerCase()));
  const selected = new Set(value.mode === "selected" ? value.documentIds : []);

  function toggle(documentId: string) {
    const next = new Set(selected);
    if (next.has(documentId)) next.delete(documentId);
    else if (next.size < 100) next.add(documentId);
    onChange({ mode: "selected", documentIds: [...next] });
  }

  return (
    <Card className="scope-card">
      <div className="section-heading scope-heading">
        <div>
          <div className="eyebrow">CHOOSE YOUR SOURCE MATERIAL</div>
          <h2>What should this chat use?</h2>
          <p>Your selection is saved with the conversation and stays fixed.</p>
        </div>
        <span className="scope-count">{value.mode === "all" ? "All ready" : `${value.documentIds.length} selected`}</span>
      </div>

      <div className="scope-options" role="group" aria-label="Conversation document scope">
        <button
          type="button"
          className={`scope-option ${value.mode === "all" ? "scope-option-active" : ""}`}
          aria-pressed={value.mode === "all"}
          onClick={() => onChange({ mode: "all" })}
        >
          <span className="scope-radio" aria-hidden="true">{value.mode === "all" && <span />}</span>
          <span className="scope-option-copy"><strong>All ready documents</strong><small>Use every document that is ready when you ask</small></span>
          <span className="scope-option-count">{documentsQuery.hasNextPage ? "Browse" : readyDocuments.length}</span>
        </button>
        <button
          type="button"
          className={`scope-option ${value.mode === "selected" ? "scope-option-active" : ""}`}
          aria-pressed={value.mode === "selected"}
          onClick={() => onChange({ mode: "selected", documentIds: value.mode === "selected" ? value.documentIds : [] })}
        >
          <span className="scope-radio" aria-hidden="true">{value.mode === "selected" && <span />}</span>
          <span className="scope-option-copy"><strong>Selected documents</strong><small>Choose a focused set for this chat</small></span>
          <span className="scope-option-count">{value.mode === "selected" ? value.documentIds.length : "—"}</span>
        </button>
      </div>

      {value.mode === "selected" && (
        <div className="document-picker-panel">
          <div className="document-picker-toolbar">
            <span>Ready documents</span>
            {readyDocuments.length > 3 && (
              <label className="document-search">
                <Search size={14} aria-hidden="true" />
                <span className="visually-hidden">Filter documents</span>
                <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter by name" />
              </label>
            )}
          </div>
          {documentsQuery.isPending && <LoadingState label="Loading ready documents" />}
          {documentsQuery.isError && <ErrorNotice error={documentsQuery.error} title="Documents couldn’t load" onRetry={() => void documentsQuery.refetch()} />}
          {documentsQuery.isSuccess && visibleDocuments.length === 0 && (
            <p className="picker-empty">{readyDocuments.length === 0 ? "No documents are ready yet. You can still start a chat and add documents later." : "No document names match this filter."}</p>
          )}
          {visibleDocuments.length > 0 && (
            <div className="picker-document-list">
              {visibleDocuments.map((document: DocumentRecord) => {
                const checked = selected.has(document.id);
                return (
                  <button key={document.id} type="button" className={`picker-document ${checked ? "picker-document-selected" : ""}`} onClick={() => toggle(document.id)} aria-pressed={checked} disabled={!checked && selected.size >= 100} title={!checked && selected.size >= 100 ? "The contract allows up to 100 selected documents" : undefined}>
                    <span className={`picker-checkbox ${checked ? "picker-checkbox-checked" : ""}`} aria-hidden="true">{checked && <Check size={13} />}</span>
                    <FileText size={16} className="picker-file-icon" aria-hidden="true" />
                    <span className="picker-document-name">{document.name}</span>
                    <span className="picker-document-size">{formatBytes(document.sizeBytes)}</span>
                  </button>
                );
              })}
            </div>
          )}
          {documentsQuery.hasNextPage && (
            <Button variant="ghost" size="sm" className="picker-load-more" onClick={() => void documentsQuery.fetchNextPage()} disabled={documentsQuery.isFetchingNextPage}>
              {documentsQuery.isFetchingNextPage ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
              Load more documents
            </Button>
          )}
          {value.documentIds.length === 0 && <p className="selection-hint" role="status">Select at least one ready document to start with a focused scope.</p>}
          {value.documentIds.length === 100 && <p className="selection-hint" role="status">The v1 contract allows up to 100 selected documents.</p>}
        </div>
      )}
    </Card>
  );
}
