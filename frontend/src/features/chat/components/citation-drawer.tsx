"use client";

import { BookOpenText, FileText } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { LoadingState } from "@/components/ui/feedback";
import { Citation } from "../api";
import { useSource } from "../queries";
import { SourceErrorNotice } from "./source-error";

function sourceLocation(citation: Citation) {
  if (citation.pageStart === null) return citation.heading ?? "Text document";
  const page = citation.pageEnd && citation.pageEnd !== citation.pageStart
    ? `pp. ${citation.pageStart}–${citation.pageEnd}`
    : `p. ${citation.pageStart}`;
  return citation.heading ? `${page} · ${citation.heading}` : page;
}

export function CitationDrawer({ citation, onClose }: { citation: Citation | null; onClose: () => void }) {
  const sourceQuery = useSource(citation?.documentId ?? null, citation?.chunkId ?? null);
  return (
    <Dialog open={Boolean(citation)} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="source-drawer" aria-describedby="source-drawer-description">
        <div className="source-drawer-topline"><span className="source-drawer-icon"><BookOpenText size={18} aria-hidden="true" /></span><span>ANSWER SOURCE</span></div>
        <DialogTitle>{citation?.documentName ?? "Source"}</DialogTitle>
        <DialogDescription id="source-drawer-description">{citation ? sourceLocation(citation) : "Original text used to support this answer."}</DialogDescription>

        <div className="source-excerpt-area">
          {sourceQuery.isPending && <LoadingState label="Opening original source" />}
          {sourceQuery.isError && <SourceErrorNotice error={sourceQuery.error} />}
          {sourceQuery.data && (
            <div className="source-excerpt-wrap">
              <div className="source-excerpt-label"><FileText size={15} aria-hidden="true" /> Original excerpt</div>
              <blockquote className="source-excerpt">{sourceQuery.data.text}</blockquote>
              <p className="source-excerpt-note">This is the original document text associated with {citation?.label}.</p>
            </div>
          )}
        </div>
        <div className="source-drawer-footer"><span>Source ID</span><code>{citation?.chunkId ?? "—"}</code></div>
      </DialogContent>
    </Dialog>
  );
}
