"use client";

import { AlertTriangle, Check, File, FileText, LoaderCircle, MoreHorizontal, Plus, RefreshCw, Trash2, UploadCloud } from "lucide-react";
import { useRef, useState, type DragEvent, type FormEvent } from "react";
import { ApiConfigNotice, ErrorNotice, LoadingState } from "@/components/ui/feedback";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { isApiConfigured } from "@/config/env";
import { apiErrorMessage } from "@/lib/http";
import { formatBytes, formatDate, titleCase } from "@/lib/utils";
import { DocumentRecord } from "../api";
import { useDeleteDocument, useReindexDocument, useRetryDocument, useUploadDocument } from "../mutations";
import { useDocuments } from "../queries";

const acceptedExtensions = [".pdf", ".txt", ".md", ".markdown"];
const acceptHint = ".pdf,.txt,.md,.markdown,text/plain,text/markdown,application/pdf";

function isLikelySupported(file: File) {
  const filename = file.name.toLowerCase();
  return acceptedExtensions.some((extension) => filename.endsWith(extension)) || file.type === "application/pdf" || file.type === "text/plain";
}

function DocumentStatus({ document }: { document: DocumentRecord }) {
  const status = document.status;
  const label = status === "ready" ? "Ready" : titleCase(status);
  const tone = status === "ready" ? "status-ready" : status === "failed" ? "status-failed" : "status-progress";

  return (
    <div className="status-stack">
      <Badge className={`status-badge ${tone}`}>
        <span className="status-indicator" aria-hidden="true" />{label}
      </Badge>
      {document.latestJob?.status === "failed" && (
        <span className="job-failure-label" title={document.latestJob.error?.message ?? "The latest indexing job failed."}>
          {status === "ready" ? "Reindex failed" : document.latestJob.error?.message ?? "Processing failed"}
        </span>
      )}
      {(document.status === "queued" || document.status === "processing" || document.latestJob?.status === "queued" || document.latestJob?.status === "processing") && document.latestJob && (
        <span className="job-stage-label">
          {titleCase(document.latestJob.stage)}
          {document.latestJob.totalChunks !== null && document.latestJob.totalChunks > 0
            ? ` · ${document.latestJob.processedChunks}/${document.latestJob.totalChunks} chunks`
            : ""}
        </span>
      )}
    </div>
  );
}

function DocumentIcon({ document }: { document: DocumentRecord }) {
  const isPdf = document.mimeType === "application/pdf" || document.name.toLowerCase().endsWith(".pdf");
  return <span className={`file-type-icon ${isPdf ? "file-type-pdf" : ""}`} aria-hidden="true">{isPdf ? <FileText size={17} /> : <File size={17} />}</span>;
}

function UploadPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<DocumentRecord | null>(null);
  const upload = useUploadDocument();

  function chooseFile(nextFile: File | undefined) {
    if (!nextFile) return;
    setFile(nextFile);
    setUploadError(null);
    setUploaded(null);
    setUploadProgress(null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files.item(0) ?? undefined);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setUploadError(null);
    setUploadProgress(0);
    try {
      const document = await upload.mutateAsync({
        file,
        onProgress: setUploadProgress,
      });
      setUploaded(document);
      setFile(null);
      setUploadProgress(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (error) {
      setUploadError(apiErrorMessage(error));
      setUploadProgress(null);
    }
  }

  return (
    <Card className="upload-card">
      <div className="section-heading upload-heading">
        <div>
          <div className="eyebrow">ADD TO YOUR LIBRARY</div>
          <h2>Bring in a document</h2>
          <p>Upload a file to make it searchable in your conversations.</p>
        </div>
        <span className="upload-heading-icon"><UploadCloud size={20} aria-hidden="true" /></span>
      </div>

      <form onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          className="visually-hidden"
          id="document-upload"
          type="file"
          accept={acceptHint}
          onChange={(event) => chooseFile(event.currentTarget.files?.item(0) ?? undefined)}
          aria-describedby="upload-help"
        />
        <div
          className={`dropzone ${dragging ? "dropzone-active" : ""} ${file ? "dropzone-selected" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
          }}
          onDrop={handleDrop}
        >
          <label className="dropzone-label" htmlFor="document-upload">
            <span className="dropzone-icon"><UploadCloud size={21} aria-hidden="true" /></span>
            <span className="dropzone-title">{file ? file.name : "Drop a file here, or browse"}</span>
            <span className="dropzone-subtitle">
              {file
                ? `${formatBytes(file.size)}${isLikelySupported(file) ? "" : " · the server may not support this file type"}`
                : "Choose one PDF, TXT, or Markdown file at a time"}
            </span>
          </label>
          {file && (
            <button className="clear-file-button" type="button" onClick={() => { setFile(null); if (inputRef.current) inputRef.current.value = ""; }} aria-label="Remove selected file">
              Remove
            </button>
          )}
        </div>

        <div className="upload-footer">
          <p id="upload-help">The server validates file type and size. Upload progress ends when the file reaches the API; indexing continues afterward.</p>
          <Button type="submit" disabled={!file || upload.isPending || !isApiConfigured()}>
            {upload.isPending ? <><LoaderCircle size={16} className="spin" aria-hidden="true" /> Uploading</> : <><Plus size={16} aria-hidden="true" /> Upload file</>}
          </Button>
        </div>
        {upload.isPending && (
          <div className="upload-progress" role="status" aria-live="polite">
            <div className="upload-progress-copy">
              <span>Uploading {file?.name ?? "file"}</span>
              <span>{uploadProgress === null ? "Sending…" : `${uploadProgress}%`}</span>
            </div>
              <div className={`progress-track ${uploadProgress === null ? "progress-indeterminate" : ""}`} aria-hidden="true"><span style={uploadProgress === null ? undefined : { width: `${uploadProgress}%` }} /></div>
          </div>
        )}
      </form>

      {uploadError && <p className="inline-error" role="alert">{uploadError}</p>}
      {uploaded && (
        <p className="inline-success" role="status">
          <Check size={16} aria-hidden="true" /> {uploaded.name} is uploaded. Its ingestion job is {uploaded.latestJob?.status ?? uploaded.status}; library status will update as the server processes it.
        </p>
      )}
    </Card>
  );
}

function DeleteDocumentDialog({ document, onClose }: { document: DocumentRecord | null; onClose: () => void }) {
  const remove = useDeleteDocument();
  const [error, setError] = useState<string | null>(null);

  async function confirmDelete() {
    if (!document) return;
    setError(null);
    try {
      await remove.mutateAsync(document.id);
      onClose();
    } catch (reason) {
      setError(apiErrorMessage(reason));
    }
  }

  return (
    <Dialog open={Boolean(document)} onOpenChange={(open) => { if (!open && !remove.isPending) { setError(null); onClose(); } }}>
      <DialogContent className="confirm-dialog" aria-describedby="delete-document-description">
        <span className="confirm-icon"><Trash2 size={19} aria-hidden="true" /></span>
        <DialogTitle>Remove this document?</DialogTitle>
        <DialogDescription id="delete-document-description">
          {document ? `“${document.name}” will be removed from your active library and its sources will no longer open.` : ""}
          {" "}Saved chat text that already quotes this document will remain in your conversation history.
        </DialogDescription>
        {error && <p className="inline-error" role="alert">{error}</p>}
        <div className="dialog-actions">
          <Button variant="outline" onClick={onClose} disabled={remove.isPending}>Keep document</Button>
          <Button variant="destructive" onClick={confirmDelete} disabled={remove.isPending}>
            {remove.isPending ? <><LoaderCircle size={15} className="spin" aria-hidden="true" /> Removing</> : "Remove document"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function DocumentRow({ document, onDelete }: { document: DocumentRecord; onDelete: (document: DocumentRecord) => void }) {
  const retry = useRetryDocument();
  const reindex = useReindexDocument();
  const [actionError, setActionError] = useState<string | null>(null);
  const busy = retry.isPending || reindex.isPending;

  async function runAction(action: () => Promise<DocumentRecord>) {
    setActionError(null);
    try {
      await action();
    } catch (error) {
      setActionError(apiErrorMessage(error));
    }
  }

  return (
    <div className="document-row" role="row">
      <div className="document-main-cell" role="cell">
        <DocumentIcon document={document} />
        <div className="document-name-cell">
          <strong title={document.name}>{document.name}</strong>
          <span>{document.mimeType || "Document"}</span>
        </div>
      </div>
      <div className="document-size-cell" role="cell">{formatBytes(document.sizeBytes)}</div>
      <div className="document-status-cell" role="cell"><DocumentStatus document={document} /></div>
      <div className="document-date-cell" role="cell" title={formatDate(document.updatedAt)}>{formatDate(document.createdAt)}</div>
      <div className="document-actions-cell" role="cell">
        {document.latestJob?.status === "failed" && (
          <Button variant="outline" size="sm" onClick={() => void runAction(() => retry.mutateAsync(document.id))} disabled={busy}>
            {retry.isPending ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <RefreshCw size={14} aria-hidden="true" />}
            Retry job
          </Button>
        )}
        {document.status === "ready" && document.latestJob?.status !== "failed" && document.latestJob?.status !== "queued" && document.latestJob?.status !== "processing" && (
          <Button variant="ghost" size="sm" onClick={() => void runAction(() => reindex.mutateAsync(document.id))} disabled={busy} title="Rebuild the index using the server’s current settings">
            {reindex.isPending ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <RefreshCw size={14} aria-hidden="true" />}
            Reindex
          </Button>
        )}
        <Button variant="ghost" size="icon" className="row-more-button" onClick={() => onDelete(document)} aria-label={`Remove ${document.name}`}>
          <Trash2 size={15} aria-hidden="true" />
        </Button>
      </div>
      {actionError && <p className="row-action-error" role="cell" aria-live="polite"><AlertTriangle size={13} aria-hidden="true" /> {actionError}</p>}
      {document.latestJob?.status === "failed" && document.latestJob.error && (
        <p className="row-job-error" role="cell"><AlertTriangle size={13} aria-hidden="true" /> {document.latestJob.error.message}</p>
      )}
    </div>
  );
}

export function DocumentLibrary() {
  const documentsQuery = useDocuments();
  const [deleteTarget, setDeleteTarget] = useState<DocumentRecord | null>(null);
  const pages = documentsQuery.data?.pages ?? [];
  const documents = pages.flatMap((page) => page.items).filter((document) => document.status !== "deleted");
  const readyCount = documents.filter((document) => document.status === "ready").length;
  const activeCount = documents.filter((document) => document.status === "queued" || document.status === "processing" || document.latestJob?.status === "processing" || document.latestJob?.status === "queued").length;

  return (
    <div className="page-stack">
      <section className="page-heading document-page-heading">
        <div>
          <div className="eyebrow">YOUR KNOWLEDGE, IN ONE PLACE</div>
          <h1>Document library</h1>
          <p>Collect the material you trust, then ask questions that lead back to the page.</p>
        </div>
        <div className="library-summary" aria-label={`${readyCount} ready documents and ${activeCount} processing`}>
          <span className="summary-item"><span className="summary-dot summary-ready" aria-hidden="true" />{readyCount} ready shown</span>
          <span className="summary-separator" aria-hidden="true" />
            <span className="summary-item"><span className="summary-dot summary-processing" aria-hidden="true" />{activeCount} processing shown</span>
        </div>
      </section>

      {!isApiConfigured() && <ApiConfigNotice />}
      <UploadPanel />

        <section className="library-section" aria-labelledby="documents-heading">
        <div className="list-section-heading">
          <div>
            <div className="eyebrow">SAVED MATERIAL</div>
            <h2 id="documents-heading">Your documents <span className="count-pill">{documents.length} loaded</span></h2>
          </div>
          <span className="list-updated-label">Availability follows the latest server status</span>
        </div>

        {documentsQuery.isPending && isApiConfigured() && <Card className="library-state-card"><LoadingState label="Loading your library" /></Card>}
        {documentsQuery.isError && <ErrorNotice error={documentsQuery.error} title="The library couldn’t load" onRetry={() => void documentsQuery.refetch()} />}
        {documentsQuery.isSuccess && documents.length === 0 && (
          <Card className="empty-library">
            <span className="empty-illustration"><FileText size={25} aria-hidden="true" /></span>
            <h3>Your library is ready for its first document</h3>
            <p>Upload a PDF, text file, or Markdown document. Once it is ready, you can ask grounded questions about it.</p>
          </Card>
        )}
        {documents.length > 0 && (
          <Card className="document-table-card" role="table" aria-label="Your documents">
            <div className="document-table-head" role="row">
              <span role="columnheader">DOCUMENT</span><span role="columnheader">SIZE</span><span role="columnheader">AVAILABILITY</span><span role="columnheader">ADDED</span><span role="columnheader"><span className="visually-hidden">Actions</span></span>
            </div>
            <div className="document-list" role="rowgroup">
              {documents.map((document) => <DocumentRow key={document.id} document={document} onDelete={setDeleteTarget} />)}
            </div>
            {documentsQuery.hasNextPage && (
              <div className="load-more-row">
                <Button variant="ghost" onClick={() => void documentsQuery.fetchNextPage()} disabled={documentsQuery.isFetchingNextPage}>
                  {documentsQuery.isFetchingNextPage ? <LoaderCircle size={15} className="spin" aria-hidden="true" /> : <MoreHorizontal size={16} aria-hidden="true" />}
                  {documentsQuery.isFetchingNextPage ? "Loading" : "Load more documents"}
                </Button>
              </div>
            )}
          </Card>
        )}
      </section>

      <DeleteDocumentDialog document={deleteTarget} onClose={() => setDeleteTarget(null)} />
    </div>
  );
}
