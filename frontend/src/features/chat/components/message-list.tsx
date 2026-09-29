"use client";

import { AlertCircle, CheckCircle2, LoaderCircle, MessageSquareText, RotateCcw } from "lucide-react";
import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { LoadingState } from "@/components/ui/feedback";
import { formatDate } from "@/lib/utils";
import { Citation, MessageRecord } from "../api";
import { AnswerMarkdown } from "./answer-markdown";

type TurnMessages = { requestId: string; user?: MessageRecord; assistant?: MessageRecord };

function groupTurns(messages: MessageRecord[]) {
  const turns = new Map<string, TurnMessages>();
  for (const message of messages) {
    const turn = turns.get(message.requestId) ?? { requestId: message.requestId };
    if (message.role === "user") turn.user = message;
    else turn.assistant = message;
    turns.set(message.requestId, turn);
  }
  return [...turns.values()].sort((left, right) => {
    const leftDate = left.user?.createdAt ?? left.assistant?.createdAt ?? "";
    const rightDate = right.user?.createdAt ?? right.assistant?.createdAt ?? "";
    return leftDate.localeCompare(rightDate);
  });
}

function CitationList({ citations, onCitation }: { citations: Citation[]; onCitation: (citation: Citation) => void }) {
  if (citations.length === 0) return null;
  return (
    <div className="citation-list" aria-label="Answer sources">
      <span className="citation-list-label">Sources</span>
      {citations.map((citation) => (
        <button key={citation.label} type="button" className="citation-source-chip" onClick={() => onCitation(citation)} aria-label={`Open source ${citation.label}: ${citation.documentName}`}>
          <span>{citation.label}</span>{citation.documentName}
        </button>
      ))}
    </div>
  );
}

function PendingAnswer({ label }: { label: string }) {
  return (
    <div className="pending-answer" role="status" aria-live="polite">
      <span>{label}</span>
      <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
    </div>
  );
}

export function MessageList({
  messages,
  generatingTurn,
  isLoading,
  hasNextPage,
  isFetchingNextPage,
  onLoadMore,
  onCitation,
  onRetry,
  retryDisabled,
}: {
  messages: MessageRecord[];
  generatingTurn: { requestId: string; content: string; createdAt: string } | null;
  isLoading: boolean;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  onLoadMore: () => void;
  onCitation: (citation: Citation) => void;
  onRetry: (question: string) => void;
  retryDisabled: boolean;
}) {
  const turns = useMemo(() => groupTurns(messages), [messages]);
  const showGeneratingTurn = generatingTurn && !turns.some((turn) => turn.requestId === generatingTurn.requestId);

  return (
    <div className="message-history" role="region" aria-label="Conversation messages">
      {hasNextPage && (
        <div className="history-load-more">
          <Button variant="ghost" size="sm" onClick={onLoadMore} disabled={isFetchingNextPage}>
            {isFetchingNextPage ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : null}
            {isFetchingNextPage ? "Loading history" : "Load more history"}
          </Button>
        </div>
      )}
      {isLoading && <LoadingState label="Loading saved conversation" />}
      {!isLoading && turns.length === 0 && !showGeneratingTurn && (
        <div className="chat-first-message">
          <span className="first-message-icon"><MessageSquareText size={22} aria-hidden="true" /></span>
          <h2>What would you like to understand?</h2>
          <p>Ask about a detail, compare ideas across documents, or find where something is explained. Answers will include clickable source markers.</p>
        </div>
      )}
      {turns.map((turn) => (
        <article className="message-turn" key={turn.requestId}>
          {turn.user && (
            <div className="user-message-row">
              <div className="user-message-content">
                <div className="message-meta"><span>You</span><time dateTime={turn.user.createdAt}>{formatDate(turn.user.createdAt)}</time></div>
                <div className="user-message-bubble">{turn.user.content}</div>
              </div>
              <span className="message-avatar user-avatar" aria-hidden="true">S</span>
            </div>
          )}
          {turn.assistant && (
            <div className="assistant-message-row">
              <span className="message-avatar assistant-avatar" aria-hidden="true"><MessageSquareText size={16} /></span>
              <div className="assistant-message-content">
                <div className="message-meta assistant-meta">
                  <span>Sourcebook</span>
                  <time dateTime={turn.assistant.createdAt}>{formatDate(turn.assistant.createdAt)}</time>
                  {turn.assistant.status === "completed" && <span className="saved-status"><CheckCircle2 size={12} aria-hidden="true" /> Saved</span>}
                </div>
                {turn.assistant.status === "pending" && (
                  <PendingAnswer label="Searching your sources" />
                )}
                {turn.assistant.status === "failed" && (
                  <div className="failed-answer" role="alert">
                    <div className="failed-answer-title"><AlertCircle size={17} aria-hidden="true" /><strong>This answer could not be completed</strong></div>
                    <p>{turn.assistant.error?.message ?? "The server could not complete this turn."}</p>
                    {turn.user && <Button variant="outline" size="sm" onClick={() => onRetry(turn.user!.content)} disabled={retryDisabled}><RotateCcw size={14} aria-hidden="true" /> Try again</Button>}
                  </div>
                )}
                {turn.assistant.status === "completed" && (
                  <div className="assistant-answer">
                    {turn.assistant.answerStatus === "insufficient_evidence" && (
                      <div className="insufficient-evidence"><span className="evidence-dot" aria-hidden="true" /><span>Not enough evidence in this document scope</span></div>
                    )}
                    <AnswerMarkdown content={turn.assistant.content} citations={turn.assistant.citations} onCitation={onCitation} />
                    <CitationList citations={turn.assistant.citations} onCitation={onCitation} />
                  </div>
                )}
              </div>
            </div>
          )}
        </article>
      ))}
      {showGeneratingTurn && generatingTurn && (
        <article className="message-turn">
          <div className="user-message-row">
            <div className="user-message-content">
              <div className="message-meta"><span>You</span><time dateTime={generatingTurn.createdAt}>{formatDate(generatingTurn.createdAt)}</time></div>
              <div className="user-message-bubble">{generatingTurn.content}</div>
            </div>
            <span className="message-avatar user-avatar" aria-hidden="true">S</span>
          </div>
          <div className="assistant-message-row">
            <span className="message-avatar assistant-avatar" aria-hidden="true"><MessageSquareText size={16} /></span>
            <div className="assistant-message-content">
              <div className="message-meta assistant-meta"><span>Sourcebook</span></div>
              <PendingAnswer label="Searching your sources" />
            </div>
          </div>
        </article>
      )}
    </div>
  );
}
