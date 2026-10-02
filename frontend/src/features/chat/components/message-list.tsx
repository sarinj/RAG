"use client";

import { AlertCircle, CheckCircle2, LoaderCircle, MessageSquareText, RotateCcw } from "lucide-react";
import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { LoadingState } from "@/components/ui/feedback";
import { formatDate } from "@/lib/utils";
import { Citation, MessageRecord } from "../api";
import { AnswerMarkdown } from "./answer-markdown";
import { AnswerProgress, StreamingDraft } from "./answer-progress";
import type { StreamReply } from "../streaming";

type TurnMessages = { requestId: string; user?: MessageRecord; assistant?: MessageRecord };
const NO_STREAMS: Record<string, StreamReply> = {};

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

function CompletedAnswer({ message, onCitation }: { message: MessageRecord; onCitation: (citation: Citation) => void }) {
  return (
    <div className="assistant-answer">
      {message.answerStatus === "insufficient_evidence" && (
        <div className="insufficient-evidence"><span className="evidence-dot" aria-hidden="true" /><span>Not enough evidence in this document scope</span></div>
      )}
      <AnswerMarkdown content={message.content} citations={message.citations} onCitation={onCitation} />
      <CitationList citations={message.citations} onCitation={onCitation} />
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
  streams = NO_STREAMS,
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
  streams?: Record<string, StreamReply>;
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
      {turns.map((turn) => {
        const stream = streams[turn.requestId];
        const assistant = stream?.turn?.assistantMessage ?? turn.assistant;
        return (
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
            {assistant && (
              <div className="assistant-message-row">
                <span className="message-avatar assistant-avatar" aria-hidden="true"><MessageSquareText size={16} /></span>
                <div className="assistant-message-content">
                  <div className="message-meta assistant-meta">
                    <span>Sourcebook</span>
                    <time dateTime={assistant.createdAt}>{formatDate(assistant.createdAt)}</time>
                    {assistant.status === "completed" && <span className="saved-status"><CheckCircle2 size={12} aria-hidden="true" /> Saved</span>}
                  </div>
                  {assistant.status === "pending" && (
                    <StreamingDraft reply={stream} />
                  )}
                  {assistant.status === "failed" && (
                    <div className="failed-answer" role="alert">
                      <AnswerProgress reply={stream} />
                      <div className="failed-answer-title"><AlertCircle size={17} aria-hidden="true" /><strong>This answer could not be completed</strong></div>
                      <p>{assistant.error?.message ?? "The server could not complete this turn."}</p>
                      {turn.user && <Button variant="outline" size="sm" onClick={() => onRetry(turn.user!.content)} disabled={retryDisabled}><RotateCcw size={14} aria-hidden="true" /> Try again</Button>}
                    </div>
                  )}
                  {assistant.status === "completed" && (
                    <>
                      <AnswerProgress reply={stream} />
                      <CompletedAnswer message={assistant} onCitation={onCitation} />
                    </>
                  )}
                </div>
              </div>
            )}
          </article>
        );
      })}
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
              {streams[generatingTurn.requestId]?.turn?.assistantMessage.status === "completed" ? (
                <>
                  <AnswerProgress reply={streams[generatingTurn.requestId]} />
                  <CompletedAnswer message={streams[generatingTurn.requestId].turn!.assistantMessage} onCitation={onCitation} />
                </>
              ) : <StreamingDraft reply={streams[generatingTurn.requestId]} />}
            </div>
          </div>
        </article>
      )}
    </div>
  );
}
