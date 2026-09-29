"use client";

import { AlertCircle, ArrowLeft, BookOpenText, LoaderCircle, Plus, RefreshCw, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiConfigNotice, ErrorNotice, LoadingState } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { isApiConfigured } from "@/config/env";
import { ApiError, apiErrorMessage, type Page } from "@/lib/http";
import { useSendMessage } from "../mutations";
import { chatKeys, useConversation, useConversationMessages } from "../queries";
import { Citation, ConversationRecord, MessageRecord, listMessages } from "../api";
import { createTurnRequest } from "../request-id";
import { CitationDrawer } from "./citation-drawer";
import { MessageComposer } from "./message-composer";
import { MessageList } from "./message-list";

type Reconciliation = {
  requestId: string;
  content: string;
  state: "checking" | "not-found" | "unavailable";
  allowNewAttempt: boolean;
  acceptedByApi: boolean;
  message: string;
};

function scopeDescription(scope: ConversationRecord["scope"]) {
  return scope.mode === "all" ? "All ready documents" : `${scope.documentIds.length} selected document${scope.documentIds.length === 1 ? "" : "s"}`;
}

function isDefinitiveClientError(error: unknown) {
  return error instanceof ApiError && error.status !== null && error.status >= 400 && error.status < 500;
}

async function readHistoryUntil(
  conversationId: string,
  requestId: string | null,
  stopAtMatch: boolean,
) {
  const pages: Page<MessageRecord>[] = [];
  const pageParams: (string | null)[] = [];
  const seenCursors = new Set<string>();
  let cursor: string | null = null;
  let requestFound = false;

  while (pages.length < 1_000) {
    if (cursor !== null && seenCursors.has(cursor)) throw new Error("Conversation history pagination did not advance.");
    if (cursor !== null) seenCursors.add(cursor);
    pageParams.push(cursor);
    const page = await listMessages(conversationId, { limit: 100, cursor });
    pages.push(page);
    const matchingMessages = requestId === null ? [] : page.items.filter((message) => message.requestId === requestId);
    const found = matchingMessages.length > 0;
    requestFound ||= found;
    const assistantSaved = matchingMessages.some((message) => message.role === "assistant");
    if (assistantSaved && stopAtMatch) return { pages, pageParams, found: true };
    if (!page.nextCursor) return { pages, pageParams, found: requestFound };
    cursor = page.nextCursor;
  }

  throw new Error("This conversation has too many pages to reconcile in one pass. Load the history and check again.");
}

function HistoryReconciliationNotice({
  reconciliation,
  onCheck,
  onNewAttempt,
  isChecking,
  turnPending,
}: {
  reconciliation: Reconciliation;
  onCheck: () => void;
  onNewAttempt: () => void;
  isChecking: boolean;
  turnPending: boolean;
}) {
  return (
    <div className={`reconciliation-notice ${reconciliation.state === "checking" ? "reconciliation-checking" : ""}`} role="status" aria-live="polite">
      {isChecking || reconciliation.state === "checking"
        ? <LoaderCircle size={17} className="spin" aria-hidden="true" />
        : <AlertCircle size={17} aria-hidden="true" />}
      <div className="reconciliation-copy">
        <strong>{reconciliation.state === "checking" ? "Checking saved history" : reconciliation.state === "unavailable" ? "Saved status is still unknown" : "No saved turn was found"}</strong>
        <p>{reconciliation.message}</p>
        <div className="reconciliation-actions">
          <Button variant="outline" size="sm" onClick={onCheck} disabled={isChecking}>
            <RefreshCw size={13} aria-hidden="true" /> Check history
          </Button>
          {reconciliation.state === "not-found" && reconciliation.allowNewAttempt && (
            <Button size="sm" onClick={onNewAttempt} disabled={isChecking || turnPending}>Send as a new attempt</Button>
          )}
          {reconciliation.state === "not-found" && !reconciliation.allowNewAttempt && (
            <span className="retry-blocked-note">Resolve the server message before trying again.</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function ChatWorkspace({ conversationId }: { conversationId: string }) {
  const queryClient = useQueryClient();
  const conversationQuery = useConversation(conversationId);
  const messagesQuery = useConversationMessages(conversationId);
  const { fetchNextPage, hasNextPage, isError: isHistoryError, isFetchingNextPage } = messagesQuery;
  const sendMutation = useSendMessage(conversationId);
  const messages = useMemo(
    () => (messagesQuery.data?.pages.flatMap((page) => page.items) ?? []).slice().sort((left, right) => left.createdAt.localeCompare(right.createdAt)),
    [messagesQuery.data],
  );
  const pendingTurn = messages.some((message) => message.role === "assistant" && message.status === "pending");
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [reconciliation, setReconciliation] = useState<Reconciliation | null>(null);
  const [isCheckingHistory, setIsCheckingHistory] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage && !isHistoryError) {
      void fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage, isFetchingNextPage, isHistoryError]);

  const updateHistory = useCallback(async (requestId: string | null, stopAtMatch: boolean) => {
    const result = await readHistoryUntil(conversationId, requestId, stopAtMatch);
    await queryClient.cancelQueries({ queryKey: chatKeys.messages(conversationId) });
    queryClient.setQueryData(chatKeys.messages(conversationId), {
      pages: result.pages,
      pageParams: result.pageParams,
    });
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: chatKeys.conversation(conversationId) }),
      queryClient.invalidateQueries({ queryKey: chatKeys.conversations() }),
    ]);
    return result.found;
  }, [conversationId, queryClient]);

  const checkSavedHistory = useCallback(async (attempt: Reconciliation) => {
    setIsCheckingHistory(true);
    setReconciliation({ ...attempt, state: "checking", message: "Reading saved messages before offering another send." });
    try {
      const found = await updateHistory(attempt.requestId, false);
      if (found) {
        setReconciliation(null);
        setSendError(null);
      } else {
        setReconciliation({
          ...attempt,
          state: "not-found",
          message: attempt.acceptedByApi
            ? "The API accepted this message, but it has not appeared in history yet. Keep checking; don’t send it again."
            : "The full saved history does not contain this request. You can deliberately create a fresh attempt with a new request ID.",
        });
      }
    } catch {
      setReconciliation({ ...attempt, state: "unavailable", message: "The history request failed. Check the connection, then check again before sending another attempt." });
    } finally {
      setIsCheckingHistory(false);
    }
  }, [updateHistory]);

  const sendQuestion = useCallback(async (content: string, afterReconciliation = false) => {
    if (pendingTurn || sendMutation.isPending) return;
    if (reconciliation && !afterReconciliation) return;
    const request = createTurnRequest(content);
    const requestId = request.requestId;
    const attempt: Reconciliation = {
      requestId,
      content,
      state: "checking",
      allowNewAttempt: true,
      acceptedByApi: false,
      message: "",
    };
    setSendError(null);
    setReconciliation(null);

    try {
      await sendMutation.mutateAsync(request);
      try {
        const found = await updateHistory(requestId, true);
        if (found) {
          setReconciliation(null);
          setSendError(null);
        } else {
          setReconciliation({
            ...attempt,
            state: "unavailable",
            allowNewAttempt: false,
            acceptedByApi: true,
            message: "The API accepted this turn, but the saved messages have not appeared yet. Check history before taking another action.",
          });
        }
      } catch {
        setReconciliation({
          ...attempt,
          state: "unavailable",
          allowNewAttempt: false,
          acceptedByApi: true,
          message: "The API accepted this turn, but history could not refresh. Check history before taking another action.",
        });
      }
    } catch (error) {
      const message = apiErrorMessage(error);
      setSendError(message);
      const errorCode = error instanceof ApiError ? error.code : undefined;
      const canRetryAfterReconciliation = !isDefinitiveClientError(error) || errorCode === "TURN_IN_PROGRESS";
      const failedAttempt = { ...attempt, allowNewAttempt: canRetryAfterReconciliation };
      setReconciliation({ ...failedAttempt, state: "checking", message: "The request outcome is unclear. Checking saved history first." });
      try {
        const found = await updateHistory(requestId, false);
        if (found) {
          setReconciliation(null);
          setSendError(null);
        } else {
          setReconciliation({
            ...failedAttempt,
            state: "not-found",
            message: errorCode === "TURN_IN_PROGRESS"
              ? "Another turn is already saved in this conversation. Wait for its reply to finish, then send this question as a new attempt."
              : canRetryAfterReconciliation
              ? "The full saved history does not contain this request. You can deliberately create a fresh attempt with a new request ID."
              : `${message} The server did not save this turn, so resolve the error before trying again.`,
          });
        }
      } catch {
        setReconciliation({ ...failedAttempt, state: "unavailable", message: "Saved history could not be checked. Refresh the connection and check again before sending another attempt." });
      }
    }
  }, [pendingTurn, reconciliation, sendMutation, updateHistory]);

  const retryAsNewAttempt = useCallback(() => {
    if (!reconciliation || !reconciliation.allowNewAttempt || reconciliation.state !== "not-found") return;
    const content = reconciliation.content;
    setReconciliation(null);
    setSendError(null);
    void sendQuestion(content, true);
  }, [reconciliation, sendQuestion]);

  const title = conversationQuery.data?.title || "New conversation";
  const scope = conversationQuery.data?.scope;

  return (
    <div className="chat-workspace-page">
      {!isApiConfigured() && <ApiConfigNotice />}
      <header className="chat-workspace-header">
        <div className="chat-header-title-group">
          <Link href="/chat" className="chat-back-link" aria-label="Back to conversations"><ArrowLeft size={16} aria-hidden="true" /></Link>
          <div>
            <div className="eyebrow">CONVERSATION</div>
            <h1>{title}</h1>
          </div>
        </div>
        <div className="chat-header-actions">
          {scope && <span className="scope-summary"><BookOpenText size={14} aria-hidden="true" />{scopeDescription(scope)}</span>}
          <Link className="button button-outline button-sm" href="/chat"><Plus size={14} aria-hidden="true" /> New chat</Link>
        </div>
      </header>

      {conversationQuery.isPending && isApiConfigured() && <Card className="workspace-loading"><LoadingState label="Opening conversation" /></Card>}
      {conversationQuery.isError && <ErrorNotice error={conversationQuery.error} title="Conversation couldn’t load" />}
      {conversationQuery.data && (
        <>
          <div className="scope-immutable-note"><ShieldCheck size={14} aria-hidden="true" /><span>This conversation is fixed to <strong>{scopeDescription(conversationQuery.data.scope)}</strong>.</span><Link href="/chat">Start a new chat to change the scope</Link></div>

          <section className="chat-panel" aria-label={`Conversation: ${title}`}>
            <div className="chat-scroll-area">
              {messagesQuery.isError && <ErrorNotice error={messagesQuery.error} title="Saved messages couldn’t load" onRetry={() => void messagesQuery.refetch()} />}
              <MessageList
                messages={messages}
                isLoading={(messagesQuery.isPending || messagesQuery.isFetchingNextPage) && isApiConfigured()}
                hasNextPage={Boolean(messagesQuery.hasNextPage)}
                isFetchingNextPage={messagesQuery.isFetchingNextPage}
                onLoadMore={() => void messagesQuery.fetchNextPage()}
                onCitation={setSelectedCitation}
                onRetry={(question) => void sendQuestion(question)}
                retryDisabled={pendingTurn || sendMutation.isPending || Boolean(reconciliation) || !isApiConfigured()}
              />
            </div>

            <div className="composer-region">
              {reconciliation && (
                <HistoryReconciliationNotice
                  reconciliation={reconciliation}
                  onCheck={() => void checkSavedHistory(reconciliation)}
                  onNewAttempt={retryAsNewAttempt}
                  isChecking={isCheckingHistory}
                  turnPending={pendingTurn}
                />
              )}
              {sendError && !reconciliation && <p className="inline-error composer-error" role="alert">{sendError}</p>}
              {pendingTurn && !reconciliation && (
                <p className="pending-composer-note" role="status"><LoaderCircle size={14} className="spin" aria-hidden="true" /> A reply is still in progress. A new message will be available when it finishes.</p>
              )}
              <MessageComposer
                disabled={!isApiConfigured() || conversationQuery.isError || messagesQuery.isError || pendingTurn || Boolean(reconciliation)}
                busy={sendMutation.isPending}
                onSend={(content) => void sendQuestion(content)}
              />
              <p className="composer-privacy-note">Answers use the saved scope above. Open any source marker to inspect its original text.</p>
            </div>
          </section>
        </>
      )}

      <CitationDrawer citation={selectedCitation} onClose={() => setSelectedCitation(null)} />
    </div>
  );
}
