"use client";

import { ArrowDownRight, ArrowUpRight, BookOpenText, Clock3, LoaderCircle, MessageSquareText, Plus, Sparkles } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ApiConfigNotice, ErrorNotice, LoadingState } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { isApiConfigured } from "@/config/env";
import { apiErrorMessage } from "@/lib/http";
import { formatRelativeDate } from "@/lib/utils";
import { useDocuments } from "@/features/documents";
import { Scope } from "../api";
import { useCreateConversation } from "../mutations";
import { useConversations } from "../queries";
import { DocumentScopePicker } from "./document-scope-picker";

export function NewChatPage() {
  const router = useRouter();
  const [scope, setScope] = useState<Scope>({ mode: "all" });
  const [createError, setCreateError] = useState<string | null>(null);
  const create = useCreateConversation();
  const conversationsQuery = useConversations();
  const documentsQuery = useDocuments();
  const readyDocuments = documentsQuery.data?.pages.flatMap((page) => page.items).filter((document) => document.status === "ready") ?? [];
  const hasAnyReady = readyDocuments.length > 0 || Boolean(documentsQuery.hasNextPage);
  const canCreate = scope.mode === "all" || (scope.documentIds.length > 0 && scope.documentIds.length <= 100);
  const conversations = conversationsQuery.data?.pages.flatMap((page) => page.items) ?? [];

  async function startConversation() {
    setCreateError(null);
    try {
      const conversation = await create.mutateAsync(scope);
      router.push(`/chat/${conversation.id}`);
    } catch (error) {
      setCreateError(apiErrorMessage(error));
    }
  }

  return (
    <div className="page-stack chat-home-page">
      <section className="page-heading chat-home-heading">
        <div>
          <div className="eyebrow">A BETTER WAY TO READ ACROSS FILES</div>
          <h1>Ask your library</h1>
          <p>Start with a question. Every answer can point you back to the exact source.</p>
        </div>
        <span className="chat-home-sparkle"><Sparkles size={21} aria-hidden="true" /></span>
      </section>

      {!isApiConfigured() && <ApiConfigNotice />}
      <DocumentScopePicker value={scope} onChange={setScope} />

      {createError && <ErrorNotice error={createError} title="Conversation couldn’t start" />}
      <div className="start-chat-row">
        <span><BookOpenText size={15} aria-hidden="true" /> {scope.mode === "all" ? "The server will use ready documents when you send a question." : `${scope.documentIds.length} document${scope.documentIds.length === 1 ? "" : "s"} selected for this conversation.`}</span>
        <Button onClick={() => void startConversation()} disabled={!canCreate || create.isPending || !isApiConfigured()}>
          {create.isPending ? <><LoaderCircle size={16} className="spin" aria-hidden="true" /> Creating</> : <>Start a conversation <ArrowUpRight size={16} aria-hidden="true" /></>}
        </Button>
      </div>

      {!hasAnyReady && documentsQuery.isSuccess && (
        <p className="chat-empty-library-note">You can create a conversation now. Add documents in the <Link href="/documents">library</Link> before asking a question.</p>
      )}

      <section className="recent-conversations" aria-labelledby="recent-heading">
        <div className="list-section-heading">
          <div>
            <div className="eyebrow">PICK UP WHERE YOU LEFT OFF</div>
            <h2 id="recent-heading">Recent conversations</h2>
          </div>
          <span className="recent-count">{conversations.length}</span>
        </div>
        {conversationsQuery.isPending && isApiConfigured() && <Card className="library-state-card"><LoadingState label="Loading conversations" /></Card>}
        {conversationsQuery.isError && <ErrorNotice error={conversationsQuery.error} title="Conversations couldn’t load" onRetry={() => void conversationsQuery.refetch()} />}
        {conversationsQuery.isSuccess && conversations.length === 0 && (
          <Card className="empty-conversations">
            <span className="empty-illustration"><MessageSquareText size={22} aria-hidden="true" /></span>
            <div><h3>No conversations yet</h3><p>Your saved questions and answers will appear here.</p></div>
            <ArrowDownRight size={18} className="empty-arrow" aria-hidden="true" />
          </Card>
        )}
        {conversations.length > 0 && (
          <Card className="conversation-list-card">
            {conversations.map((conversation) => (
              <Link className="conversation-row" key={conversation.id} href={`/chat/${conversation.id}`}>
                <span className="conversation-icon"><MessageSquareText size={16} aria-hidden="true" /></span>
                <span className="conversation-title"><strong>{conversation.title || "New chat"}</strong><small>{conversation.scope.mode === "all" ? "All ready documents" : `${conversation.scope.documentIds.length} selected documents`}</small></span>
                <span className="conversation-time"><Clock3 size={13} aria-hidden="true" />{formatRelativeDate(conversation.updatedAt)}</span>
                <ArrowUpRight size={15} className="conversation-open" aria-hidden="true" />
              </Link>
            ))}
            {conversationsQuery.hasNextPage && (
              <div className="load-more-row"><Button variant="ghost" onClick={() => void conversationsQuery.fetchNextPage()} disabled={conversationsQuery.isFetchingNextPage}>
                {conversationsQuery.isFetchingNextPage ? <LoaderCircle size={15} className="spin" aria-hidden="true" /> : <Plus size={15} aria-hidden="true" />} Load older conversations
              </Button></div>
            )}
          </Card>
        )}
      </section>
    </div>
  );
}
