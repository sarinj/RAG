import type { Metadata } from "next";
import { ChatWorkspace } from "@/features/chat";

export const metadata: Metadata = { title: "Conversation" };

export default async function ConversationPage({ params }: { params: Promise<{ conversationId: string }> }) {
  const { conversationId } = await params;
  return <ChatWorkspace conversationId={conversationId} />;
}
