import type { Metadata } from "next";
import { NewChatPage } from "@/features/chat";

export const metadata: Metadata = { title: "New conversation" };

export default function ChatPage() {
  return <NewChatPage />;
}
