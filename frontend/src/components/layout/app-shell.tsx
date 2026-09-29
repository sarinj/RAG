"use client";

import { BookOpenText, MessageSquareText, Plus, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const onDocuments = pathname.startsWith("/documents");
  const onChat = pathname.startsWith("/chat");

  return (
    <div className="app-frame">
      <aside className="sidebar">
        <Link className="brand" href="/documents" aria-label="Sourcebook home">
          <span className="brand-mark"><BookOpenText size={18} strokeWidth={2.3} aria-hidden="true" /></span>
          <span>sourcebook</span>
        </Link>

        <div className="sidebar-workspace">
          <span className="workspace-avatar">S</span>
          <span className="workspace-copy"><strong>My workspace</strong><small>Personal library</small></span>
          <span className="workspace-dot" aria-hidden="true" />
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          <span className="sidebar-label">Workspace</span>
          <Link className={cn("nav-link", onDocuments && "nav-link-active")} href="/documents" aria-current={onDocuments ? "page" : undefined}>
            <BookOpenText size={17} aria-hidden="true" />
            <span>Documents</span>
          </Link>
          <Link className={cn("nav-link", onChat && "nav-link-active")} href="/chat" aria-current={onChat ? "page" : undefined}>
            <MessageSquareText size={17} aria-hidden="true" />
            <span>Conversations</span>
          </Link>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="note-icon"><Sparkles size={14} aria-hidden="true" /></span>
            <p>Answers stay grounded in the sources you choose.</p>
          </div>
          <div className="sidebar-profile">
            <span className="profile-avatar">S</span>
            <span className="profile-copy"><strong>Local workspace</strong><small>Only you can see this</small></span>
            <span className="local-lock" aria-label="Local workspace"><span aria-hidden="true">●</span></span>
          </div>
        </div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className="topbar-context">
            <span className="topbar-kicker">PERSONAL KNOWLEDGE BASE</span>
            <span className="topbar-divider" aria-hidden="true">/</span>
            <span>{onDocuments ? "Library" : onChat ? "Chat" : "Workspace"}</span>
          </div>
          <Link href="/chat" className="button button-primary button-sm topbar-action">
            <Plus size={15} aria-hidden="true" /> New conversation
          </Link>
        </header>
        <main className="main-content">{children}</main>
      </div>
    </div>
  );
}
