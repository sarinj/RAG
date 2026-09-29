"use client";

import { ArrowUp, LoaderCircle } from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";

export function MessageComposer({ disabled, busy, onSend }: {
  disabled: boolean;
  busy: boolean;
  onSend: (content: string) => void;
}) {
  const [draft, setDraft] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || disabled || busy || content.length > 8_000) return;
    onSend(content);
    setDraft("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <form className="composer-form" onSubmit={submit}>
      <label className="visually-hidden" htmlFor="message-composer">Ask a question about your sources</label>
      <textarea
        id="message-composer"
        className="message-textarea"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your documents…"
        rows={1}
        maxLength={8_000}
        disabled={disabled || busy}
        aria-describedby="composer-hint"
      />
      <div className="composer-bottom-row">
        <span id="composer-hint">Enter to send <span aria-hidden="true">·</span> Shift + Enter for a new line</span>
        <div className="composer-bottom-actions">
          <span className={draft.length > 7_500 ? "character-count character-count-near-limit" : "character-count"}>{draft.length.toLocaleString()} / 8,000</span>
          <Button type="submit" size="icon" className="send-button" disabled={disabled || busy || !draft.trim()} aria-label="Send message">
            {busy ? <LoaderCircle size={17} className="spin" aria-hidden="true" /> : <ArrowUp size={18} aria-hidden="true" />}
          </Button>
        </div>
      </div>
    </form>
  );
}
