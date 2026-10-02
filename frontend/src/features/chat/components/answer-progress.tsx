"use client";

import { CheckCircle2, ChevronDown, ChevronRight, CircleDashed, ListChecks, LoaderCircle } from "lucide-react";
import { useDeferredValue, useId, useState } from "react";
import type { Citation } from "../api";
import type { StreamReply } from "../streaming";
import { AnswerMarkdown } from "./answer-markdown";

const NO_CITATIONS: Citation[] = [];
const ignoreCitation = () => undefined;

export function AnswerProgress({ reply }: { reply?: StreamReply }) {
  const [expanded, setExpanded] = useState(true);
  const id = useId();
  if (!reply || reply.steps.length === 0) return null;
  const active = reply.status === "active";
  return (
    <section className="answer-progress" aria-label="Answer progress">
      <button type="button" className="answer-progress-toggle" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded} aria-controls={id}>
        {active ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <ListChecks size={14} aria-hidden="true" />}
        <span>{active ? "Working with your sources" : reply.status === "error" ? "Answer interrupted" : "How this answer was prepared"}</span>
        {expanded ? <ChevronDown size={13} aria-hidden="true" /> : <ChevronRight size={13} aria-hidden="true" />}
      </button>
      {active && <p className="answer-progress-current" role="status" aria-live="polite">{reply.latestMessage}</p>}
      <ol id={id} className="answer-progress-steps" hidden={!expanded}>
        {reply.steps.map((step) => (
          <li key={step.stage} className={`answer-progress-step answer-progress-${step.status}`}>
            {active && step.status === "active"
              ? <LoaderCircle size={12} className="spin" aria-hidden="true" />
              : step.status === "complete"
                ? <CheckCircle2 size={12} aria-hidden="true" />
                : <CircleDashed size={12} aria-hidden="true" />}
            <span>{step.message}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function StreamingDraft({ reply }: { reply?: StreamReply }) {
  const draft = useDeferredValue(reply?.draft ?? "");
  return (
    <div className="streaming-reply" aria-busy={reply ? reply.status === "active" : true}>
      <AnswerProgress reply={reply} />
      {draft && reply?.status === "active" ? (
        <div className="assistant-answer streaming-answer">
          <span className="streaming-draft-label">Draft · checking sources</span>
          <AnswerMarkdown content={draft} citations={NO_CITATIONS} onCitation={ignoreCitation} />
          <span className="streaming-cursor" aria-hidden="true" />
        </div>
      ) : (
        <div className="pending-answer" role="status">
          <span>{reply?.latestMessage ?? "Searching your sources"}</span>
          <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
        </div>
      )}
    </div>
  );
}
