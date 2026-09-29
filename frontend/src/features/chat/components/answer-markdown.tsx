"use client";

import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import React, { isValidElement, type ReactElement, type ReactNode } from "react";
import { Citation } from "../api";

function withCitationMarkers(node: ReactNode, citations: Map<string, Citation>, onCitation: (citation: Citation) => void): ReactNode {
  if (typeof node === "string") {
    const parts = node.split(/(\[S\d+\])/g);
    return parts.map((part, index) => {
      const match = /^\[(S\d+)\]$/.exec(part);
      if (!match) return part;
      const citation = citations.get(match[1]);
      if (!citation) return <span className="citation-unresolved" key={`${part}-${index}`} title="Citation details are unavailable">{part}</span>;
      return (
        <button
          className="citation-marker"
          key={`${part}-${index}`}
          type="button"
          onClick={() => onCitation(citation)}
          aria-label={`Open source ${citation.label} from ${citation.documentName}`}
          title={`${citation.documentName}${citation.heading ? ` · ${citation.heading}` : ""}`}
        >{part}</button>
      );
    });
  }
  if (Array.isArray(node)) return node.map((child, index) => <span key={index}>{withCitationMarkers(child, citations, onCitation)}</span>);
  if (isValidElement<{ children?: ReactNode }>(node)) {
    const element = node as ReactElement<{ children?: ReactNode; className?: string }>;
    if (element.props.className === "citation-marker" || element.props.className === "citation-unresolved") return node;
    if (element.props.children === undefined) return node;
    return React.cloneElement(element, undefined, withCitationMarkers(element.props.children, citations, onCitation));
  }
  return node;
}

export function AnswerMarkdown({ content, citations, onCitation }: {
  content: string;
  citations: Citation[];
  onCitation: (citation: Citation) => void;
}) {
  const citationMap = new Map(citations.map((citation) => [citation.label, citation]));
  const safeUrl = (url: string) => {
    if (/^(https?:|mailto:|#)/i.test(url) || (url.startsWith("/") && !url.startsWith("//"))) return url;
    return "";
  };
  const decorate = (children: ReactNode) => withCitationMarkers(children, citationMap, onCitation);

  return (
    <div className="answer-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
        urlTransform={safeUrl}
        components={{
          p: ({ children, ...props }) => <p {...props}>{decorate(children)}</p>,
          li: ({ children, ...props }) => <li {...props}>{decorate(children)}</li>,
          h1: ({ children, ...props }) => <h1 {...props}>{decorate(children)}</h1>,
          h2: ({ children, ...props }) => <h2 {...props}>{decorate(children)}</h2>,
          h3: ({ children, ...props }) => <h3 {...props}>{decorate(children)}</h3>,
          td: ({ children, ...props }) => <td {...props}>{decorate(children)}</td>,
          th: ({ children, ...props }) => <th {...props}>{decorate(children)}</th>,
          blockquote: ({ children, ...props }) => <blockquote {...props}>{decorate(children)}</blockquote>,
          a: ({ href, children, ...props }) => <a {...props} href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
