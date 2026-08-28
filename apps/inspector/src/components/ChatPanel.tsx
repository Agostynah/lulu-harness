import { useState } from "react";
import type { HistoryMessage } from "../types";

interface ChatPanelProps {
  history: HistoryMessage[];
  onSend: (prompt: string) => void;
  pending: boolean;
}

function messageText(message: HistoryMessage): string {
  if (message.tool_calls.length > 0) {
    return message.tool_calls.map((tc) => `→ ${tc.name}(${JSON.stringify(tc.arguments)})`).join("\n");
  }
  if (message.tool_results.length > 0) {
    return message.tool_results
      .map((tr) => `${tr.is_error ? "✗" : "✓"} ${tr.content}`)
      .join("\n");
  }
  return message.content;
}

function messageClass(message: HistoryMessage): string {
  if (message.tool_calls.length > 0 || message.tool_results.length > 0) return "tool";
  return message.role;
}

export default function ChatPanel({ history, onSend, pending }: ChatPanelProps) {
  const [draft, setDraft] = useState("");

  const submit = () => {
    const trimmed = draft.trim();
    if (!trimmed || pending) return;
    onSend(trimmed);
    setDraft("");
  };

  return (
    <div className="panel">
      <div className="panel-header">Chat</div>
      <div className="chat-messages">
        {history.length === 0 && (
          <div className="trace-empty">No messages yet -- say something below.</div>
        )}
        {history.map((message, i) => (
          <div key={i} className={`chat-message ${messageClass(message)}`}>
            {messageText(message)}
          </div>
        ))}
        {pending && (
          <div className="chat-message assistant" aria-live="polite">
            …
          </div>
        )}
      </div>
      <div className="chat-input-row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="Ask Lulu something…"
          disabled={pending}
        />
        <button onClick={submit} disabled={pending || !draft.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
