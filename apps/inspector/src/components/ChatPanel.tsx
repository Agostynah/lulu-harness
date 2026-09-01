import { useEffect, useRef, useState } from "react";
import type { HistoryMessage } from "../types";

interface ChatPanelProps {
  history: HistoryMessage[];
  onSend: (prompt: string) => void;
  pending: boolean;
  error?: Error | null;
}

// Turn failures (most commonly: no provider API key configured -- see
// SettingsPanel.tsx) used to be silent: the input just un-disabled
// itself with no explanation. This maps the raw fetch error (api.ts's
// json() throws "{status} {statusText}: {body}") into something a
// person can act on, falling back to the raw message for anything
// unrecognized rather than hiding it.
function friendlyError(error: Error): string {
  const message = error.message;
  if (/^(401|403)\b/.test(message) || /api.?key/i.test(message)) {
    return "The model provider rejected the request -- check that an API key is set (see the settings panel).";
  }
  if (/^50\d\b/.test(message)) {
    return "Lulu's server hit an error running that turn. Check the lulu-server terminal for details.";
  }
  if (/Failed to fetch|NetworkError/i.test(message)) {
    return "Can't reach lulu-server -- is it still running?";
  }
  return message;
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

export default function ChatPanel({ history, onSend, pending, error }: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  // The draft the user was sending when a turn fails needs to come
  // back into the input -- losing what you typed because the request
  // failed (missing key, server down, whatever) is its own papercut on
  // top of the error itself being invisible before this fix.
  const lastSentDraft = useRef("");

  useEffect(() => {
    if (error) setDraft(lastSentDraft.current);
  }, [error]);

  const submit = () => {
    const trimmed = draft.trim();
    if (!trimmed || pending) return;
    lastSentDraft.current = trimmed;
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
      {error && (
        <div className="chat-error-banner" role="alert">
          {friendlyError(error)}
        </div>
      )}
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
