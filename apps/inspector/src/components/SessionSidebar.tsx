import type { SessionSummary } from "../types";

// Hidden by default, toggled from a round button in the title bar --
// matches the "start simple, reveal on demand" direction this was built
// under, rather than Hermes's always-visible sidebar taking up a third
// of the window from the first launch.
interface SessionSidebarProps {
  open: boolean;
  sessions: SessionSummary[];
  activeSessionId: string | undefined;
  onSelect: (sessionId: string) => void;
  onNewSession: () => void;
  onClose: () => void;
}

function relativeTime(unixSeconds: number): string {
  const diffMs = Date.now() - unixSeconds * 1000;
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function SessionSidebar({
  open,
  sessions,
  activeSessionId,
  onSelect,
  onNewSession,
  onClose,
}: SessionSidebarProps) {
  if (!open) return null;

  return (
    <>
      <div className="session-sidebar-overlay" onClick={onClose} />
      <div className="session-sidebar">
        <div className="session-sidebar-header">
          <span>Sessions</span>
          <button className="session-sidebar-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <button className="session-new-btn" onClick={onNewSession}>
          + New session
        </button>
        <div className="session-list">
          {sessions.length === 0 && <div className="session-list-empty">No past sessions yet.</div>}
          {sessions.map((s) => (
            <button
              key={s.session_id}
              className={`session-list-item${s.session_id === activeSessionId ? " active" : ""}`}
              onClick={() => onSelect(s.session_id)}
            >
              <span className="session-list-preview">{s.preview}</span>
              <span className="session-list-time">{relativeTime(s.modified_at)}</span>
            </button>
          ))}
        </div>
      </div>
    </>
  );
}
