import { useEffect, useRef, useState } from "react";

// Chrome-profile-style picker, but for something real Lulu already
// has: memory `scope` (the permission boundary evals/leakage.py proves
// -- see docs/THESIS.md). This does NOT switch sessions or chat
// history, only which scope tag rides along on the *next* message sent
// (matching how the CLI's own --scope flag and server.py's per-turn
// `scope` field already work) -- a fuller "separate world per profile,
// like Chrome" would mean a session-per-scope model the backend doesn't
// have yet, so this is scoped honestly to what's actually there today.
const AVATAR_COLORS = ["#a78bfa", "#60a5fa", "#4ade80", "#fbbf24", "#f87171", "#f472b6"];

function colorFor(scope: string): string {
  let hash = 0;
  for (let i = 0; i < scope.length; i++) hash = (hash * 31 + scope.charCodeAt(i)) >>> 0;
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
}

interface ScopeSelectorProps {
  scope: string | null;
  knownScopes: string[];
  onChange: (scope: string | null) => void;
  onAddScope: (scope: string) => void;
}

export default function ScopeSelector({ scope, knownScopes, onChange, onAddScope }: ScopeSelectorProps) {
  const [open, setOpen] = useState(false);
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClickAway = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setAdding(false);
      }
    };
    document.addEventListener("mousedown", onClickAway);
    return () => document.removeEventListener("mousedown", onClickAway);
  }, [open]);

  const submitNewScope = () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    onAddScope(trimmed);
    onChange(trimmed);
    setDraft("");
    setAdding(false);
    setOpen(false);
  };

  return (
    <div className="scope-selector" ref={ref}>
      <button className="scope-selector-trigger" title="Memory scope" onClick={() => setOpen((v) => !v)}>
        {scope ? (
          <span className="scope-avatar" style={{ background: colorFor(scope) }}>
            {scope[0]?.toUpperCase()}
          </span>
        ) : (
          <span className="scope-avatar scope-avatar-personal">
            <HomeIcon />
          </span>
        )}
      </button>
      {open && (
        <div className="scope-menu" role="menu">
          <button className={`scope-menu-item${scope === null ? " active" : ""}`} onClick={() => onChange(null)}>
            <span className="scope-avatar scope-avatar-personal">
              <HomeIcon />
            </span>
            Personal (unscoped)
          </button>
          {knownScopes.map((s) => (
            <button
              key={s}
              className={`scope-menu-item${s === scope ? " active" : ""}`}
              onClick={() => onChange(s)}
            >
              <span className="scope-avatar" style={{ background: colorFor(s) }}>
                {s[0]?.toUpperCase()}
              </span>
              {s}
            </button>
          ))}
          <div className="scope-menu-divider" />
          {adding ? (
            <div className="scope-menu-add-row">
              <input
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submitNewScope();
                  if (e.key === "Escape") setAdding(false);
                }}
                placeholder="scope name…"
              />
              <button onClick={submitNewScope}>Add</button>
            </div>
          ) : (
            <button className="scope-menu-item" onClick={() => setAdding(true)}>
              <span className="scope-avatar scope-avatar-add">+</span>
              New scope
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function HomeIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
      <path d="M2 7.5L8 2l6 5.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3.5 6.5V13.5h9V6.5" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  );
}
