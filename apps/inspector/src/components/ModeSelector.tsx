import { useEffect, useRef, useState } from "react";
import { ATTENTION_MODES, type AttentionMode } from "../types";

// Surfaces the one thing about Lulu that was completely invisible in
// the UI before this: which AttentionMode a session is running under,
// and lets you change it live (see server.py's /mode endpoint --
// PermissionChecker.mode is a plain mutable attribute, so this takes
// effect on the very next tool call, no reconnect needed). This is the
// actual "security mindset" mechanism the project is built around; it
// deserves to be a first-class, always-visible control, not something
// buried in a config file only the CLI reads.
const MODE_INFO: Record<AttentionMode, { label: string; hint: string; dot: string }> = {
  manual: { label: "Manual", hint: "Every tool call asks first", dot: "var(--text-dim)" },
  plan: { label: "Plan", hint: "Read-only -- no writes execute", dot: "#60a5fa" },
  auto_edits: { label: "Auto-edits", hint: "File edits run; risky ops still ask", dot: "var(--accent)" },
  auto: { label: "Auto", hint: "Most tools run without asking", dot: "var(--success)" },
};

interface ModeSelectorProps {
  mode: AttentionMode | null;
  onChange: (mode: AttentionMode) => void;
}

export default function ModeSelector({ mode, onChange }: ModeSelectorProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClickAway = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClickAway);
    return () => document.removeEventListener("mousedown", onClickAway);
  }, [open]);

  if (!mode) return null;
  const info = MODE_INFO[mode];

  return (
    <div className="mode-selector" ref={ref}>
      <button className="mode-selector-trigger" onClick={() => setOpen((v) => !v)}>
        <span className="mode-dot" style={{ background: info.dot }} />
        {info.label}
      </button>
      {open && (
        <div className="mode-menu" role="menu">
          {ATTENTION_MODES.map((m) => (
            <button
              key={m}
              className={`mode-menu-item${m === mode ? " active" : ""}`}
              role="menuitem"
              onClick={() => {
                onChange(m);
                setOpen(false);
              }}
            >
              <span className="mode-dot" style={{ background: MODE_INFO[m].dot }} />
              <span className="mode-menu-text">
                <span className="mode-menu-label">{MODE_INFO[m].label}</span>
                <span className="mode-menu-hint">{MODE_INFO[m].hint}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
