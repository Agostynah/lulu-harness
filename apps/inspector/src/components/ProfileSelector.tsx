import { useEffect, useRef, useState } from "react";

// The real "Hermes profile" equivalent -- see the conversation this was
// built in: a profile is a named persona (system prompt), file-backed at
// .lulu/profiles/<name>/persona.md (profiles.py), so git already
// versions it -- no custom versioning system, no GitHub API
// integration. Switching a session's profile changes AgentLoop.system
// live (server.py's /profile endpoint), same mutable-attribute pattern
// as AttentionMode and scope.
interface ProfileSelectorProps {
  profile: string | null;
  profiles: string[];
  onOpenDropdown: () => void;
  onChange: (profile: string) => void;
  onCreate: (name: string, cloneFrom: string, persona: string) => void;
  creating: boolean;
  createError: string | null;
}

export default function ProfileSelector({
  profile,
  profiles,
  onOpenDropdown,
  onChange,
  onCreate,
  creating,
  createError,
}: ProfileSelectorProps) {
  const [open, setOpen] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClickAway = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClickAway);
    return () => document.removeEventListener("mousedown", onClickAway);
  }, [open]);

  return (
    <div className="profile-selector" ref={ref}>
      <button
        className="profile-selector-trigger"
        title="Profile (persona)"
        onClick={() => {
          setOpen((v) => !v);
          onOpenDropdown();
        }}
      >
        <PersonaIcon />
        {profile ?? "default"}
      </button>
      {open && (
        <div className="profile-menu" role="menu">
          {profiles.map((p) => (
            <button
              key={p}
              className={`profile-menu-item${p === profile ? " active" : ""}`}
              onClick={() => {
                onChange(p);
                setOpen(false);
              }}
            >
              {p}
            </button>
          ))}
          <div className="scope-menu-divider" />
          <button
            className="profile-menu-item"
            onClick={() => {
              setOpen(false);
              setModalOpen(true);
            }}
          >
            + New profile
          </button>
        </div>
      )}
      {modalOpen && (
        <NewProfileModal
          profiles={profiles}
          creating={creating}
          createError={createError}
          onCancel={() => setModalOpen(false)}
          onCreate={(name, cloneFrom, persona) => {
            onCreate(name, cloneFrom, persona);
            setModalOpen(false);
          }}
        />
      )}
    </div>
  );
}

const NAME_PATTERN = /^[a-z0-9][a-z0-9_-]*$/;

function NewProfileModal({
  profiles,
  creating,
  createError,
  onCancel,
  onCreate,
}: {
  profiles: string[];
  creating: boolean;
  createError: string | null;
  onCancel: () => void;
  onCreate: (name: string, cloneFrom: string, persona: string) => void;
}) {
  const [name, setName] = useState("");
  const [cloneFrom, setCloneFrom] = useState(profiles[0] ?? "default");
  const [persona, setPersona] = useState("");

  const nameValid = NAME_PATTERN.test(name);

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span>New profile</span>
          <button className="settings-close" onClick={onCancel} aria-label="Close">
            ×
          </button>
        </div>
        <p className="modal-subtitle">Profiles are independent personas: a separate system prompt.</p>

        <label className="modal-field">
          <span className="modal-field-label">Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-profile" autoFocus />
          <span className="modal-field-hint">
            Lowercase letters, digits, hyphens, and underscores. Must start with a letter or digit.
          </span>
        </label>

        <label className="modal-field">
          <span className="modal-field-label">Clone from</span>
          <select value={cloneFrom} onChange={(e) => setCloneFrom(e.target.value)}>
            {profiles.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <span className="modal-field-hint">Copies the selected profile's persona as a starting point.</span>
        </label>

        <label className="modal-field">
          <span className="modal-field-label">Persona -- optional</span>
          <textarea
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            placeholder="The system prompt for this profile. Leave blank to keep the cloned persona."
            rows={4}
          />
        </label>

        {createError && <div className="modal-error">{createError}</div>}

        <div className="modal-actions">
          <button className="modal-cancel" onClick={onCancel}>
            Cancel
          </button>
          <button
            disabled={!nameValid || creating}
            onClick={() => onCreate(name, cloneFrom, persona)}
          >
            {creating ? "Creating…" : "Create profile"}
          </button>
        </div>
      </div>
    </div>
  );
}

function PersonaIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="5.5" r="3" stroke="currentColor" strokeWidth="1.3" />
      <path d="M2.5 14c0-3 2.5-5 5.5-5s5.5 2 5.5 5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}
