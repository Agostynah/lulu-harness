import { useEffect, useState } from "react";
import type { AttentionMode, UiTier } from "../types";
import ModeSelector from "./ModeSelector";
import ProfileSelector from "./ProfileSelector";
import ScopeSelector from "./ScopeSelector";
import TierSwitcher from "./TierSwitcher";

// The bar itself (brand, mode selector, settings) renders in BOTH the
// plain-browser dev flow and the Tauri desktop shell -- it's not just
// window chrome, the mode selector is a real functional control that
// has to work wherever the app runs. Only the native window controls
// (min/max/close) and the drag-to-move region are Tauri-specific, since
// decorations:false there means the OS draws no title bar at all and
// this component has to supply those buttons itself; in a plain
// browser tab the browser's own chrome already provides them, so
// rendering a second close button would be redundant, not helpful.
const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

interface TitleBarProps {
  onToggleSettings: () => void;
  onToggleSessions: () => void;
  mode: AttentionMode | null;
  onChangeMode: (mode: AttentionMode) => void;
  scope: string | null;
  knownScopes: string[];
  onChangeScope: (scope: string | null) => void;
  onAddScope: (scope: string) => void;
  profile: string | null;
  profiles: string[];
  onLoadProfiles: () => void;
  onChangeProfile: (profile: string) => void;
  onCreateProfile: (name: string, cloneFrom: string, persona: string) => void;
  creatingProfile: boolean;
  createProfileError: string | null;
  tier: UiTier;
  onChangeTier: (tier: UiTier) => void;
}

// Which controls each tier sees -- see decisions_todo.md's onboarding
// plan for why: `basic` should feel like "just chat," `advanced` gets
// the operationally useful controls without the niche ones, and
// `technomancer` is everything that exists today, unchanged. The
// TierSwitcher itself is exempt from this -- it always shows, in every
// tier, since it's the only way to get from one tier's UI to another's.
const TIER_VISIBILITY: Record<UiTier, { profile: boolean; mode: boolean; scope: boolean; sessions: boolean }> = {
  basic: { profile: false, mode: false, scope: false, sessions: false },
  advanced: { profile: false, mode: true, scope: false, sessions: true },
  technomancer: { profile: true, mode: true, scope: true, sessions: true },
};

export default function TitleBar({
  onToggleSettings,
  onToggleSessions,
  mode,
  onChangeMode,
  scope,
  knownScopes,
  onChangeScope,
  onAddScope,
  profile,
  profiles,
  onLoadProfiles,
  onChangeProfile,
  onCreateProfile,
  creatingProfile,
  createProfileError,
  tier,
  onChangeTier,
}: TitleBarProps) {
  const visible = TIER_VISIBILITY[tier];
  const [isMaximized, setIsMaximized] = useState(false);
  const [appWindow, setAppWindow] = useState<Awaited<ReturnType<typeof import("@tauri-apps/api/window").getCurrentWindow>> | null>(null);

  useEffect(() => {
    if (!isTauri) return;
    import("@tauri-apps/api/window").then(({ getCurrentWindow }) => {
      const win = getCurrentWindow();
      setAppWindow(win);
      win.isMaximized().then(setIsMaximized);
      const unlisten = win.onResized(() => {
        win.isMaximized().then(setIsMaximized);
      });
      return () => {
        unlisten.then((fn) => fn());
      };
    });
  }, []);

  const dragProps = isTauri ? { "data-tauri-drag-region": true } : {};

  // data-tauri-drag-region is the passive/declarative mechanism and
  // should be enough on its own, but this codebase's own experience
  // with every OTHER window command (minimize/close/toggle-maximize/
  // resize) needing an explicit core:window:allow-* capability grant
  // (capabilities/default.json) before it actually worked is reason
  // enough not to assume drag-region is exempt from that pattern.
  // Calling startDragging() explicitly on mousedown is a second, direct
  // path that doesn't depend on getting that assumption right -- only
  // wired on titlebar-brand (icon + name, no buttons inside it), not
  // the whole bar, so it can never swallow a click meant for one of the
  // actual controls in titlebar-actions.
  const startDrag = () => {
    if (appWindow) appWindow.startDragging();
  };

  return (
    <div className="titlebar" {...dragProps}>
      <div className="titlebar-brand" {...dragProps} onMouseDown={startDrag}>
        <img src="/lulu-mark.png" alt="" className="titlebar-icon" draggable={false} />
        <span className="titlebar-name">Lulu</span>
      </div>
      <div className="titlebar-actions">
        <TierSwitcher tier={tier} onChange={onChangeTier} />
        {visible.profile && (
          <ProfileSelector
            profile={profile}
            profiles={profiles}
            onOpenDropdown={onLoadProfiles}
            onChange={onChangeProfile}
            onCreate={onCreateProfile}
            creating={creatingProfile}
            createError={createProfileError}
          />
        )}
        {visible.mode && <ModeSelector mode={mode} onChange={onChangeMode} />}
        {visible.scope && (
          <ScopeSelector scope={scope} knownScopes={knownScopes} onChange={onChangeScope} onAddScope={onAddScope} />
        )}
        {visible.sessions && (
          <button
            className="titlebar-icon-btn"
            title="Sessions"
            onClick={onToggleSessions}
            aria-label="Sessions"
          >
            <HistoryIcon />
          </button>
        )}
        <button
          className="titlebar-icon-btn"
          title="Provider settings"
          onClick={onToggleSettings}
          aria-label="Provider settings"
        >
          <GearIcon />
        </button>
        {isTauri && appWindow && (
          <div className="titlebar-window-controls">
            <button
              className="titlebar-window-btn"
              title="Minimize"
              aria-label="Minimize"
              onClick={() => appWindow.minimize()}
            >
              <MinimizeIcon />
            </button>
            <button
              className="titlebar-window-btn"
              title={isMaximized ? "Restore" : "Maximize"}
              aria-label={isMaximized ? "Restore" : "Maximize"}
              onClick={() => appWindow.toggleMaximize()}
            >
              {isMaximized ? <RestoreIcon /> : <MaximizeIcon />}
            </button>
            <button
              className="titlebar-window-btn titlebar-close"
              title="Close"
              aria-label="Close"
              onClick={() => appWindow.close()}
            >
              <CloseIcon />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function GearIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path
        d="M8 10a2 2 0 100-4 2 2 0 000 4z"
        stroke="currentColor"
        strokeWidth="1.3"
      />
      <path
        d="M13 8.5a5 5 0 000-1l1.2-1-1-1.7-1.4.5a5 5 0 00-.9-.5L10.6 3H9.4l-.3 1.3a5 5 0 00-.9.5l-1.4-.5-1 1.7L7 7.5a5 5 0 000 1l-1.2 1 1 1.7 1.4-.5c.28.2.58.37.9.5l.3 1.3h1.2l.3-1.3c.32-.13.62-.3.9-.5l1.4.5 1-1.7-1.2-1z"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8.5" r="5.5" stroke="currentColor" strokeWidth="1.3" />
      <path d="M8 5.5V8.5L10 10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5 2.2A5.5 5.5 0 003 4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function MinimizeIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <line x1="2" y1="6" x2="10" y2="6" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  );
}

function MaximizeIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <rect x="2.5" y="2.5" width="7" height="7" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  );
}

function RestoreIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <rect x="3.5" y="1.5" width="6" height="6" stroke="currentColor" strokeWidth="1" />
      <rect x="1.5" y="3.5" width="6" height="6" stroke="currentColor" strokeWidth="1" fill="var(--bg)" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <line x1="2" y1="2" x2="10" y2="10" stroke="currentColor" strokeWidth="1.1" />
      <line x1="10" y1="2" x2="2" y2="10" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  );
}
