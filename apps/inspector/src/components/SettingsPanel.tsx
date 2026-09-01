import type { LuluConfigResponse } from "../types";

interface SettingsPanelProps {
  config: LuluConfigResponse | null;
  onClose: () => void;
}

// Read-only for now: writing a key from the UI needs a backend endpoint
// that persists it (lulu.toml or an OS credential store) and this doesn't
// exist yet -- see ROADMAP.md's Tauri desktop shell entry, step 3. This
// panel exists so the provider/key story is visible and dismissible
// instead of a blocking setup wizard, not to fully replace editing .env.
export default function SettingsPanel({ config, onClose }: SettingsPanelProps) {
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <span>Provider</span>
          <button className="settings-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        {config ? (
          <div className="settings-body">
            <div className="settings-row">
              <span className="settings-label">provider</span>
              <span className="settings-value">{config.provider}</span>
            </div>
            <div className="settings-row">
              <span className="settings-label">model</span>
              <span className="settings-value">{config.model}</span>
            </div>
            <div className="settings-row">
              <span className="settings-label">mode</span>
              <span className="settings-value">{config.attention_mode}</span>
            </div>
            <p className="settings-hint">
              Set the matching API key in this project's <code>.env</code> (see{" "}
              <code>.env.example</code>) and restart <code>lulu-server</code> to change
              provider or model.
            </p>
          </div>
        ) : (
          <p className="settings-hint">Not connected to lulu-server yet.</p>
        )}
      </div>
    </div>
  );
}
