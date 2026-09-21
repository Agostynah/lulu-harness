import type { UseMutationResult } from "@tanstack/react-query";
import { useState } from "react";
import type { LuluConfigResponse } from "../types";

interface SettingsPanelProps {
  config: LuluConfigResponse | null;
  onClose: () => void;
  onSaveProviderKey: (apiKey: string) => void;
  providerKeyMutation: UseMutationResult<{ provider: string }, Error, { provider: string; apiKey: string }>;
  onSaveJevKey: (apiKey: string) => void;
  jevKeyMutation: UseMutationResult<{ judge: string }, Error, string>;
}

// Extracted from api.ts's json() error format ("{status} {statusText}: {body}")
// -- a raw "400 Bad Request: {"detail":"..."}" is not something a person
// reading a settings panel should have to parse themselves.
function friendlyKeyError(error: Error): string {
  try {
    const bodyStart = error.message.indexOf("{");
    if (bodyStart === -1) return error.message;
    const detail = JSON.parse(error.message.slice(bodyStart))?.detail;
    return typeof detail === "string" ? detail : error.message;
  } catch {
    return error.message;
  }
}

// Only the read-only status fields, not `mutate` itself (KeyRow calls the
// passed `onSave` callback instead, same pattern as every other App.tsx
// mutation) -- UseMutationResult's `mutate` is contravariant in its
// variables type, so a full UseMutationResult<..., unknown> parameter
// can't accept the specific mutation types App.tsx actually passes down.
type MutationStatus = Pick<
  UseMutationResult<unknown, Error, unknown>,
  "isPending" | "isSuccess" | "isError" | "error" | "reset"
>;

// One row: a masked key input, a Save button, and every state Save can be
// in (idle/pending/success/error) rendered explicitly -- per the "feedback
// on every action" rule, a save button that just goes back to normal with
// no confirmation is indistinguishable from having done nothing.
function KeyRow({
  label,
  placeholder,
  configuredLabel,
  configured,
  onSave,
  mutation,
}: {
  label: string;
  placeholder: string;
  configuredLabel: string;
  configured: boolean | null;
  onSave: (apiKey: string) => void;
  mutation: MutationStatus;
}) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || mutation.isPending) return;
    onSave(trimmed);
  };

  return (
    <div className="settings-key-row">
      <div className="settings-key-header">
        <span className="settings-label">{label}</span>
        {configured !== null && (
          <span className={`settings-badge ${configured ? "configured" : "missing"}`}>
            {configured ? `${configuredLabel} configured` : `${configuredLabel} missing`}
          </span>
        )}
      </div>
      <div className="settings-key-input-row">
        <input
          type="password"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            if (mutation.isSuccess || mutation.isError) mutation.reset();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder={placeholder}
          disabled={mutation.isPending}
        />
        <button onClick={submit} disabled={mutation.isPending || !value.trim()}>
          {mutation.isPending ? "Saving…" : "Save"}
        </button>
      </div>
      {mutation.isSuccess && <p className="settings-key-feedback success">Saved -- active immediately, no restart needed.</p>}
      {mutation.isError && (
        <p className="settings-key-feedback error" role="alert">
          {friendlyKeyError(mutation.error as Error)}
        </p>
      )}
    </div>
  );
}

export default function SettingsPanel({
  config,
  onClose,
  onSaveProviderKey,
  providerKeyMutation,
  onSaveJevKey,
  jevKeyMutation,
}: SettingsPanelProps) {
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <span>API Keys & Connections</span>
          <button className="settings-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        {config ? (
          <div className="settings-body">
            <div className="settings-section">
              <div className="settings-section-title">Model provider</div>
              <div className="settings-row">
                <span className="settings-label">provider</span>
                <span className="settings-value">{config.provider}</span>
              </div>
              <div className="settings-row">
                <span className="settings-label">model</span>
                <span className="settings-value">{config.model}</span>
              </div>
              {config.provider === "ollama" ? (
                <p className="settings-hint">Ollama runs locally -- no API key needed.</p>
              ) : (
                <KeyRow
                  label={`${config.provider} API key`}
                  placeholder="sk-…"
                  configuredLabel="Key"
                  configured={config.provider_configured}
                  onSave={onSaveProviderKey}
                  mutation={providerKeyMutation}
                />
              )}
            </div>

            <div className="settings-section">
              <div className="settings-section-title">Connections</div>
              <div className="settings-row">
                <span className="settings-label">memory judge</span>
                <span className="settings-value">{config.judge}</span>
              </div>
              <p className="settings-hint">
                Jev (from TypeSafe AI) reads memory-shard content to decide when Lulu has enough
                context, instead of the geometric score-gap heuristic. Falls back to it
                automatically if Jev is unreachable.
              </p>
              <KeyRow
                label="Jev API key"
                placeholder="apikey_…"
                configuredLabel="Jev"
                configured={config.jev_configured}
                onSave={onSaveJevKey}
                mutation={jevKeyMutation}
              />
            </div>
          </div>
        ) : (
          <p className="settings-hint">Not connected to lulu-server yet.</p>
        )}
      </div>
    </div>
  );
}
