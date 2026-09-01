import type {
  AttentionMode,
  CostResponse,
  HistoryMessage,
  LuluConfigResponse,
  ProfileListResponse,
  SessionCreateResponse,
  SessionListResponse,
  TurnResult,
} from "./types";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return response.json() as Promise<T>;
}

export async function getConfig(): Promise<LuluConfigResponse> {
  return json(await fetch("/api/config"));
}

export async function createSession(sessionId?: string, profile?: string): Promise<SessionCreateResponse> {
  const params = new URLSearchParams();
  if (sessionId) params.set("session_id", sessionId);
  if (profile) params.set("profile", profile);
  const qs = params.toString();
  return json(await fetch(`/api/sessions${qs ? `?${qs}` : ""}`, { method: "POST" }));
}

export async function listSessions(): Promise<SessionListResponse> {
  return json(await fetch("/api/sessions"));
}

export async function listProfiles(): Promise<ProfileListResponse> {
  return json(await fetch("/api/profiles"));
}

export async function createProfile(
  name: string,
  cloneFrom?: string,
  persona?: string
): Promise<{ name: string; persona: string }> {
  return json(
    await fetch("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, clone_from: cloneFrom ?? null, persona: persona ?? null }),
    })
  );
}

export async function setProfile(sessionId: string, profile: string): Promise<{ session_id: string; profile: string }> {
  return json(
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile }),
    })
  );
}

export async function runTurn(sessionId: string, prompt: string, scope?: string): Promise<TurnResult> {
  return json(
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, scope: scope ?? null }),
    })
  );
}

export async function getHistory(
  sessionId: string
): Promise<{ session_id: string; history: HistoryMessage[]; attention_mode: AttentionMode; profile: string }> {
  return json(await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/history`));
}

export async function getCost(sessionId: string): Promise<CostResponse> {
  return json(await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/cost`));
}

export async function setMode(
  sessionId: string,
  mode: AttentionMode
): Promise<{ session_id: string; attention_mode: AttentionMode }> {
  return json(
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    })
  );
}
