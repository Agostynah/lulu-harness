import type { CostResponse, HistoryMessage, LuluConfigResponse, SessionCreateResponse, TurnResult } from "./types";

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

export async function createSession(sessionId?: string): Promise<SessionCreateResponse> {
  const url = sessionId ? `/api/sessions?session_id=${encodeURIComponent(sessionId)}` : "/api/sessions";
  return json(await fetch(url, { method: "POST" }));
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

export async function getHistory(sessionId: string): Promise<{ session_id: string; history: HistoryMessage[] }> {
  return json(await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/history`));
}

export async function getCost(sessionId: string): Promise<CostResponse> {
  return json(await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/cost`));
}
