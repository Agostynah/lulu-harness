// Mirrors the JSON shapes server.py actually emits (dataclasses.asdict()
// on lulu_router's RoutingTrace/Budget/Cost/ShardScore/ExpansionRound/
// SearchResult, and server.py's own _serialize_history/_serialize_turn_result).
// Kept as one file, hand-written against the real backend rather than
// generated, so a shape drift shows up as a TypeScript error at build
// time, not a silent runtime mismatch.

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  tool_call_id: string;
  content: string;
  is_error: boolean;
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
  tool_calls: ToolCall[];
  tool_results: ToolResult[];
}

export interface Budget {
  max_tokens: number;
  max_latency_ms: number;
  max_usd: number;
}

export interface Cost {
  latency_ms: number;
  usd: number;
  tokens: number;
}

export interface ShardScore {
  shard_id: string;
  centroid_similarity: number;
  contacted: boolean;
  skip_reason: string | null;
}

export interface ExpansionRound {
  round_index: number;
  shard_id: string;
  confidence_before: number;
  confidence_after: number;
  judge: string;
  verdict: "sufficient" | "expand";
  reasoning: string | null;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface RoutingTrace {
  query: string;
  strategy: string;
  judge: string;
  budget: Budget;
  shards_considered: ShardScore[];
  rounds: ExpansionRound[];
  confidence: number;
  spent: Cost;
  results: SearchResult[];
  timestamp: number;
}

export interface UsageTotals {
  input_tokens: number;
  output_tokens: number;
}

export interface TurnResult {
  final_text: string;
  stopped_reason: "final_text" | "max_iterations";
  iterations: number;
  trace: RoutingTrace | null;
  usage_totals: UsageTotals;
}

export const ATTENTION_MODES = ["manual", "plan", "auto_edits", "auto"] as const;
export type AttentionMode = (typeof ATTENTION_MODES)[number];

// Picked once via OperatorSelect.tsx, persisted in localStorage
// (App.tsx's UI_TIER_STORAGE_KEY) so it's never asked again on later
// launches. Purely a frontend display-complexity concept -- the backend
// has no idea a tier exists, nothing here changes what Lulu can DO.
export const UI_TIERS = ["basic", "advanced", "technomancer"] as const;
export type UiTier = (typeof UI_TIERS)[number];

export interface SessionCreateResponse {
  session_id: string;
  history: HistoryMessage[];
  attention_mode: AttentionMode;
  profile: string;
}

export interface SessionSummary {
  session_id: string;
  preview: string;
  modified_at: number;
}

export interface SessionListResponse {
  sessions: SessionSummary[];
}

export interface ProfileListResponse {
  profiles: string[];
}

export interface LuluConfigResponse {
  provider: string;
  model: string;
  attention_mode: AttentionMode;
  root: string;
}

export interface Counterfactual {
  label: string;
  cost: Cost;
  tokens_saved_pct: number;
}

export interface CostResponse {
  spent: Cost;
  counterfactuals: Counterfactual[];
}
