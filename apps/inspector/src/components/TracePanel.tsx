import { useState } from "react";
import type { CostResponse, RoutingTrace } from "../types";

interface TracePanelProps {
  trace: RoutingTrace | null;
  cost: CostResponse | null;
}

function BudgetBar({ trace }: { trace: RoutingTrace }) {
  const pct = trace.budget.max_tokens > 0 ? Math.min(100, (trace.spent.tokens / trace.budget.max_tokens) * 100) : 0;
  return (
    <div className="budget-bar">
      <div className="section-title">Budget spent this turn</div>
      <div className="budget-row">
        <span>{trace.spent.tokens} tok</span>
        <span>
          {trace.spent.latency_ms.toFixed(0)}ms · ${trace.spent.usd.toFixed(4)}
        </span>
      </div>
      <div className="budget-track">
        <div className="budget-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ShardList({ trace }: { trace: RoutingTrace }) {
  return (
    <div>
      <div className="section-title">
        Shards ({trace.shards_considered.filter((s) => s.contacted).length}/{trace.shards_considered.length} contacted)
      </div>
      <div className="shard-list">
        {trace.shards_considered.map((shard) => (
          <div key={shard.shard_id} className="shard-row">
            <span className={`shard-mark ${shard.contacted ? "yes" : "no"}`}>
              {shard.contacted ? "✓" : "✗"}
            </span>
            <span className="shard-id">{shard.shard_id}</span>
            <span className="shard-reason">
              {shard.contacted
                ? `centroid=${shard.centroid_similarity.toFixed(2)}`
                : shard.skip_reason ?? "skipped"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RoundsList({ trace }: { trace: RoutingTrace }) {
  if (trace.rounds.length === 0) return null;
  return (
    <div>
      <div className="section-title">Judge verdicts ({trace.judge})</div>
      <div className="rounds-list">
        {trace.rounds.map((round) => (
          <div key={round.round_index} className="round-row">
            [{round.round_index}] {round.shard_id}: {round.confidence_before.toFixed(2)} →{" "}
            {round.confidence_after.toFixed(2)} <span className={`round-verdict ${round.verdict}`}>{round.verdict}</span>
            {round.reasoning && <div className="round-reasoning">{round.reasoning}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultsToggle({ trace }: { trace: RoutingTrace }) {
  const [open, setOpen] = useState(false);
  if (trace.results.length === 0) return null;
  return (
    <div>
      <button className="results-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} what entered the prompt ({trace.results.length})
      </button>
      {open && (
        <div className="results-list">
          {trace.results.map((r) => (
            <div key={r.id} className="result-row">
              <span className="score">{r.score.toFixed(2)}</span>
              {r.content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CostFooter({ cost }: { cost: CostResponse }) {
  return (
    <div className="cost-panel">
      <div className="cost-row spent">
        <span>spent</span>
        <span>
          {cost.spent.tokens} tok · {cost.spent.latency_ms.toFixed(0)}ms · ${cost.spent.usd.toFixed(4)}
        </span>
      </div>
      {cost.counterfactuals.map((cf) => (
        <div key={cf.label} className="cost-row">
          <span>{cf.label} would have been</span>
          <span>
            {cf.cost.tokens} tok
            {" · "}
            <span className="cost-savings">
              {cf.tokens_saved_pct >= 0 ? "+" : ""}
              {cf.tokens_saved_pct.toFixed(0)}% tok
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}

export default function TracePanel({ trace, cost }: TracePanelProps) {
  return (
    <div className="panel">
      <div className="panel-header">Context Assembly</div>
      {trace === null ? (
        <div className="trace-empty">No trace yet -- send a message to see how Lulu routed memory for it.</div>
      ) : (
        <div className="trace-body">
          <div>
            <div className="section-title">
              strategy={trace.strategy} · judge={trace.judge}
            </div>
            <div className="confidence-gauge">confidence: {trace.confidence.toFixed(2)}</div>
          </div>
          <BudgetBar trace={trace} />
          <ShardList trace={trace} />
          <RoundsList trace={trace} />
          <ResultsToggle trace={trace} />
        </div>
      )}
      {cost && <CostFooter cost={cost} />}
    </div>
  );
}
