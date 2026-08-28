# Roadmap

What's built, what isn't yet, and why — so this doesn't have to live only
in commit history or a private notes file.

## Done

- **The router** (`packages/lulu-router`): all 6 strategies from the paper,
  KMeans partitioning, scope-based permission boundary, two interchangeable
  judges (geometric + LLM-backed), full DBpedia14 replication with real,
  reported numbers (including the negative/nuanced ones).
- **The harness** (`packages/lulu-core`): three model providers behind
  the same `ModelClient` protocol -- Anthropic, plus OpenRouter and Ollama
  sharing one `OpenAICompatibleClient` implementation (proving "any
  model, same API" is a real property of the code, not a README claim),
  each enforcing the requirement that the model actually supports
  tool-calling since the harness's loop has no fallback path for a model
  that can't emit structured tool calls. Sandboxed file tools + bash,
  `/trace` and `/cost` rendering, session persistence and resume, an MCP
  connector (unit-tested against a fake session, not yet a live server —
  see below), and a CLI entrypoint with `--provider`/`--mode` overrides.
- **The attention-interface pilot**: four `AttentionMode` presets
  (`manual`/`plan`/`auto_edits`/`auto`) over the permission system, plus
  `suggest_promotions()` — a non-ML approval-streak suggestion that reads
  the same log `/cost` already writes. It only ever suggests; nothing here
  can grant itself more autonomy without an explicit human confirmation.
- **Cross-session file locking**: two concurrent `lulu` processes on the
  same project won't silently race on the same file — an advisory,
  TTL-based claim escalates the second one to a human `ASK`, tested
  end-to-end (not just in isolated units) through two full `main()` runs.
- **CI/CD**: tests + lint on Linux/macOS/Windows, secret scanning, a small
  bot that flags PRs touching permission-relevant files, Dependabot. Found
  a real cross-platform bug (backslash traversal handled differently on
  POSIX) before it ever shipped.

## Next

- **evals/agent_tasks/** — a small (~25 task), real hand-labeled eval, the
  actual apples-to-apples test of the LLM judge's quality (DBpedia14's
  "queries" are just similarity probes, not real task questions, so it
  can't fairly judge a *content-reading* judge).
- **evals/leakage.py** — the permission-boundary demo: 200 queries from a
  scope-restricted agent, `flat_topk` leaks across scopes and routing
  doesn't. Binary pass/fail, no interpretation needed.
- **A live MCP server behind the `remote` shard.** The connector
  (`connectors/mcp.py`) is done and tested against a fake `ClientSession`;
  it hasn't talked to a real server yet, which matters because the whole
  point of that shard is a *measured*, not simulated, cost.
- **Inspector UI** (`apps/inspector/`) — a local web app (FastAPI + SSE,
  not Electron — see the trade-off this project already worked through)
  showing the `RoutingTrace` for the turn in progress: which shards were
  contacted and why, the judge's verdict each round, the counterfactual
  savings. Explicitly the first thing to cut if time is short; the CLI's
  `/trace` and `/cost` already surface the same data as text.

## Later, deliberately scoped down for now

- **More judge backends.** Same story — `Judge` is already a Protocol;
  `OllamaJudge` (fully local, $0, nothing leaves the machine) is the
  natural next one.
- **Memory as the communication channel between concurrent sessions.**
  The file lock (done) stops two sessions from racing; it says nothing
  about what either one is *doing*. The router already gives every session
  a shared, scope-permissioned place to write "here's what I found" and
  have another session's next retrieval surface it — no new subsystem,
  reusing the same mechanism a third time.
- **A real learning loop over `suggest_promotions()`.** Today it's a
  streak count over the permissions log. The log itself is already being
  written; a genuinely learned attention policy is a separate, much larger
  project, not something to half-build.
- **Volatility-aware re-indexing** — a code shard invalidates on every
  commit, an episodic one almost never does; re-index only what's actually
  stale instead of everything.
- **A Tauri desktop wrap** once the web UI exists — same UI, ~1 day of
  work, no rewrite, an order of magnitude smaller than the Electron
  alternative this project already ruled out.
