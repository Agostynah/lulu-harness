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
- **evals/agent_tasks/** — a small (25-task), real hand-labeled eval, the
  actual apples-to-apples test of the LLM judge's quality (DBpedia14's
  "queries" are just similarity probes, not real task questions, so it
  can't fairly judge a *content-reading* judge). Real result: the LLM judge
  matched or nearly matched the geometric judge's recall while contacting
  roughly a third as many shards.
- **evals/leakage.py** — the permission-boundary demo: 200 queries from a
  scope-restricted agent. A genuinely flat, merged index leaks across
  scopes (~26.5% leak rate at the default separation); Lulu's scoped
  routing stays at 0% leaks across all 6 strategies. Binary pass/fail, no
  interpretation needed.
- **Inspector UI** (`apps/inspector/`) — a local web app (FastAPI + SSE,
  not Electron — see the trade-off this project already worked through)
  showing the `RoutingTrace` for the turn in progress: which shards were
  contacted and why, the judge's verdict each round, the counterfactual
  savings.
- **Adversarial security review** — a full-repo red-team pass caught and
  fixed: a cross-tenant memory-isolation bug in `MemoryStore` (two scopes
  writing the same shard type were silently merged into one searchable
  store instead of staying physically separate), an unauthenticated
  path-traversal in `session.py` (`session_id` was interpolated straight
  into a filesystem path), a blocking-event-loop / nested-`asyncio.run()`
  crash risk in `server.py`'s streaming endpoint, and a narrowed
  claim-before-write ordering in the file-lock TOCTOU window. All fixed
  with regression tests, not just patched.
- **Tauri desktop shell — window wrap (step 1 of 3, see below for the
  rest).** Custom title bar (drag region, brand, min/max/close), full
  icon set, no white-flash-on-open, and a real fix for a Windows/Docker
  shell mismatch in the dev flow. The backend sidecar and a real
  installer are still open — see Next.
- **AttentionMode / scope / profile selectors + session sidebar.** All
  live-mutable through the running server (`PermissionChecker.mode`,
  `AgentLoop.system` are read fresh every turn, no session rebuild
  needed): `POST /api/sessions/{id}/mode`, per-turn memory `scope` now
  actually wired from the UI (the API already supported it, nothing
  called it), `profiles.py` (file-backed personas at
  `.lulu/profiles/<name>/persona.md`, git-versionable by design) +
  `POST /api/profiles` + `POST /api/sessions/{id}/profile`, and
  `GET /api/sessions` (session list with preview text) for the sidebar.
- **The `.env`-loading gap, closed.** Nothing in the codebase ever
  actually read `.env` before this — `.env.example`'s own comment
  claiming automatic pickup was simply wrong unless a shell had already
  exported the variable. `config.py`'s `load_dotenv`/`write_env_var`
  fixes it (no new dependency) and is what the onboarding wizard (Next)
  writes through.
- **A real provider/model-selection bug, fixed.** `build_model_client`
  passed Anthropic-style `model`/`fallback_models` defaults to every
  provider unconditionally — switching to OpenRouter/Ollama without also
  hand-editing `lulu.toml`'s model field silently sent an invalid model
  slug. Now falls back to each client's own correct default unless the
  user explicitly customized it.

## Next

Ordered so each step is independently shippable and testable before the
next one starts — not a flat backlog. Kept to what's genuinely close;
further-out ideas live in private planning notes, not here.

1. **Finish the operator-select + tiered UI onboarding flow.**
   `OperatorSelect.tsx` (a blurred background behind 3 tier cards) is
   drafted but not wired in or styled yet. Once it is: gate the title
   bar by tier (`basic` = chat only; `advanced` = mode selector +
   sessions + settings; `technomancer` = everything, exactly as it
   exists today). This unblocks the work below.
2. **API key wizard (backend already done).** `basic` = mandatory,
   can't skip, guided (pick provider → contextual help text/link that
   changes per provider → paste key → auto-configures a sane model, no
   model choice shown). `advanced` = same fields, one condensed brief
   screen, skippable. `technomancer` = no wizard at all.
3. **Model picker in the UI.** Real must-have gap — there's a
   mode/scope/profile selector today but no way to see or switch which
   *model* is active.
4. **The interactive "prove it can't leak" demo, in the product itself.**
   `evals/leakage.py` already proves scoped routing never crosses
   customer boundaries while a flat index does — expose that as a
   clickable "simulate a cross-scope query" action with a visible result,
   not something you have to read a script to verify.
5. **The Python backend as a Tauri sidecar + a real installer.**
   Compile `lulu-server` to a standalone binary (PyInstaller) per
   platform, launch it automatically on app start — what makes the
   desktop app actually installable end-to-end instead of still needing
   `uv run lulu-server` by hand.
6. **`local-only` and `quarantine` shard types.** THESIS.md already
   documents these as not-yet-built; `shards_for_scope` is the same
   primitive they need.
7. **A live MCP server behind the `remote` shard.** The connector
   (`connectors/mcp.py`) is done and tested against a fake
   `ClientSession`; it hasn't talked to a real server yet, which matters
   because the whole point of that shard is a *measured*, not simulated,
   cost.

## Later, deliberately scoped down for now

- **More judge backends.** `Judge` is already a Protocol;
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
