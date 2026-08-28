# Lulu Inspector

Local web UI: chat on the left, a live **Context Assembly** panel on the
right showing the `RoutingTrace` for the turn in progress -- which memory
shards were contacted and why, the judge's verdict each expansion round,
and the counterfactual cost of `query_all`/`flat_topk` for the same query.
This is the thesis (`docs/THESIS.md`) made visible, turn by turn.

Not Electron -- a plain local web app talking to `lulu-server` over HTTP +
SSE. See `decisions_todo.md` / `ROADMAP.md` for that trade-off.

## Run it

```bash
# terminal 1, from the repo root
uv run lulu-server --root /path/to/your/project

# terminal 2
cd apps/inspector
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`, or the next
free port if that's taken).

## Verified this session, honestly scoped

Built without a browser tool available (see `decisions_todo.md`), so
verification stopped at: TypeScript compiles clean (`npm run typecheck`),
the production build succeeds (`npm run build`), and the dev server
correctly proxies `/api/*` to a real running `lulu-server` (confirmed with
curl against both the direct and proxied endpoints, not just assumed).
**Actual rendering, styling, and interaction have not been visually
confirmed** -- run it and see for yourself; report back what's broken.

## Known limitation: not token-level streaming yet

`/api/sessions/{id}/turn/stream` is real SSE transport, but the backend
runs a turn to completion first and emits the result as one burst of
events -- see `server.py`'s module docstring for why (only
`AnthropicClient` does real incremental streaming today). The UI
currently uses the plain `POST /turn` endpoint, not the stream one, for
exactly this reason: there's nothing progressive to show yet.
