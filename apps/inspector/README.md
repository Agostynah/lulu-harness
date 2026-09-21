# Lulu Inspector

Chat on the left, a live **Context Assembly** panel on the right showing
the `RoutingTrace` for the turn in progress -- which memory shards were
contacted and why, the judge's verdict each expansion round (including
which judge: geometric, Jev, or the fallback between them), and the
counterfactual cost of `query_all`/`flat_topk` for the same query. This
is the thesis (`docs/THESIS.md`) made visible, turn by turn.

A local web app (FastAPI + SSE, not Electron -- see `decisions_todo.md`
/ `ROADMAP.md` for that trade-off) wrapped in a Tauri desktop shell
(`src-tauri/`). Runs standalone in a browser for dev, or as a packaged
app where `lulu-server` ships as a bundled sidecar -- no terminal needed.

## Run it from source (dev)

```bash
# terminal 1, from the repo root
uv run lulu-server --root /path/to/your/project

# terminal 2
cd apps/inspector
npm install
npm run dev
```

Open the URL Vite prints (`http://localhost:5183` -- fixed via
`strictPort` in `vite.config.ts` so the Tauri shell's `devUrl` always
knows where to find it; see that file's comment for why 5183 and not
5173). API keys (model provider and Jev) can be set from the app itself
via the settings panel, or by hand in `.env` -- see the root README's
`.env.example`.

## Run it packaged (Linux)

```bash
# from the repo root -- see the root README's Quick start for the full,
# always-current version of this
uv run --with pyinstaller --with pyinstaller-hooks-contrib pyinstaller \
  --onefile --name lulu-server --paths packages/lulu-core/src \
  --paths packages/lulu-router/src packages/lulu-core/sidecar_entry.py
cp dist/lulu-server src-tauri/binaries/lulu-server-x86_64-unknown-linux-gnu
NO_STRIP=1 cargo tauri build --bundles appimage
```

Produces `src-tauri/target/release/bundle/appimage/Lulu_<version>_amd64.AppImage`
-- double-click it, no `uv`/Python/Node needed on the machine running it.
Windows/macOS installers aren't built by hand this way yet; see
`ROADMAP.md` and `.github/workflows/release.yml` for the CI path.

## Known limitation: not token-level streaming yet

`/api/sessions/{id}/turn/stream` is real SSE transport, but the backend
runs a turn to completion first and emits the result as one burst of
events -- see `server.py`'s module docstring for why (only
`AnthropicClient` does real incremental streaming today). The UI
currently uses the plain `POST /turn` endpoint, not the stream one, for
exactly this reason: there's nothing progressive to show yet.
