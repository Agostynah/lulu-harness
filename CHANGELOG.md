# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
The version number here is the app's -- `apps/inspector/src-tauri/tauri.conf.json`
is the single source of truth; `Cargo.toml`, `package.json`, and both
`pyproject.toml` files are kept in lockstep with it, not versioned
independently (this is one product, not a set of separately published
libraries).

## [Unreleased]

## [0.2.0] - 2026-09-21

### Added

- `JevJudge`: a content-reading memory-router judge backed by
  [TypeSafe AI's Jev](https://typesafe.ai) model, same `Judge` protocol
  slot as `GeometricJudge`/`ClaudeCLIJudge`.
- `FallbackJudge`: composes a primary and secondary judge; falls back to
  the secondary only on a genuine failure to reach a verdict, never on a
  real low-confidence one. `default_judge()` now prefers Jev whenever
  `JEV_API_KEY` is set, wrapped in `FallbackJudge` with `GeometricJudge`
  as the fallback -- no key means no network call is ever attempted.
- `POST /api/apikey/jev`: persists `JEV_API_KEY` and live-swaps the
  active judge with no restart needed, same pattern as the existing
  `POST /api/apikey` for model providers.
- `GET /api/config` now reports `judge`, `jev_configured`, and
  `provider_configured` so the UI can show real key/connection status
  instead of guessing from a failed turn.
- Settings panel rebuilt into a real API Keys & Connections panel:
  save a model-provider key or a Jev key from the UI, with explicit
  loading/success/error feedback per save -- previously read-only,
  pointing you at editing `.env` by hand.
- Linux desktop package: `lulu-server` now ships as a PyInstaller
  sidecar (`packages/lulu-core/sidecar_entry.py`), launched
  automatically by the Tauri shell and killed on app exit -- no `uv` or
  terminal needed on the machine running the packaged app. Produces a
  working `Lulu_amd64.AppImage`; see the README's Quick start.
- `tauri-plugin-log` now runs in release builds too, not just debug --
  writes to the platform log directory, which is what makes sidecar
  spawn failures in a packaged build diagnosable at all.

### Fixed

- `api.ts`'s relative `fetch("/api/...")` only ever worked under Vite's
  dev proxy; the packaged app has no dev server in front of it, so it
  never reached the sidecar. Now uses an absolute `API_BASE` outside dev.
- The server's CORS allowlist didn't include the packaged webview's real
  origin (`tauri://localhost` / `http://tauri.localhost`), so even a
  reachable sidecar had its responses blocked client-side.
- `configQuery`/`sessionMutation`'s retry budget was too short for the
  sidecar's cold start (a PyInstaller onefile binary unpacks itself on
  every launch), surfacing as a permanent "can't reach lulu-server"
  error screen instead of just waiting.

## [0.1.0] - 2026-09-20 and earlier

Everything up to and including the Tauri desktop shell (window wrap),
tiered UI selectors, profiles, and `.env` loading -- see `ROADMAP.md`'s
"Done" section for the full list; this file starts tracking releases
from 0.2.0 onward.

[Unreleased]: https://github.com/Agostynah/lulu-harness/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Agostynah/lulu-harness/releases/tag/v0.2.0
