# Contributing

## Setup

```bash
uv sync                 # core deps -- includes fastembed (memory.py needs it for
                         # real `lulu` runs); the test suite mocks embedding, no
                         # model download required to run it
uv sync --group eval    # + datasets, only needed to download evals/dbpedia's corpus
cp .env.example .env    # only needed to run `uv run lulu` against a real model --
                         # the test suite mocks the SDK, no key required
```

## Running things

```bash
uv run pytest           # the whole workspace (packages/lulu-router + packages/lulu-core)
uv run pytest packages/lulu-router   # one package at a time
uv run ruff check .
```

`pytest` runs with `--import-mode=importlib` (see `pyproject.toml`) because
`lulu-router/tests/` and `lulu-core/tests/` are both literally packages
named `tests` -- keep using relative imports inside test files
(`from .conftest import ...`, not `from tests.conftest import ...`), or
they'll collide again.

## Layout

```
packages/lulu-router/   the routing thesis, standalone -- no dependency on the harness
packages/lulu-core/     the harness itself (loop, tools, permissions, sessions)
evals/                  reproducible experiments, not part of the test suite
docs/THESIS.md          the actual argument this project is making, and its falsification criteria
```

## What a PR should look like

- **One thing.** If it grew into several unrelated changes, split it.
- **Tests alongside the code**, not after. If you're touching
  `permissions.py`, `attention.py`, `blast_radius.py`, `locks.py`, or
  `tools/sandbox.py`, add an adversarial test: a case that *tries to
  defeat* the check, not just one that confirms the happy path. Every
  existing test in those files was written that way -- e.g.
  `test_sandbox.py::test_absolute_path_to_a_sibling_directory_sharing_a_prefix_is_blocked`
  exists because a naive `str.startswith()` containment check would have
  passed a `../project-evil/` path.
- **New defaults fail closed.** An unrecognized tool defaults to `ASK`,
  not `ALLOW` (see `permissions.py`'s fail-safe default). If you add a
  new side-effecting tool, add it to `attention.MODE_PRESETS` explicitly
  -- don't rely on the fallback to cover it silently.
- **No mode/preset should be able to silently bypass a safety check.**
  Blast-radius and lock-conflict detection both escalate to `ASK`
  regardless of the active `AttentionMode`, including `AUTO`. If you add
  a new cross-cutting check like those, it should follow the same rule.
- **Cross-platform assumptions get checked, not assumed.** CI runs the
  full suite on Linux, macOS, and Windows because this project has already
  shipped two OS-specific bugs caught exactly that way (Windows silently
  corrupting line endings on every file edit; POSIX treating backslash as
  an ordinary filename character, letting a traversal string through that
  Windows correctly rejected). If you're touching path handling, don't
  assume your local OS's behavior is universal -- it probably isn't.

A CI workflow leaves an automatic comment on any PR touching the files
listed above, as a checklist reminder -- not a blocker, just a nudge.

## Releasing

Versioning follows [SemVer](https://semver.org/); the version lives in
`apps/inspector/src-tauri/tauri.conf.json` and is mirrored (kept equal,
not independently bumped) in `Cargo.toml`, `package.json`, and both
`pyproject.toml` files -- this is one product, not separately published
libraries, so one version number covers all of it.

1. Bump the version in all five files above to the same `X.Y.Z`.
2. Move `CHANGELOG.md`'s `[Unreleased]` entries under a new `## [X.Y.Z] -
   YYYY-MM-DD` heading (add one if `[Unreleased]` is empty).
3. Commit as `Release vX.Y.Z`, then tag and push:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```
4. Pushing the tag triggers `.github/workflows/release.yml`, which builds
   the Linux AppImage and Windows installer and attaches them to a
   GitHub Release as draft -- review it, then publish.

macOS isn't in the release matrix yet (untested with the PyInstaller
sidecar) -- see `ROADMAP.md`.

## Reporting a security issue

If you find a way to defeat the path sandbox, the permission system, or
the lock mechanism, please don't open a public issue -- email instead (see
the repo's contact info) so there's time to fix it before it's public.
