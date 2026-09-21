"""PyInstaller entrypoint for the `lulu-server` sidecar (apps/inspector's
Tauri shell spawns the built binary instead of needing `uv run lulu-server`
by hand -- see ROADMAP.md item 5). Kept as a separate one-line script
rather than pointing PyInstaller at lulu/server.py directly, since
PyInstaller needs a real `if __name__ == "__main__"` entry file, not a
library module with a `run()` function.
"""

from lulu.server import run

if __name__ == "__main__":
    run()
