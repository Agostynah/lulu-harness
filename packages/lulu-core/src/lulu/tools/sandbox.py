"""Path sandboxing: the actual security boundary for file tools.

Every file tool resolves user-supplied paths through here before touching
disk. The bash tool does NOT get an equivalent sandbox -- string-inspecting
shell commands for safety is a well-known losing game (quoting, encoding,
`$()`, pipes, and a hundred other ways to smuggle a path past a naive
filter). The real boundary for bash is permissions.py: it defaults to
"ask" and blast-radius detection forces "ask" regardless of mode. Don't
try to rebuild that here with string matching on file paths, which at
least admits a sound check (fully resolve, then verify containment).
"""

from __future__ import annotations

from pathlib import Path


class PathEscapesRootError(Exception):
    def __init__(self, user_path: str, root: Path):
        self.user_path = user_path
        self.root = root
        super().__init__(f"{user_path!r} resolves outside project root {root}")


def is_within_root(root: Path, path: Path) -> bool:
    """Sound containment check on two already-resolved paths. Factored out
    of resolve_within_root so glob/grep results -- which `Path.glob()`
    will happily return *outside* root when the pattern itself contains
    `..` (confirmed empirically: `root.glob("../secret.txt")` matches) --
    can be filtered through the same check rather than a separate,
    weaker one."""
    root = root.resolve()
    path = path.resolve()
    return path == root or root in path.parents


def resolve_within_root(root: Path, user_path: str) -> Path:
    """Resolves `user_path` (relative or absolute, forward or back
    slashes, with however many `..` segments) against `root` and raises
    PathEscapesRootError unless the fully-resolved result is root itself
    or something under it.

    Resolving *before* checking containment is what makes this sound: a
    string check on the unresolved path ("does it contain '..'?") is
    bypassable by an absolute path, a mixed-separator path, or a path that
    both goes up and back down further than root. `Path.resolve()`
    collapses all of that to a single canonical absolute path first, and
    only then is containment checked -- there's no clever input that
    survives resolution and still lands outside root.

    Backslashes are normalized to forward slashes before anything else,
    on every platform, not just Windows. Caught by CI, not assumed:
    `pathlib.PosixPath` treats a backslash as an ordinary filename
    character with no separator meaning at all, so `"..\\..\\secret.txt"` run through
    this function on Linux was silently resolving to a harmless (if
    oddly-named) file *inside* root instead of being rejected -- a
    traversal attempt that only got blocked by accident of which OS
    happened to be running the harness. The string a model emits doesn't
    know or care what OS it'll run on; the sandboxing shouldn't either.
    """
    root = root.resolve()
    normalized = user_path.replace("\\", "/")
    candidate = Path(normalized)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    if not is_within_root(root, resolved):
        raise PathEscapesRootError(user_path, root)
    return resolved
