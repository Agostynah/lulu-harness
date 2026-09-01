"""profiles.py: named, file-backed persona bundles -- Lulu's answer to
"Hermes profiles" (separate config, skills, and SOUL.md per profile),
scoped down to the piece that matters most for a solo/small project: the
system prompt. Each profile is a plain directory,
`.lulu/profiles/<name>/persona.md`, on purpose (not a database, not a
custom format) -- ordinary git already versions this, so "can I commit
my profile and push it to my own GitHub repo" is just "yes, it's a
markdown file," not a feature to build. See ROADMAP.md re: a future
git-init helper for `.lulu/profiles/` and why it should default private
unless the user explicitly wants to share a profile.

The "default" profile always exists even if nothing has been written to
disk yet -- DEFAULT_PERSONA is Lulu's own out-of-the-box system prompt
(what cli.py hardcoded before profiles existed), not something a user
has to create for the harness to be usable session one.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

DEFAULT_PROFILE_NAME = "default"
DEFAULT_PERSONA = (
    "You are Lulu, an agentic coding assistant. You have file and shell "
    "tools scoped to the current project. Be direct and make the "
    "requested change."
)

# Same slug convention Hermes's own "New profile" dialog uses (lowercase
# letters/digits/hyphens/underscores, must start with a letter or digit)
# -- it's a generic filesystem-safe naming rule, not anything specific
# to that product.
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class InvalidProfileNameError(ValueError):
    pass


class ProfileNotFoundError(ValueError):
    pass


class ProfileAlreadyExistsError(ValueError):
    pass


def _validate_name(name: str) -> None:
    if not _VALID_NAME.match(name):
        raise InvalidProfileNameError(
            f"invalid profile name {name!r}: must start with a lowercase letter or digit "
            "and contain only lowercase letters, digits, hyphens, and underscores"
        )


@dataclasses.dataclass
class Profile:
    name: str
    persona: str


def _profiles_dir(root: Path) -> Path:
    return root / ".lulu" / "profiles"


def _persona_path(root: Path, name: str) -> Path:
    return _profiles_dir(root) / name / "persona.md"


def list_profiles(root: Path) -> list[str]:
    """"default" always comes first, even if nothing's on disk yet --
    see the module docstring."""
    on_disk = set()
    profiles_dir = _profiles_dir(root)
    if profiles_dir.exists():
        on_disk = {p.name for p in profiles_dir.iterdir() if p.is_dir() and (p / "persona.md").exists()}
    on_disk.discard(DEFAULT_PROFILE_NAME)
    return [DEFAULT_PROFILE_NAME, *sorted(on_disk)]


def load_profile(root: Path, name: str) -> Profile:
    path = _persona_path(root, name)
    if name == DEFAULT_PROFILE_NAME:
        persona = path.read_text(encoding="utf-8") if path.exists() else DEFAULT_PERSONA
        return Profile(name=name, persona=persona)
    if not path.exists():
        raise ProfileNotFoundError(f"unknown profile {name!r}")
    return Profile(name=name, persona=path.read_text(encoding="utf-8"))


def create_profile(
    root: Path, name: str, clone_from: str | None = None, persona: str | None = None
) -> Profile:
    """persona wins if given (even cloning from another profile); empty
    persona falls back to clone_from's text; neither given falls back to
    DEFAULT_PERSONA -- mirrors Hermes's own "Clone from ... leave blank
    to keep the cloned default" behavior, which is a sensible default
    regardless of what it's borrowed from."""
    _validate_name(name)
    path = _persona_path(root, name)
    if path.exists():
        raise ProfileAlreadyExistsError(f"profile {name!r} already exists")

    if persona is not None and persona.strip():
        text = persona
    elif clone_from is not None:
        text = load_profile(root, clone_from).persona
    else:
        text = DEFAULT_PERSONA

    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="": don't let a plain write silently rewrite line endings
    # on a file the user might open/edit/diff by hand -- same reasoning
    # as tools/file_tools.py's edit_file.
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    return Profile(name=name, persona=text)
