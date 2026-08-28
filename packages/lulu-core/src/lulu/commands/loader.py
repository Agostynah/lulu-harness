"""commands/loader.py: loads slash-command definitions from
.lulu/commands/*.md -- same pattern local-memory already used
(.opencode/commands/mem.md). v0 doesn't parse frontmatter or templating;
a command is just a name (the filename stem) and a markdown body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Command:
    name: str
    body: str


def load_commands(commands_dir: Path) -> dict[str, Command]:
    if not commands_dir.exists():
        return {}
    return {
        path.stem: Command(name=path.stem, body=path.read_text(encoding="utf-8"))
        for path in sorted(commands_dir.glob("*.md"))
    }
