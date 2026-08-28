"""Blast-radius detection for bash commands.

Unlike sandbox.py's path containment check, this is NOT a sound boundary
-- it's string heuristics over a shell command, and shell syntax has a
hundred ways to smuggle intent past a naive filter (quoting, `$()`,
variable expansion, base64, whatever). That's precisely why file tools are
sandboxed structurally (resolve_within_root) instead of by pattern-
matching their arguments, and why bash itself is not "sandboxed" at all --
see tools/sandbox.py's module docstring.

What this module IS for: catching the common, unremarkable cases (an
agent about to run `rm -rf .`, `sudo`, a network fetch, or a forced
git operation) so those get an extra "ask" even in a mode that would
otherwise auto-approve. Every heuristic here is deliberately tuned to
over-trigger rather than under-trigger: a false positive costs one
unnecessary confirmation prompt, a false negative is a security miss.
Given that asymmetry, err toward asking.
"""

from __future__ import annotations

import re

_RM_TOKEN = re.compile(r"\brm\b", re.IGNORECASE)
_SUDO_TOKEN = re.compile(r"\bsudo\b", re.IGNORECASE)
_NETWORK_TOKEN = re.compile(r"\b(curl|wget|nc|netcat|ssh|scp|ftp)\b", re.IGNORECASE)
_GIT_FORCE_PUSH = re.compile(r"\bgit\s+push\b.*(--force\b|-f\b)", re.IGNORECASE)
_GIT_RESET_HARD = re.compile(r"\bgit\s+reset\b.*--hard\b", re.IGNORECASE)
_GIT_CLEAN_FORCE = re.compile(r"\bgit\s+clean\b.*-\w*f", re.IGNORECASE)
_FIND_DELETE = re.compile(r"\bfind\b.*-delete\b", re.IGNORECASE)
_DD_TOKEN = re.compile(r"\bdd\b.*\bof=", re.IGNORECASE)
_MKFS_TOKEN = re.compile(r"\bmkfs\b", re.IGNORECASE)
_SHRED_TOKEN = re.compile(r"\bshred\b", re.IGNORECASE)
_TRUNCATE_ZERO = re.compile(r"\btruncate\b.*-s\s*0\b", re.IGNORECASE)


def _looks_like_rm_rf(command: str) -> bool:
    """Flags a `rm` invocation that combines recursive + force, in any of
    the common spellings (-rf, -fr, -r -f, --recursive --force, mixed
    short/long). Scans only the flag tokens immediately following an `rm`
    token, stopping at the first non-flag token (the target path) --
    which keeps `rm important_file_rf.txt` from matching just because the
    filename happens to contain the letters."""
    if not _RM_TOKEN.search(command):
        return False
    tokens = command.split()
    rm_indices = [i for i, t in enumerate(tokens) if t.lower() == "rm"]
    for idx in rm_indices:
        short_flags = ""
        long_flags: set[str] = set()
        for t in tokens[idx + 1 :]:
            if t.startswith("--"):
                long_flags.add(t.lower())
            elif t.startswith("-") and len(t) > 1:
                short_flags += t[1:].lower()
            else:
                break
        has_recursive = "r" in short_flags or "--recursive" in long_flags
        has_force = "f" in short_flags or "--force" in long_flags
        if has_recursive and has_force:
            return True
    return False


def assess_blast_radius(command: str) -> list[str]:
    """Returns the list of matched blast-radius reasons for a bash
    command (empty if none matched). Multiple reasons can co-occur
    (e.g. `sudo rm -rf /`)."""
    reasons: list[str] = []
    if _looks_like_rm_rf(command):
        reasons.append("recursive force delete (rm -rf or equivalent)")
    if _SUDO_TOKEN.search(command):
        reasons.append("privilege escalation (sudo)")
    if _NETWORK_TOKEN.search(command):
        reasons.append("network egress (curl/wget/nc/ssh/scp/ftp)")
    if _GIT_FORCE_PUSH.search(command):
        reasons.append("force push (rewrites remote history)")
    if _GIT_RESET_HARD.search(command):
        reasons.append("git reset --hard (discards local changes)")
    if _GIT_CLEAN_FORCE.search(command):
        reasons.append("git clean -f (deletes untracked files)")
    if _FIND_DELETE.search(command):
        reasons.append("find -delete (bulk deletes matched files)")
    if _DD_TOKEN.search(command):
        reasons.append("dd with of= (raw device/file overwrite)")
    if _MKFS_TOKEN.search(command):
        reasons.append("mkfs (formats a filesystem, destroys existing data)")
    if _SHRED_TOKEN.search(command):
        reasons.append("shred (irreversibly overwrites file contents)")
    if _TRUNCATE_ZERO.search(command):
        reasons.append("truncate -s 0 (destroys file contents in place)")
    return reasons
