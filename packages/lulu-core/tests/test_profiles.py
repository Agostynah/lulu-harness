"""profiles.py: named, file-backed persona bundles."""

from __future__ import annotations

from pathlib import Path

import pytest

from lulu.profiles import (
    DEFAULT_PERSONA,
    DEFAULT_PROFILE_NAME,
    InvalidProfileNameError,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    create_profile,
    list_profiles,
    load_profile,
)


def test_list_profiles_is_just_default_with_nothing_on_disk(tmp_path: Path):
    assert list_profiles(tmp_path) == ["default"]


def test_load_default_profile_with_nothing_on_disk_returns_the_builtin_persona(tmp_path: Path):
    profile = load_profile(tmp_path, DEFAULT_PROFILE_NAME)
    assert profile.persona == DEFAULT_PERSONA


def test_load_unknown_profile_raises(tmp_path: Path):
    with pytest.raises(ProfileNotFoundError):
        load_profile(tmp_path, "does-not-exist")


def test_create_profile_writes_persona_to_disk(tmp_path: Path):
    profile = create_profile(tmp_path, "reviewer", persona="You are a strict code reviewer.")

    assert profile.name == "reviewer"
    assert profile.persona == "You are a strict code reviewer."
    on_disk = (tmp_path / ".lulu" / "profiles" / "reviewer" / "persona.md").read_text(encoding="utf-8")
    assert on_disk == "You are a strict code reviewer."


def test_created_profile_shows_up_in_list_profiles_sorted_after_default(tmp_path: Path):
    create_profile(tmp_path, "zebra", persona="z")
    create_profile(tmp_path, "alpha", persona="a")

    assert list_profiles(tmp_path) == ["default", "alpha", "zebra"]


def test_create_profile_duplicate_name_raises(tmp_path: Path):
    create_profile(tmp_path, "reviewer", persona="one")
    with pytest.raises(ProfileAlreadyExistsError):
        create_profile(tmp_path, "reviewer", persona="two")


def test_create_profile_invalid_name_raises(tmp_path: Path):
    with pytest.raises(InvalidProfileNameError):
        create_profile(tmp_path, "Not Valid!", persona="x")


def test_create_profile_with_no_persona_or_clone_from_falls_back_to_default_persona(tmp_path: Path):
    profile = create_profile(tmp_path, "blank")
    assert profile.persona == DEFAULT_PERSONA


def test_create_profile_clones_from_another_profiles_persona(tmp_path: Path):
    create_profile(tmp_path, "source", persona="the source persona")

    cloned = create_profile(tmp_path, "clone", clone_from="source")

    assert cloned.persona == "the source persona"


def test_create_profile_explicit_persona_wins_over_clone_from(tmp_path: Path):
    create_profile(tmp_path, "source", persona="the source persona")

    profile = create_profile(tmp_path, "custom", clone_from="source", persona="a totally different persona")

    assert profile.persona == "a totally different persona"


def test_load_default_profile_prefers_an_explicitly_written_default_persona(tmp_path: Path):
    create_profile(tmp_path, DEFAULT_PROFILE_NAME, persona="a customized default persona")

    profile = load_profile(tmp_path, DEFAULT_PROFILE_NAME)

    assert profile.persona == "a customized default persona"
