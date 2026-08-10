"""Host-facts tests (ADR-0100 P3 residual #3, agent half).

The defect these pin: when the Scout was retired, the ONLY instruction telling
anything not to emit ``~/Desktop`` went with it — on Windows, where no shell
expands ``~``. So the assertions below are mostly about what the block REFUSES to
say (a guessed path) and what it always says (never write ``~``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cambrian_agent_sdk import host as host_mod
from cambrian_agent_sdk.host import goos, host_block, host_facts


@pytest.fixture(autouse=True)
def _clear_cache():
    """host_facts is lru_cached for prompt-cache stability; tests need it fresh."""
    host_facts.cache_clear()
    yield
    host_facts.cache_clear()


def test_goos_uses_go_vocabulary_not_python_platform():
    # The kernel shows the planner runtime.GOOS. If the agent said "win32" while
    # the plan said "windows", the model has two names for one machine.
    assert goos() in {"windows", "darwin", "linux"} or goos() == sys.platform
    if sys.platform.startswith("win"):
        assert goos() == "windows", "must not leak Python's win32 spelling"


def test_facts_always_carry_an_os():
    assert host_facts()["os"] == goos()


def test_paths_that_are_reported_actually_exist():
    # The whole point of the module: a value present in the block is authoritative,
    # so anything it reports must be real. A guessed path is worse than an absent
    # one because the model writes to it confidently.
    facts = host_facts()
    for key in ("home", "desktop", "cwd"):
        if key in facts:
            assert Path(facts[key]).is_dir(), f"{key}={facts[key]!r} was reported but is not a directory"


def test_desktop_is_omitted_rather_than_guessed(monkeypatch, tmp_path):
    """A home directory with no Desktop must yield NO desktop key.

    This is the mutation-sensitive one: the obvious wrong implementation is
    ``home / "Desktop"`` unconditionally, which passes every other test here and
    fails this one.
    """
    fake_home = tmp_path / "nohome"
    fake_home.mkdir()
    monkeypatch.setattr(host_mod.Path, "home", staticmethod(lambda: fake_home))
    # Neutralise the platform-specific lookups so only the conventional path is left.
    monkeypatch.setattr(host_mod, "_windows_desktop", lambda: None)
    monkeypatch.setattr(host_mod, "_xdg_desktop", lambda home: None)
    host_facts.cache_clear()

    facts = host_facts()
    assert facts["home"] == str(fake_home)
    assert "desktop" not in facts, "a non-existent desktop must be omitted, never guessed"
    # Check for the FACT LINE, not the substring: the rule prose legitimately talks
    # about the desktop, so a bare `"desktop" not in block` would fail for the wrong
    # reason and mask a regression.
    fact_lines = [ln.strip() for ln in host_block().splitlines() if ln.startswith("  ")]
    assert not any(ln.startswith("desktop:") for ln in fact_lines), fact_lines


def test_desktop_is_reported_when_it_does_exist(monkeypatch, tmp_path):
    # The negative test above is only meaningful if the positive case works.
    fake_home = tmp_path / "withdesktop"
    (fake_home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(host_mod.Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr(host_mod, "_windows_desktop", lambda: None)
    monkeypatch.setattr(host_mod, "_xdg_desktop", lambda home: None)
    host_facts.cache_clear()

    assert host_facts()["desktop"] == str(fake_home / "Desktop")


def test_windows_desktop_wins_over_the_conventional_path(monkeypatch, tmp_path):
    """OneDrive redirection: the registry value must beat ``home/Desktop``.

    On a redirected machine the real desktop is under OneDrive and
    ``%USERPROFILE%\\Desktop`` may not exist at all, so preferring the
    conventional path would report a directory the user does not see.
    """
    fake_home = tmp_path / "user"
    (fake_home / "Desktop").mkdir(parents=True)
    redirected = tmp_path / "user" / "OneDrive" / "Desktop"
    redirected.mkdir(parents=True)

    monkeypatch.setattr(host_mod.Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr(host_mod, "goos", lambda: "windows")
    monkeypatch.setattr(host_mod, "_windows_desktop", lambda: redirected)
    host_facts.cache_clear()

    assert host_facts()["desktop"] == str(redirected)


def test_xdg_desktop_wins_on_linux(monkeypatch, tmp_path):
    # A localised desktop is ~/Escritorio, not ~/Desktop.
    fake_home = tmp_path / "user"
    localised = fake_home / "Escritorio"
    localised.mkdir(parents=True)
    (fake_home / "Desktop").mkdir()

    monkeypatch.setattr(host_mod.Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr(host_mod, "goos", lambda: "linux")
    monkeypatch.setattr(host_mod, "_windows_desktop", lambda: None)
    monkeypatch.setattr(host_mod, "_xdg_desktop", lambda home: localised)
    host_facts.cache_clear()

    assert host_facts()["desktop"] == str(localised)


def test_xdg_parses_user_dirs_file(monkeypatch, tmp_path):
    # Exercise the real parser, not a stub: it must expand $HOME and strip quotes.
    home = tmp_path / "h"
    (home / ".config").mkdir(parents=True)
    (home / "Bureau").mkdir()
    (home / ".config" / "user-dirs.dirs").write_text(
        '# comment\nXDG_DESKTOP_DIR="$HOME/Bureau"\nXDG_MUSIC_DIR="$HOME/Musique"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("XDG_DESKTOP_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))

    assert host_mod._xdg_desktop(home) == home / "Bureau"


def test_block_forbids_tilde_explicitly():
    # The rule the Scout's deletion silently removed. Its absence is the defect.
    block = host_block()
    assert "~" in block, "the block must mention ~ in order to forbid it"
    assert "NEVER" in block
    lowered = block.lower()
    assert "do not expand" in lowered or "does not" in lowered or "not expand" in lowered


def test_block_names_the_shells_that_do_not_expand_tilde():
    block = host_block()
    assert "cmd.exe" in block and "PowerShell" in block


def test_block_is_stable_across_calls():
    # Loop-invariance: react.py puts this in the cacheable prefix, so two calls
    # differing would silently break prompt caching every round.
    assert host_block() == host_block()


def test_block_reports_the_separator_for_this_platform():
    assert f"path_separator: {os.sep}" in host_block()


def test_windows_block_says_backslash_and_warns_about_onedrive(monkeypatch):
    monkeypatch.setattr(host_mod, "goos", lambda: "windows")
    host_facts.cache_clear()
    block = host_block()
    assert "backslash" in block
    assert "OneDrive" in block


def test_non_windows_block_says_forward_slash(monkeypatch):
    monkeypatch.setattr(host_mod, "goos", lambda: "linux")
    host_facts.cache_clear()
    assert "forward-slash" in host_block()
