"""Host facts for the executing agent (the agent half of ADR-0100's P3 residual #3).

An agent process runs ON the machine the work happens on, so it can *look* at the
host; the planner can only be *told* about it. That asymmetry is why concrete
filesystem paths are the executor's job:

- The planner is given the OS and an explicit rule that paths are not its
  business (``plannerDecisionRules``, ``buildHostBlock`` in the kernel). It must
  describe the target — "a folder named Reports on the desktop".
- The agent is given the block this module builds, with absolute values, and
  resolves that description against the real machine.

This split also keeps host specifics out of stored plans. Plans are replayed into
later planning as ``<PlanLTM>`` structural references, so an absolute
``C:\\Users\\someone\\Desktop\\...`` baked into a step is wrong the moment it is
recalled under a different account, and it leaks a username into memory.

History: this guidance used to live in the planner prompt inside the Scout's
``<environment>`` block. The Scout was retired 2026-08-07 and nothing produced
that block afterwards, which silently took the only "never write ``~/Desktop``"
rule with it — on Windows, where no shell expands ``~``.

Design rules, both deliberate:

- **Facts are discovered, never assumed.** A value that cannot be determined is
  OMITTED from the block rather than guessed. A wrong absolute path is worse than
  an absent one: the model treats what it is given as authoritative, so a guess
  becomes a confident write to the wrong place.
- **The block is loop-invariant.** It is computed once and cached, so the ReAct
  prompt's cacheable prefix stays byte-identical across turns.
"""

from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

__all__ = ["host_facts", "host_block", "goos"]


def goos() -> str:
    """The OS in Go's ``runtime.GOOS`` vocabulary.

    Deliberately Go's spelling rather than Python's ``sys.platform``: the kernel
    shows the planner ``runtime.GOOS``, and a step whose plan says ``windows``
    while its executor says ``win32`` invites the model to treat them as two
    different machines. One vocabulary across the two prompts.
    """
    p = sys.platform
    if p.startswith("win"):
        return "windows"
    if p == "darwin":
        return "darwin"
    if p.startswith("linux"):
        return "linux"
    # Anything else (freebsd, aix, ...) already matches Go's spelling closely
    # enough, and reporting it verbatim beats mapping it to a lie.
    return p


def _windows_desktop() -> Optional[Path]:
    """Resolve the Windows desktop, honouring OneDrive redirection.

    ``%USERPROFILE%\\Desktop`` is the wrong answer on any machine where OneDrive
    has taken over Known Folders — the real desktop is
    ``%USERPROFILE%\\OneDrive\\Desktop`` and the literal path may not exist at
    all. The registry's Shell Folders entry is what Explorer itself uses, so it
    is the only source that survives redirection.
    """
    try:
        import winreg  # type: ignore[import-not-found]  # Windows-only stdlib

        key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
            value, _ = winreg.QueryValueEx(k, "Desktop")
        if value:
            p = Path(os.path.expandvars(str(value)))
            if p.is_dir():
                return p
    except Exception:
        # No registry, no such value, or a non-Windows import error: fall through
        # to the conventional location rather than failing the whole block.
        pass
    return None


def _xdg_desktop(home: Path) -> Optional[Path]:
    """Resolve the Linux desktop from XDG, which localises the folder NAME.

    On a non-English desktop the folder is ``~/Escritorio`` or ``~/Bureau``, not
    ``~/Desktop``, so ``user-dirs.dirs`` is the only correct source.
    """
    env = os.environ.get("XDG_DESKTOP_DIR")
    if env:
        p = Path(os.path.expandvars(env.strip().strip('"')))
        if p.is_dir():
            return p
    cfg = Path(os.environ.get("XDG_CONFIG_HOME") or (home / ".config")) / "user-dirs.dirs"
    try:
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("XDG_DESKTOP_DIR"):
                continue
            _, _, raw = line.partition("=")
            raw = raw.strip().strip('"')
            # The file writes "$HOME/Desktop"; expandvars needs HOME present.
            expanded = raw.replace("$HOME", str(home))
            p = Path(os.path.expandvars(expanded))
            if p.is_dir():
                return p
    except OSError:
        pass
    return None


def _desktop(home: Optional[Path]) -> Optional[Path]:
    if goos() == "windows":
        if found := _windows_desktop():
            return found
    if home is None:
        return None
    if goos() == "linux":
        if found := _xdg_desktop(home):
            return found
    conventional = home / "Desktop"
    return conventional if conventional.is_dir() else None


@lru_cache(maxsize=1)
def host_facts() -> Dict[str, str]:
    """Discovered host facts. Keys are omitted when they cannot be determined.

    Cached: the values do not change over a process's life, and recomputing would
    break the ReAct prompt's byte-identical cacheable prefix.
    """
    facts: Dict[str, str] = {"os": goos()}

    home: Optional[Path] = None
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        home = None
    if home is not None and home.is_dir():
        facts["home"] = str(home)

    if desktop := _desktop(home):
        facts["desktop"] = str(desktop)

    try:
        facts["cwd"] = str(Path.cwd())
    except OSError:
        pass

    facts["path_separator"] = os.sep

    if release := platform.release():
        facts["os_release"] = release

    return facts


def host_block() -> str:
    """The ``<host>`` prompt section, or ``""`` if there is nothing useful to say.

    The rule text travels WITH the facts rather than sitting in a separate
    constant, because the two are only correct together: "build paths from these
    values" is meaningless without the values, and the values invite guessing
    without the rule.
    """
    facts = host_facts()
    if not facts:
        return ""

    lines = ["<host>"]
    for key in ("os", "os_release", "home", "desktop", "cwd", "path_separator"):
        if key in facts:
            lines.append(f"  {key}: {facts[key]}")
    lines.append("</host>")

    rule = [
        "HOST PATH RULES:",
        "- You are running ON the machine described in <host>. Those values are absolute and authoritative.",
        "- Build every file and folder path from them. A step that says \"a folder on the desktop\" means the"
        " <host> desktop value; a step that names a file without a directory means the <host> cwd.",
        "- NEVER write \"~\" or \"~/Desktop\" in a path or a shell command. cmd.exe and PowerShell do not"
        " expand \"~\" at all, so it becomes a literal directory of that name.",
    ]
    if facts.get("os") == "windows":
        rule.append(
            "- This host is Windows: use backslash paths and quote any path containing spaces."
            " Do not assume the desktop is under the home directory — the <host> desktop value already"
            " accounts for OneDrive redirection, so use it verbatim."
        )
    else:
        rule.append(
            "- Use forward-slash paths and quote any path containing spaces."
        )
    rule.append(
        "- If <host> does not carry a value you need (it omits what it could not determine, rather than"
        " guessing), find it out with a command instead of inventing it."
    )

    return "\n".join(lines) + "\n\n" + "\n".join(rule)
