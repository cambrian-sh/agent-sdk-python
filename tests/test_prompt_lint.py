"""The prompt lint over the SDK prompts (cambrian-core/prompts/SPEC.md §5).

One rule set, two runners: cambrian-core/prompts/lint_test.go runs it over the core
prompts, this file over ``cambrian_agent_sdk/prompts``. The loader enforces the
structural rules L1-L5 (a violation fails ``PromptLibrary.load``); this file adds
the style rules L6-L7 and the warnings. The acronym allowlist and the shared lint
cases are compared with the Go runner's through the parity fixture
(test_prompt_hash_parity.py).
"""
import re
import warnings

import pytest

from cambrian_agent_sdk import prompt_library as pl

ACRONYMS = frozenset((
    "API", "CLI", "CSV", "DAG", "HTML", "HTTP", "HTTPS", "ID", "IDS", "ISO", "JSON",
    "LLM", "MCP", "OK", "PDF", "SDK", "SQL", "URI", "URL", "UTC", "UUID", "XML", "YAML",
))
MAX_CAPS_RUN = 3
MIN_SHOUT_WORDS = 2

_ADR = re.compile(r"\bADR[-_ ]?\d{3,4}\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9_']+")


def _is_caps_word(w: str) -> bool:
    if any("a" <= c <= "z" for c in w):
        return False
    return sum("A" <= c <= "Z" for c in w) >= 2


def _is_acronym(w: str) -> bool:
    return w.strip("'") in ACRONYMS


def lint_body(body: str):
    """L6 and L7 over a normalised body -> (failures, warnings)."""
    failures, warns = [], []
    m = _ADR.search(body)
    if m:
        failures.append(f"L6: decision number in model-facing text: {m.group(0)}")
    run = []

    def flush():
        shout = sum(1 for w in run if not _is_acronym(w))
        if len(run) >= MAX_CAPS_RUN and shout >= MIN_SHOUT_WORDS:
            failures.append("L7: shouting: " + " ".join(run))
        run.clear()

    for w in _WORD.findall(body):
        if _is_caps_word(w):
            run.append(w)
            if not _is_acronym(w) and len(w.strip("'")) >= 3:
                warns.append(f"emphasis by capitals: {w}")
            continue
        flush()
    flush()
    return failures, warns


def _sdk_prompts():
    return list(pl.PromptLibrary.load(pl.package_root()))


def test_sdk_prompts_load():
    # L1-L5: the loader rejects the tree on the first structural violation.
    lib = pl.PromptLibrary.load(pl.package_root())
    assert lib.ids(), "no SDK prompts packaged"
    # Every v*.md on disk was loaded (a misnamed v01.md would be unreachable).
    on_disk = [f for d in pl.package_root().iterdir() if d.is_dir()
               for f in d.iterdir() if f.name.endswith(".md")]
    assert len(on_disk) == len(list(lib))


@pytest.mark.parametrize("prompt", _sdk_prompts(), ids=lambda p: f"{p.id}-v{p.version}")
def test_sdk_prompt_style(prompt):
    failures, warns = lint_body(prompt.body)
    for w in warns:
        warnings.warn(f"{prompt.id}/v{prompt.version}: {w}")
    assert not failures, failures


@pytest.mark.parametrize("body,fail", [
    ("Output EXACTLY ONE JSON object.", True),
    ("See ADR-0139 for why.", True),
    ("as decided in adr 052", True),
    ("YOU MUST NEVER do this", True),
    ("NEVER EVER do this", False),
    ("Reply with JSON via the HTTP API.", False),
    ("Do NOT output JSON.", False),
    ("It FAILED, so say WHY and WHAT to change.", False),
])
def test_lint_catches_clear_violations(body, fail):
    assert bool(lint_body(body)[0]) == fail


def test_single_emphasis_word_is_a_warning():
    assert len(lint_body("Do NOT output JSON.")[1]) == 1
