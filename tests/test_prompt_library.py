"""The SDK prompt library loader (ADR-0139 C4; cambrian-core/prompts/SPEC.md)."""
from pathlib import Path

import pytest

from cambrian_agent_sdk import prompt_library as pl
from cambrian_agent_sdk.reflection import build_reflection_prompt

_GOLDEN = Path(__file__).resolve().parent / "golden"
_FRONT = "id: greet\nversion: {v}\ntier: stable\ndescription: Say hello.\ninputs: [name]\n"


def _write(root: Path, rel: str, text: str) -> None:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(text.encode("utf-8"))


def _file(front: str, body: str) -> str:
    return "---\n" + front + "---\n" + body


def test_render_fills_slots_and_hashes_the_static_text(tmp_path):
    _write(tmp_path, "greet/v1.md", _file(_FRONT.format(v=1), "Hello {{name}}, bye {{name}}.\n"))
    p = pl.PromptLibrary.load(tmp_path).get("greet")
    a = p.render({"name": "Ada"})
    b = p.render({"name": "a much longer {{name}} value"})
    assert a.text == "Hello Ada, bye Ada."
    assert "a much longer {{name}} value" in b.text      # single pass
    assert a.static_hash == b.static_hash
    assert a.static_hash == pl.static_hash("Hello {{}}, bye {{}}.")
    assert (a.id, a.version) == ("greet", 1)


def test_render_rejects_missing_and_extra_slots():
    p = pl.Prompt(id="x", version=1, tier="stable", inputs=("a", "b"), description="d",
                  body="{{a}} {{b}}")
    with pytest.raises(pl.SlotMismatch):
        p.render({"a": "1"})
    with pytest.raises(pl.SlotMismatch):
        p.render({"a": "1", "b": "2", "c": "3"})
    assert p.render({"a": "", "b": ""}).text == " "


def test_get_latest_and_get_version(tmp_path):
    _write(tmp_path, "greet/v1.md", _file(_FRONT.format(v=1), "one {{name}}"))
    _write(tmp_path, "greet/v2.md", _file(_FRONT.format(v=2), "two {{name}}"))
    lib = pl.PromptLibrary.load(tmp_path)
    assert lib.get("greet").version == 2
    assert lib.get_version("greet", 1).body == "one {{name}}"
    with pytest.raises(pl.PromptNotFound):
        lib.get_version("greet", 3)
    with pytest.raises(pl.PromptNotFound):
        lib.get("nope")


_BAD = {
    "version gap": {"greet/v1.md": _file(_FRONT.format(v=1), "{{name}}"),
                    "greet/v3.md": _file(_FRONT.format(v=3), "{{name}}")},
    "version mismatch": {"greet/v1.md": _file(_FRONT.format(v=2), "{{name}}")},
    "id mismatch": {"other/v1.md": _file(_FRONT.format(v=1), "{{name}}")},
    "bad tier": {"greet/v1.md": _file(_FRONT.format(v=1).replace("stable", "fixed"), "{{name}}")},
    "unknown field": {"greet/v1.md": _file(_FRONT.format(v=1) + "respons_schema: x.schema.json\n", "{{name}}")},
    "missing field": {"greet/v1.md": _file(_FRONT.format(v=1).replace("description: Say hello.\n", ""), "{{name}}")},
    "bare inputs": {"greet/v1.md": _file(_FRONT.format(v=1).replace("inputs: [name]", "inputs:"), "static")},
    "undeclared slot": {"greet/v1.md": _file(_FRONT.format(v=1), "{{name}} {{age}}")},
    "unused input": {"greet/v1.md": _file(_FRONT.format(v=1), "no slots")},
    "malformed slot": {"greet/v1.md": _file(_FRONT.format(v=1), "{{name}} {{ Name }}")},
    "no frontmatter": {"greet/v1.md": "just a body"},
    "schema missing": {"greet/v1.md": _file(_FRONT.format(v=1) + "response_schema: r.schema.json\n", "{{name}}")},
    "schema not json": {"greet/v1.md": _file(_FRONT.format(v=1) + "response_schema: r.schema.json\n", "{{name}}"),
                        "greet/r.schema.json": "{nope"},
    "schema not sibling": {"greet/v1.md": _file(_FRONT.format(v=1) + "response_schema: ../r.schema.json\n", "{{name}}")},
    "outside the subset": {"greet/v1.md": _file(_FRONT.format(v=1) + "model_notes: a: b\n", "{{name}}")},
}


@pytest.mark.parametrize("name", sorted(_BAD))
def test_load_rejects_invalid_files(tmp_path, name):
    for rel, text in _BAD[name].items():
        _write(tmp_path, rel, text)
    with pytest.raises(pl.InvalidPrompt):
        pl.PromptLibrary.load(tmp_path)


def test_block_list_and_crlf_and_inline_schema(tmp_path):
    front = ("id: greet\r\nversion: 1\r\ntier: volatile\r\ndescription: 'It''s fine'\r\n"
             "inputs:\r\n  - name\r\n  - age\r\n"
             'response_schema: {"type": "object", "required": ["a"]}\r\n')
    _write(tmp_path, "greet/v1.md", "---\r\n" + front + "---\r\n{{name}} is {{age}}\r\n")
    p = pl.PromptLibrary.load(tmp_path).get("greet")
    assert p.inputs == ("name", "age")
    assert p.description == "It's fine"
    assert p.body == "{{name}} is {{age}}"
    assert p.response_schema == b'{"required":["a"],"type":"object"}'


def test_packaged_library_loads():
    assert "reflection" in pl.default_library().ids()
    assert pl.get("reflection").version >= 1


def test_reflection_prompt_renders_through_the_library():
    got = build_reflection_prompt("a coder", "find big files", "execute_command",
                                  {"command": "find . -size +1M"}, "DEADLINE_EXCEEDED",
                                  ["earlier lesson"])
    p = pl.get("reflection")
    assert got == p.render({
        "role_sentence": "You are a coder.", "task": "find big files",
        "action": got.split("action: ", 1)[1].split("\n", 1)[0],
        "failure": "DEADLINE_EXCEEDED", "prior_reflections": "- earlier lesson",
    }).text


def test_reflection_prompt_golden():
    """The exact text sent. A change to prompts/reflection/ shows here as a diff."""
    got = build_reflection_prompt("a coder", "find big files", "execute_command",
                                  {"command": "find . -size +1M"}, "DEADLINE_EXCEEDED",
                                  ["earlier lesson about timeouts"])
    want = (_GOLDEN / "reflection_v1.txt").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert got == want
