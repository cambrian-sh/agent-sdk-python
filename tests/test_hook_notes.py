"""Contract 0104 (ADR-0139 D11, ENG-629): the kernel's pre_tool / post_tool hook
annotations arrive beside the tool result and are shown to the model under
``_hook_notes``; the result itself is unchanged."""

from cambrian_agent_sdk import react


class _Substrate:
    def __init__(self, resp):
        self.resp = resp

    def execute_tool(self, *_a, **_kw):
        return self.resp


class _Agent:
    def __init__(self, resp):
        self.substrate = _Substrate(resp)


def test_notes_attach_to_a_dict_result():
    got = react._call_system_tool(_Agent({"result_json": '{"total": 5}', "annotations": ["post: unverified"]}), "t", {})
    assert got == {"total": 5, "_hook_notes": ["post: unverified"]}


def test_notes_wrap_a_non_dict_result():
    got = react._call_system_tool(_Agent({"result_json": "[1,2]", "annotations": ["pre: cached"]}), "t", {})
    assert got == {"result": [1, 2], "_hook_notes": ["pre: cached"]}


def test_no_notes_no_key():
    got = react._call_system_tool(_Agent({"result_json": '{"a": 1}'}), "t", {})
    assert got == {"a": 1}
    got = react._call_system_tool(_Agent({"result_json": '{"a": 1}', "annotations": []}), "t", {})
    assert got == {"a": 1}
