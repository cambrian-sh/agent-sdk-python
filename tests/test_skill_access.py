"""Per-agent skill access in the ReAct loop (decisions/skill-access-grants.md).

A profile agent's skill set is its profile's skills plus the skills an operator
granted it. find_skills may discover skills outside the set (marked requestable);
use_skill on one makes the kernel ask an operator, and the answer is granted (the
body loads in the same run), pending or denied (no body — the loop says why, once).
"""

import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent, react


class _Bot(CognitiveAgent):
    role = "a skill-using assistant"

    def run(self, task):
        return self.think(task)


class _Kernel:
    """A bounding kernel: "report" is in the set; "proposed-x" is outside it."""

    def __init__(self, responses, answer):
        self._responses = list(responses)
        self.answer = answer  # what a full load of proposed-x answers
        self.prompts = []
        self.calls = []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def list_skills(self, query="", k=0, names=None, full=False, session_token_id="",
                    rationale="", discover=False):
        self.calls.append({"query": query, "names": names, "full": full,
                           "rationale": rationale, "discover": discover})
        if names:
            out = []
            for n in names:
                if n == "report":
                    out.append({"name": n, "description": "Report.", "access": "profile",
                                "instructions": "REPORT BODY" if full else "", "tool_grants": []})
                elif n == "proposed-x":
                    acc = self.answer if full else "requestable"
                    out.append({"name": n, "description": "Proposed.", "access": acc,
                                "access_detail": "approved by op" if acc == "granted" else "nobody answered",
                                "instructions": "PROPOSED BODY" if acc == "granted" else "",
                                "tool_grants": []})
            return out
        if query and discover:
            return [{"name": "proposed-x", "description": "Proposed.", "access": "requestable"}]
        return [{"name": "report", "description": "Report.", "access": "profile"}]


def _run(answer, actions):
    bot = _Bot(agent_id="analyst")
    bot.substrate = _Kernel([json.dumps(a) for a in actions], answer)
    res = bot.think(AgentTask(text="write the quarterly report"))
    return bot.substrate, res


def test_find_skills_lists_requestable_skills():
    k, res = _run("pending", [
        {"action": "find_skills", "need": "a proposed procedure"},
        {"action": "final_answer", "answer": "done"},
    ])
    assert res.text == "done"
    assert any(c["discover"] for c in k.calls)
    assert "outside your skill set" in k.prompts[1] and "proposed-x" in k.prompts[1]


def test_use_skill_pending_says_so_once_and_loads_nothing():
    k, res = _run("pending", [
        {"action": "use_skill", "skill": "proposed-x"},
        {"action": "use_skill", "skill": "proposed-x"},
        {"action": "final_answer", "answer": "done"},
    ])
    assert res.text == "done"
    loads = [c for c in k.calls if c["full"]]
    assert len(loads) == 1, "a refused skill is requested once per run"
    assert loads[0]["rationale"].startswith("needed for task: write the quarterly report")
    assert "an operator was asked to grant it" in k.prompts[1]
    assert "nobody answered" in k.prompts[1]
    assert "PROPOSED BODY" not in k.prompts[-1]
    assert "an operator was asked to grant it" in k.prompts[2]


def test_use_skill_denied_note():
    k, _ = _run("denied", [
        {"action": "use_skill", "skill": "proposed-x"},
        {"action": "final_answer", "answer": "done"},
    ])
    assert "is not available to you: nobody answered" in k.prompts[1]
    assert "NOT loaded" in k.prompts[1]


def test_use_skill_granted_live_loads_the_body_in_the_same_run():
    k, res = _run("granted", [
        {"action": "use_skill", "skill": "proposed-x"},
        {"action": "final_answer", "answer": "done"},
    ])
    assert res.text == "done"
    assert "PROPOSED BODY" in k.prompts[-1]


def test_nudges_v3_keep_v2_sections_byte_identical():
    from cambrian_agent_sdk import prompt_library as pl
    v2, v3 = pl.sections(pl.get_version("react-nudges", 2)), pl.sections(pl.get_version("react-nudges", 3))
    for name, frag in v2.items():
        assert v3[name].body == frag.body, name
    assert {"skill-pending", "skill-denied", "skills-requestable"} <= set(v3)
    assert react._nudge("skill-denied", skill="'x'", detail="no").startswith("<note>Skill 'x' is not available")
