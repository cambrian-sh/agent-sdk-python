"""ADR-0046 D5: agent-local skills in the ReAct loop.

Agent skills are SDK-local (the @tools analog): always present in the menu,
listed before system skills, shadowing same-name system skills, and loaded
locally by use_skill without a kernel Tier-2 fetch.
"""

import json

from cambrian_agent_sdk import AgentTask, CognitiveAgent


class _Bot(CognitiveAgent):
    role = "a skill-using assistant"

    def run(self, task):
        return self.think(task)


class _FakeSub:
    def __init__(self, responses, system_skills):
        self._responses = list(responses)
        self.prompts = []
        self.list_skills_calls = []
        self._system = system_skills

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def list_skills(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.list_skills_calls.append({"query": query, "names": names, "full": full})
        if names and full:
            return [s for s in self._system if s["name"] in names]
        # menu push: Tier-1 (name + description) for the system skills
        return [{"name": s["name"], "description": s["description"]} for s in self._system]


def test_agent_local_skill_always_in_menu_and_loads_locally():
    bot = _Bot(agent_id="b")
    bot.local_skills = [{
        "name": "deploy", "description": "Deploy locally.",
        "instructions": "step one: do X", "tool_grants": [],
    }]
    bot.substrate = _FakeSub([
        json.dumps({"action": "use_skill", "skill": "deploy"}),
        json.dumps({"action": "final_answer", "answer": "done"}),
    ], system_skills=[])

    res = bot.think(AgentTask(text="please deploy"))
    assert res.text == "done"
    # The local skill is listed in the menu even with no system match.
    assert "deploy" in bot.substrate.prompts[0]
    # It loads LOCALLY — no Tier-2 (full=true) kernel fetch was issued.
    assert not any(c["full"] for c in bot.substrate.list_skills_calls)
    # Its instructions were injected for the next turn.
    assert "step one: do X" in bot.substrate.prompts[-1]


def test_agent_local_skill_shadows_same_name_system_skill():
    bot = _Bot(agent_id="b")
    bot.local_skills = [{
        "name": "deploy", "description": "LOCAL deploy procedure.",
        "instructions": "local steps", "tool_grants": [],
    }]
    bot.substrate = _FakeSub(
        [json.dumps({"action": "final_answer", "answer": "ok"})],
        system_skills=[{"name": "deploy", "description": "SYSTEM deploy procedure."}],
    )

    bot.think(AgentTask(text="x"))
    menu = bot.substrate.prompts[0]
    assert "LOCAL deploy procedure." in menu
    assert "SYSTEM deploy procedure." not in menu  # shadowed by the agent's own
