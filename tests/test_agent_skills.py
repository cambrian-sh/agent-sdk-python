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


class _RankingSub(_FakeSub):
    """A kernel whose skill push is relevance-ranked: the top-1 skill depends on the
    query, as the kernel's SkillRetriever does; an empty query is the full menu."""

    def list_skills(self, query="", k=0, names=None, full=False, session_token_id=""):
        self.list_skills_calls.append({"query": query, "names": names, "full": full})
        menu = [{"name": s["name"], "description": s["description"]} for s in self._system]
        if not query:
            return menu
        return [s for s in menu if s["name"] in query][:1] or menu[:1]


def _system_tier(prompt: str) -> str:
    # The legacy single prompt = system text, then the volatile <Task> user turn.
    return prompt.split("<Task>")[0]


def test_full_tool_menu_agent_gets_the_full_skill_menu_stable_across_turns():
    """ENG-126: the skill menu is rendered into the cached system tier, so an agent on
    the full tool menu (seed_tools_full, the single-agent lane's chat agent) must not
    get a per-message top-k: two turns with different messages send the same tier."""
    skills = [{"name": "chart-report", "description": "Draw a chart."},
              {"name": "summarise", "description": "Summarise a text."}]
    tiers = []
    for text in ("summarise this mail", "make a chart-report of sales"):
        bot = _Bot(agent_id="b")
        bot.seed_tools_full = True
        bot.substrate = _RankingSub([json.dumps({"action": "final_answer", "answer": "ok"})],
                                    system_skills=skills)
        bot.think(AgentTask(text=text))
        assert bot.substrate.list_skills_calls[0]["query"] == "", "full menu: no relevance query"
        tiers.append(_system_tier(bot.substrate.prompts[0]))
    assert tiers[0] == tiers[1]
    assert "chart-report" in tiers[0] and "summarise" in tiers[0]


def test_task_relevant_skill_push_is_unchanged_without_the_full_menu():
    bot = _Bot(agent_id="b")
    bot.substrate = _RankingSub([json.dumps({"action": "final_answer", "answer": "ok"})],
                                system_skills=[{"name": "summarise", "description": "Summarise."}])
    bot.think(AgentTask(text="summarise this"))
    assert bot.substrate.list_skills_calls[0]["query"] == "summarise this"
