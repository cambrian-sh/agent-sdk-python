"""ADR-0036 issue 0036-04: the think() ReAct loop.

reason/retrieve/act with three branches (memory-query, tool-call, final-answer),
a capped memory budget (graceful degradation) and a capped tool budget (typed error
caught by the default run() ⇒ type='error'). Prompts are built via the PROMPTREQ
4-section builder with the @tool registry injected into the output schema.
"""

import json

import pytest

from cambrian_agent_sdk import AgentResult, AgentTask, CognitiveAgent, tool
from cambrian_agent_sdk.react import ReActLoopError


class _FakeLLM:
    """Scripted LLM: returns queued responses; records the prompts it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, session_token_id, prompt, **kw):
        self.prompts.append(prompt)
        self.last_kwargs = kw
        return self.responses.pop(0)


class Bot(CognitiveAgent):
    role = "a careful test assistant"

    def run(self, task):
        return self.think(task)


class _FakeMemory:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def recall(self, query, **kw):
        self.calls += 1
        return self.results


class ToolBot(CognitiveAgent):
    role = "a tool-using assistant"

    @tool
    def add(self, a: int, b: int) -> int:
        return a + b

    def run(self, task):
        return self.think(task)


def test_think_parses_and_returns_final_answer():
    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "hello"})])

    res = bot.think(AgentTask(text="hi"))
    assert isinstance(res, AgentResult)
    assert res.text == "hello"


def test_think_final_answer_non_string_is_coerced_not_crash():
    """Regression: an LLM that returns a structured (dict/list) "answer" must not
    crash the loop. Previously answer[:200] raised KeyError(slice) and
    AgentResult.from_text(dict) raised AttributeError."""
    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([
        json.dumps({"action": "final_answer", "answer": {"observations": "x", "conclusion": "y"}})
    ])

    res = bot.think(AgentTask(text="hi"))
    assert isinstance(res, AgentResult)
    # The dict answer is serialized to text, not dropped or crashed.
    assert "conclusion" in res.text and "y" in res.text


class _FakeSubstrateWithTools:
    """A fake substrate exposing both generate_stream (for the LLM) and
    execute_tool (for kernel system tools, ADR-0039)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.tool_calls = []

    def generate(self, *a, **kw):
        return self._responses.pop(0)

    def execute_tool(self, tool_name, args_json="", session_token_id="", step_index=0, timeout_ms=0, task_id=""):
        self.tool_calls.append((tool_name, json.loads(args_json or "{}")))
        return {"result_json": json.dumps({"content": "the file says hello"}),
                "result_cid": "", "denied": False, "deny_reason": "", "error": "",
                "arg_hash": "h", "result_hash": "h"}


def test_think_routes_system_tool_to_execute_tool():
    """A tool_call for a name that is NOT a local @tool is routed to the kernel's
    ExecuteTool (ADR-0039), and its result feeds the loop."""
    bot = Bot(agent_id="b")  # Bot has no @tool
    bot.substrate = _FakeSubstrateWithTools([
        json.dumps({"action": "tool_call", "tool": "read_file", "args": {"path": "/data/x.txt"}}),
        json.dumps({"action": "final_answer", "answer": "done"}),
    ])

    res = bot.think(AgentTask(text="read the file"))
    assert res.text == "done"
    assert bot.substrate.tool_calls == [("read_file", {"path": "/data/x.txt"})]


def test_think_system_tool_missing_substrate_degrades():
    """No substrate ⇒ a system tool call returns a structured error, not a crash."""
    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([
        json.dumps({"action": "tool_call", "tool": "read_file", "args": {}}),
        json.dumps({"action": "final_answer", "answer": "ok"}),
    ])
    # _FakeLLM has no execute_tool → routing degrades gracefully.
    res = bot.think(AgentTask(text="x"))
    assert res.text == "ok"


def test_think_tool_call_branch_then_final():
    bot = ToolBot(agent_id="tb")
    bot.substrate = _FakeLLM([
        json.dumps({"action": "tool_call", "tool": "add", "args": {"a": 2, "b": 3}}),
        json.dumps({"action": "final_answer", "answer": "the sum is 5"}),
    ])

    res = bot.think(AgentTask(text="add 2 and 3"))
    assert res.text == "the sum is 5"
    # the tool result was fed back into the loop's scratchpad (2nd prompt sees it)
    assert "5" in bot.substrate.prompts[1]


def test_tool_card_carries_provenance_in_prompt():
    """ADR-0041 D1 + v2 trajectory: a tool call is recorded as an invocation card
    with tool + args (as named children in a <call> block) + status + result.
    The next prompt shows what was run, how it was called, and the outcome."""
    bot = ToolBot(agent_id="tb")
    bot.substrate = _FakeLLM([
        json.dumps({"action": "tool_call", "tool": "add", "args": {"a": 2, "b": 3}}),
        json.dumps({"action": "final_answer", "answer": "5"}),
    ])
    bot.think(AgentTask(text="add 2 and 3"))
    p = bot.substrate.prompts[1]
    # v2: tool name in the typed Step attribute, args as named <call> children.
    assert "'add'" in p or "tool='add'" in p      # tool name
    assert "<a>2</a>" in p and "<b>3</b>" in p     # args — provenance (what it called with)
    assert "status=" in p                         # outcome status
    assert "5" in p                               # result still present (behavior preserved)


class _FakeSubstrateToolError:
    """A fake substrate whose system tool always fails (for provenance-on-failure)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def execute_tool(self, tool_name, args_json="", **kw):
        return {"result_json": "", "result_cid": "", "denied": False, "deny_reason": "",
                "error": "DEADLINE_EXCEEDED", "arg_hash": "", "result_hash": ""}


def test_failed_tool_call_records_status_and_args():
    """A failed tool call is recorded with status='error' AND its args — exactly the
    provenance the agent lacked when it re-issued near-identical failing commands."""
    bot = Bot(agent_id="b")
    bot.substrate = _FakeSubstrateToolError([
        json.dumps({"action": "tool_call", "tool": "execute_command",
                    "args": {"command": "find . -size +1M"}}),
        json.dumps({"action": "final_answer", "answer": "done"}),
    ])
    bot.think(AgentTask(text="find big files"))
    p = bot.substrate.prompts[1]
    assert "status='error'" in p
    assert "find . -size +1M" in p              # the args of the FAILED call are visible


class _FakeSubstrateOffload:
    """A system tool whose result the kernel OFFLOADED (result_cid set, payload
    cleared), and which serves the full content back via get_context_node."""

    def __init__(self, responses, cid, content):
        self._responses = list(responses)
        self.prompts = []
        self._cid = cid
        self._content = content

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def execute_tool(self, tool_name, args_json="", **kw):
        return {"result_json": "", "result_cid": self._cid, "denied": False,
                "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}

    def get_context_node(self, cid):
        if cid != self._cid:
            return None

        class _N:
            data = self._content.encode("utf-8")

        return _N()


def test_heavy_system_tool_result_offloaded_as_cid_not_payload():
    """ADR-0041 D3: a heavy system-tool result is represented in-prompt by a
    {marker + cid}, not the full payload; the full content drills down by cid."""
    from cambrian_agent_sdk.working_memory import ToolCard, resolve_content

    big = "BIG_FILE_CONTENT " * 500
    bot = Bot(agent_id="b")
    bot.substrate = _FakeSubstrateOffload(
        [
            json.dumps({"action": "tool_call", "tool": "read_file", "args": {"path": "/big"}}),
            json.dumps({"action": "final_answer", "answer": "done"}),
        ],
        cid="Qm42",
        content=big,
    )
    bot.think(AgentTask(text="read the big file"))

    p = bot.substrate.prompts[1]
    assert "Qm42" in p                       # cid present
    assert "offloaded" in p.lower()          # marker, not the payload
    assert "BIG_FILE_CONTENT" not in p       # the full payload is NOT inlined

    # Drill-down retrieves the full content the card summarized.
    card = ToolCard.from_result("read_file", {"path": "/big"}, {"cid": "Qm42"})
    assert "BIG_FILE_CONTENT" in resolve_content(card, bot.substrate)


class _CountingFailSubstrate:
    """A system tool that ALWAYS fails; counts how many times it actually ran."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.tool_call_count = 0

    def generate(self, session_token_id=None, prompt="", **kw):
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def execute_tool(self, tool_name, args_json="", **kw):
        self.tool_call_count += 1
        return {"result_json": "", "result_cid": "", "denied": False, "deny_reason": "",
                "error": "DEADLINE_EXCEEDED", "arg_hash": "", "result_hash": ""}


class _CountingOkSubstrate(_CountingFailSubstrate):
    def execute_tool(self, tool_name, args_json="", **kw):
        self.tool_call_count += 1
        return {"result_json": '{"ok": true}', "result_cid": "", "denied": False,
                "deny_reason": "", "error": "", "arg_hash": "", "result_hash": ""}


_FAIL_CMD = json.dumps({"action": "tool_call", "tool": "execute_command",
                        "args": {"command": "find . -size +1M"}})


def test_recurrence_gate_soft_nudge_then_veto_then_escalate():
    """ADR-0041 D4: a stubbornly re-issued FAILING action is allowed exactly one
    retry (soft nudge), then hard-vetoed and escalated to an honest failure —
    never a silent retry-storm."""
    from cambrian_agent_sdk.react import run_think

    bot = Bot(agent_id="b")
    bot.substrate = _CountingFailSubstrate([_FAIL_CMD] * 10)
    res = run_think(bot, AgentTask(text="find big files"))

    # Ran twice: round 1 (novel) + round 2 (the one allowed retry); then vetoed.
    assert bot.substrate.tool_call_count == 2
    assert res.type == "error"          # escalated to honest failure, not a spin


def test_recurrence_gate_no_false_positive_on_distinct_successes():
    """DISTINCT successful calls (a real multi-step plan) both run — the success
    dedup guard is content-keyed and must never veto different actions."""
    from cambrian_agent_sdk.react import run_think

    call_a = json.dumps({"action": "tool_call", "tool": "read_file", "args": {"path": "/a"}})
    call_b = json.dumps({"action": "tool_call", "tool": "read_file", "args": {"path": "/b"}})
    final = json.dumps({"action": "final_answer", "answer": "done"})
    bot = Bot(agent_id="b")
    bot.substrate = _CountingOkSubstrate([call_a, call_b, final])
    res = run_think(bot, AgentTask(text="read two files"))

    assert res.text == "done"
    assert bot.substrate.tool_call_count == 2  # both DISTINCT successes ran


def test_success_dedup_skips_repeated_identical_call():
    """The loop bug (qwen3 wrote hello.txt 5×): the SAME succeeding call re-issued is
    executed ONCE; the identical repeat is deduped (not re-run) and steered to finish."""
    from cambrian_agent_sdk.react import run_think

    same = json.dumps({"action": "tool_call", "tool": "read_file", "args": {"path": "/x"}})
    final = json.dumps({"action": "final_answer", "answer": "done"})
    bot = Bot(agent_id="b")
    bot.substrate = _CountingOkSubstrate([same, same, final])
    res = run_think(bot, AgentTask(text="read x; idempotent repeat"))

    assert res.text == "done"
    assert bot.substrate.tool_call_count == 1  # the identical repeat was deduped, not re-run


def test_success_dedup_completes_gracefully_when_model_wont_advance():
    """If the model keeps re-proposing an already-succeeded call past the veto depth,
    the loop finishes gracefully (a success, not an error or an infinite spin — the
    dedup path skips the tool-round cap, so the ladder must terminate it)."""
    from cambrian_agent_sdk.react import run_think

    same = json.dumps({"action": "tool_call", "tool": "read_file", "args": {"path": "/x"}})
    bot = Bot(agent_id="b")
    bot.substrate = _CountingOkSubstrate([same] * 10)  # never emits final_answer
    res = run_think(bot, AgentTask(text="spin on a success"))

    assert res.type != "error"                 # graceful completion — the work succeeded
    assert bot.substrate.tool_call_count == 1  # executed once; repeats deduped → finish


def test_recurrence_gate_can_be_disabled():
    """recurrence_enabled=False ⇒ no veto; the failing call runs until the tool cap."""
    from cambrian_agent_sdk.react import ReActLoopError, run_think

    bot = Bot(agent_id="b")
    bot.substrate = _CountingFailSubstrate([_FAIL_CMD] * 10)
    with pytest.raises(ReActLoopError):
        run_think(bot, AgentTask(text="x"), max_tool_rounds=3, recurrence_enabled=False)
    assert bot.substrate.tool_call_count == 3  # ran every time, only the tool budget stopped it


def test_token_bound_regression_caps_prompt_below_round_count():
    """ADR-0041 D2 / Falsification: across many rounds the assembled prompt stays
    bounded (≤ cap cards) — strictly fewer than the round count, which the old flat
    loop replayed in full (the O(N²) growth LRW removes)."""
    from cambrian_agent_sdk.react import run_think
    from cambrian_agent_sdk.working_memory import DEFAULT_CAP

    n_rounds = 15
    calls = [json.dumps({"action": "tool_call", "tool": "add", "args": {"a": i, "b": 1}})
             for i in range(n_rounds)]
    calls.append(json.dumps({"action": "final_answer", "answer": "done"}))
    bot = ToolBot(agent_id="tb")
    bot.substrate = _FakeLLM(calls)
    run_think(bot, AgentTask(text="many steps"), max_tool_rounds=n_rounds + 5)

    cards_in_prompt = bot.substrate.prompts[-1].count("<tool")
    assert cards_in_prompt <= DEFAULT_CAP   # bounded
    assert cards_in_prompt < n_rounds       # strictly below what the flat loop would replay


def test_yield_subgoal_action_returns_yield_result():
    """ADR-0041 D5: the loop turns a yield_subgoal action into a yielded SubGoal
    (delegated to the kernel), not in-agent execution."""
    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([
        json.dumps({"action": "yield_subgoal", "intent": "fetch the EUR/USD rate",
                    "capability_hint": "currency"}),
    ])
    res = bot.think(AgentTask(text="convert 100 EUR to USD"))
    assert res.is_yield
    assert res.subgoal.intent == "fetch the EUR/USD rate"
    assert res.subgoal.capability_hint == "currency"


def test_resumed_yield_seeds_delegated_result_card():
    """ADR-0037 D10 resume (delegate-and-continue): a re-dispatched parent receives
    the delegated sub-result in context; run_think seeds it into working memory so
    the Executive uses it and answers, instead of re-yielding."""
    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "the rate is 1.09"})])
    task = AgentTask(
        text="convert 100 EUR to USD",
        context={"_yield_result": "EUR/USD = 1.09", "_yield_resumed_intent": "fetch the EUR/USD rate"},
    )
    res = bot.think(task)
    assert res.text == "the rate is 1.09"
    assert "EUR/USD = 1.09" in bot.substrate.prompts[0]   # the delegated result reached the prompt
    assert "delegated" in bot.substrate.prompts[0]


def test_yield_subgoal_empty_intent_degrades_not_crash():
    """A malformed yield (no intent) degrades to a note + re-prompt — never crashes
    on yield_subgoal's ValueError."""
    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([
        json.dumps({"action": "yield_subgoal", "intent": ""}),
        json.dumps({"action": "final_answer", "answer": "did it inline"}),
    ])
    res = bot.think(AgentTask(text="x"))
    assert not res.is_yield
    assert res.text == "did it inline"


def test_trivial_reasoning_resolves_without_tool_calls():
    """ADR-0041 D5: simple reasoning is done inline (one final_answer), not
    decomposed into a tool_call per step."""
    bot = ToolBot(agent_id="tb")  # an 'add' @tool exists but should not be needed
    bot.substrate = _FakeLLM([
        json.dumps({"action": "final_answer", "answer": "area diff = 3.14*(R1^2 - R2^2)"}),
    ])
    res = bot.think(AgentTask(text="difference of two circle areas"))
    assert "3.14" in res.text
    assert len(bot.substrate.prompts) == 1  # resolved in a single pass, no tool round


def test_menu_offers_yield_and_inline_reasoning_guidance():
    from cambrian_agent_sdk.react import build_output_schema

    schema = build_output_schema(ToolBot(agent_id="tb"))
    assert "yield_subgoal" in schema
    assert "trivial" in schema and "final_answer" in schema  # inline-reasoning steer


def test_no_in_agent_scheduler_introduced():
    """ADR-0041 D5: batching is delegated to the kernel — the agent loop must NOT
    grow an in-process scheduler / parallel executor."""
    import inspect

    import cambrian_agent_sdk.react as r

    src = inspect.getsource(r)
    assert "ThreadPool" not in src
    assert "concurrent.futures" not in src
    assert "asyncio" not in src


def test_think_memory_query_branch_then_final():
    bot = Bot(agent_id="b")
    bot.memory = _FakeMemory([{"text": "paris is the capital"}])
    bot.substrate = _FakeLLM([
        json.dumps({"action": "memory_query", "query": "capital of france"}),
        json.dumps({"action": "final_answer", "answer": "Paris"}),
    ])

    res = bot.think(AgentTask(text="capital of france?"))
    assert res.text == "Paris"
    # 2 recalls now: the mandatory seed recall + the LLM's explicit memory_query.
    assert bot.memory.calls == 2
    assert "paris is the capital" in bot.substrate.prompts[1]


def test_seed_recall_always_fires_even_without_llm_memory_query():
    """Every cognitive run consults LTM at least once — the retrieval loop is on."""
    bot = Bot(agent_id="b")
    bot.memory = _FakeMemory([{"text": "a remembered fact"}])
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "done"})])

    bot.think(AgentTask(text="anything"))
    assert bot.memory.calls == 1  # the mandatory seed recall fired
    assert "a remembered fact" in bot.substrate.prompts[0]  # and reached the prompt


def test_result_type_override_wins_over_llm_declared_type():
    class CodeBot(CognitiveAgent):
        role = "a coder"
        result_type = "code"  # output contract: always route as code

        def run(self, task):
            return self.think(task)

    bot = CodeBot(agent_id="cb")
    # LLM declares type=text, but the agent's contract forces code.
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "print(1)", "type": "text"})])
    res = bot.think(AgentTask(text="write hello"))
    assert res.type == "code"


def test_given_working_memory_is_seeded_into_the_prompt():
    from cambrian_agent_sdk.types import ContextRef

    bot = Bot(agent_id="b")
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "ok"})])
    ref = ContextRef(cid="c1", type="fact", activation=1.0, precision=0.9, snippet="GIVEN_WORKSPACE_FACT")
    bot.think(AgentTask(text="q", working_memory=[ref]))
    assert "GIVEN_WORKSPACE_FACT" in bot.substrate.prompts[0]


def test_generation_params_are_threaded_from_agent_config_to_generate():
    """An agent's max_tokens / temperature reach substrate.generate() inside think()."""

    class Tuned(CognitiveAgent):
        role = "a tuned bot"
        max_tokens = 256
        temperature = 0.1

        def run(self, task):
            return self.think(task)

    bot = Tuned(agent_id="t")
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "ok"})])
    bot.think(AgentTask(text="hi"))
    assert bot.substrate.last_kwargs["max_tokens"] == 256
    assert bot.substrate.last_kwargs["temperature"] == 0.1


def test_default_generation_params():
    bot = Bot(agent_id="b")  # no overrides → CognitiveAgent defaults
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "ok"})])
    bot.think(AgentTask(text="hi"))
    assert bot.substrate.last_kwargs["max_tokens"] == 1024
    assert bot.substrate.last_kwargs["temperature"] == 0.7


def test_seed_recall_can_be_disabled():
    class NoSeed(CognitiveAgent):
        role = "x"
        seed_recall = False

        def run(self, task):
            return self.think(task)

    bot = NoSeed(agent_id="ns")
    bot.memory = _FakeMemory([{"text": "f"}])
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "ok"})])
    bot.think(AgentTask(text="q"))
    assert bot.memory.calls == 0  # opt-out respected


def test_memory_queries_capped_with_graceful_degradation():
    """An LLM that only ever asks for memory must terminate best-effort, not crash."""
    bot = Bot(agent_id="b")
    bot.memory = _FakeMemory([{"text": "fact"}])
    # always memory_query — would loop forever without the cap + hard stop
    bot.substrate = _FakeLLM([json.dumps({"action": "memory_query", "query": "x"})] * 20)

    res = bot.think(AgentTask(text="q"), max_memory_queries=3)
    assert isinstance(res, AgentResult)  # graceful, no crash
    assert bot.memory.calls == 3  # cap enforced — never more than the budget


# DISTINCT calls (varying args) so the success-dedup guard never intercepts them:
# the tool-round cap is the backstop for a runaway plan of *different* actions,
# whereas identical successful repeats are stopped earlier by the dedup ladder.
def _distinct_tool_calls(n):
    return [json.dumps({"action": "tool_call", "tool": "add", "args": {"a": i, "b": 1}})
            for i in range(n)]


def test_tool_rounds_capped_raises_typed_error():
    bot = ToolBot(agent_id="tb")
    bot.substrate = _FakeLLM(_distinct_tool_calls(20))  # never stops calling tools
    with pytest.raises(ReActLoopError):
        bot.think(AgentTask(text="loop"), max_tool_rounds=2)


def test_default_run_converts_tool_loop_to_error_result():
    """The DEFAULT run() catches the typed loop error ⇒ a type='error' result, no crash."""

    class DefaultRunBot(CognitiveAgent):
        role = "a default-run bot"

        @tool
        def add(self, a: int, b: int) -> int:
            return a + b

        # NB: no run() override — relies on CognitiveAgent's default run().

    bot = DefaultRunBot(agent_id="dr")
    bot.substrate = _FakeLLM(_distinct_tool_calls(50))  # exceeds the default tool-round cap
    result = bot.run(AgentTask(text="loop"))
    assert result.type == "error"
    assert result.confidence == 0.0


def test_prompt_uses_four_section_builder_with_tools_in_output_schema():
    bot = ToolBot(agent_id="tb")
    bot.substrate = _FakeLLM([json.dumps({"action": "final_answer", "answer": "done"})])
    bot.think(AgentTask(text="hi"))

    prompt = bot.substrate.prompts[0]
    # PROMPTREQ 4-section structure
    assert "<System>" in prompt and "<Role>" in prompt
    assert "<Task>" in prompt
    assert "<OutputSchema>" in prompt
    # the @tool registry is injected into the output schema (closed menu)
    assert "add" in prompt
