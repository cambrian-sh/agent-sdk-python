"""The think() ReAct loop (ADR-0036 issue 0036-04).

A reason/retrieve/act loop with three branches:

- **memory-query** — capped re-queries with *graceful degradation*: once the budget
  is spent the loop stops retrieving and asks the LLM to answer best-effort; it never
  crashes at the cap;
- **tool-call** — capped rounds; exceeding the cap raises :class:`ReActLoopError`,
  a *typed* error the default ``run()`` catches and turns into ``type="error"``;
- **final-answer** — parsed and returned as an :class:`AgentResult`.

The prompt is assembled by :func:`helpers.build_prompt_parts` as a cached, stable
system tier (role, constraints, anti-patterns, the action protocol with the closed
tool and skill menu, the per-turn contract) plus a volatile user turn (task,
trajectory, per-turn notes), and sent on the ADR-0139 C1 fields. The texts are the
SDK prompt-library files ``prompts/react-*/v1.md``.

v2 additions (see docs/research/agent-prompting/SUMMARY.md):
- ActionProtocol is grouped (Memory / Tools / Skills / Answer / Delegation) with
  one worked example per action.
- ``_render_recall(query, results)`` now includes the query as an XML attribute
  on the ``<step>``, so the agent can see WHAT was asked, not just what was
  returned.
- The 4 Rules also live in ``<Constraints>`` at the top of <System> (via
  ``helpers.DEFAULT_CONSTRAINTS``) for recency-priority visibility.
- Trajectory entries are emitted via :func:`working_memory.add_step` for typed
  steps (memory_query, tool_call) — the v2 rich XML renderer produces
  self-describing ``<step n="N" type="..." status="...">`` blocks with named
  ``<call>`` children and ``<result offloaded_cid="...">`` summaries.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# build_prompt stays importable from here for callers that used react.build_prompt;
# the loop itself sends build_prompt_parts (system + messages, ADR-0139 D2).
from .helpers import (_esc, build_prompt, build_prompt_parts, default_anti_patterns,  # noqa: F401
                      default_constraints, prompt_versions)
from .prompt_library import fragment
from .host import host_block
from .recurrence import count_failed_duplicates, count_successful_duplicates
from .reflection import DEFAULT_MAX_REFLECTION_TOKENS, build_reflection_prompt
from .errors import ToolCallingUnsupported
from .types import AgentResult, AgentTask, yield_subgoal
from .working_memory import (
    Step,
    ToolCard,
    WorkingMemory,
    action_text,
    memory_provenance_attrs,
    render_entry_xml,
    _render_memory_results,
)

logger = logging.getLogger("cambrian.react")


class ReActLoopError(Exception):
    """Raised when the tool-round budget is exhausted (runaway loop guard)."""


_DEFAULT_MAX_MEMORY_QUERIES = 3
# Raised 5 -> 12 (2026-07-28). 5 was below the floor for ordinary multi-file work:
# create a directory, write a source file, read it back, create an output directory,
# write a derived file is already five calls, so a single step doing routine file work
# died on the guard rather than on anything runaway. Measured on the repeat-task suite,
# it was the difference between a task completing and a ReActLoopError.
#
# 12 is chosen to clear that floor with headroom while still bounding a genuine loop.
# The budget is NOT the primary runaway defence and should not be tuned as if it were:
# content-keyed success-dedup (recurrence_veto_depth) already catches the common spin,
# where a model re-proposes an action that already succeeded. This cap is the backstop
# for the case those miss.
_DEFAULT_MAX_TOOL_ROUNDS = 12
# ADR-0045: cap describe_tool fetches to avoid a describe loop.
_DEFAULT_MAX_DESCRIBE_CALLS = 5
# ADR-0046: cap use_skill loads per run.
_DEFAULT_MAX_SKILL_LOADS = 5
# ADR-0139 D3: the total budget for loaded skill bodies in one run, in characters
# (~4 chars/token, so ~8k tokens). A loaded skill is pinned in full for the rest of
# the run, so the budget is what bounds the instructions region; a load that would
# overflow it is REFUSED with a note, never truncated: a cut procedure is a wrong
# procedure. Characters rather than tokens because the SDK has no tokenizer; the
# Phase 2 context engine replaces this with a token budget.
_DEFAULT_MAX_SKILL_CHARS = 32000
# The sectioned-write tool the truncation guidance may recommend, when (and only
# when) it is actually in the agent's menu.
_LARGE_WRITE_TOOL = "fast_large_write_file"
# ADR-0044: default size of the task-relevant tool menu the kernel serves (the
# push). The kernel ranks the granted tools against the task and returns this many;
# the find_tools pull fetches more on demand, so this can stay small.
# The ADR-0044 seed size: how many task-relevant system tools are pushed into the menu
# without the agent asking. 3 is a small guess, and a wrong guess is expensive —
# measured 2026-07-28, a create-a-file suite was seeded `edit_file, read_file,
# fast_edit_blocks` with no write_file, and the model spent its budget editing files that
# did not exist. Overridable so the size can be measured as an arm rather than argued.
_DEFAULT_TOOL_MENU_K = int(os.getenv("CAMBRIAN_TOOL_MENU_K", "3") or 3)
# #2: how many consecutive token-limit truncations the loop re-prompts (steering
# toward chunked authoring) before giving up gracefully rather than spinning.
_MAX_TRUNCATION_RETRIES = 3
# How many times a non-action response is re-prompted before it is accepted as the
# answer. 1 = nudge once. See INFERRED_FINAL_ANSWER_NOTE for why this exists at all.
_MAX_INFERRED_ANSWER_RETRIES = 1
# An action kind no branch handles (a hallucinated verb, or one removed from the
# menu, e.g. recall_precedents on 2026-11-05) used to fall through to the
# final_answer tail and return an EMPTY answer as success. It is re-prompted this
# many times, then the run fails loudly (T-20261105-react-text-loop-ends-the-run-with-an-empty).
_MAX_UNKNOWN_ACTION_RETRIES = 2
# ADR-0052: how many verbal reflections the loop will extract per run. Each costs one
# small LLM call on a hard-veto; bounded so a thrashing run cannot reflect endlessly.
_DEFAULT_MAX_REFLECTIONS = 3
# resolve_cid (ADR-0048 residual): how many INLINE resolves per run. Inline
# resolves put the full body in the prompt, so they cost N tokens. Offload
# resolves are not capped (they just create a new cid; no token cost). The cap
# prevents a runaway model from filling the context with full bodies.
_DEFAULT_MAX_INLINE_RESOLVES = 3


# The ReAct prompt texts live in the SDK prompt library (ADR-0139 C4):
# prompts/react-action-protocol/v1.md, react-turn/v1.md, react-nudges/v1.md. This
# module chooses and joins sections; it holds no model-facing prose of its own.
_AP = "react-action-protocol"


def _ap(section: str, **values) -> str:
    return fragment(_AP, section).render(values)


# The Tool-actions half of the menu, in its two encodings (kept as module names for
# callers that import them).
#
# Under native tool-calling the JSON tool_call action must NOT be advertised. Telling a
# model to emit `{"action":"tool_call",...}` while ALSO attaching real tool schemas
# gives it two ways to do one thing, and it takes the one that does nothing. Measured
# 2026-07-28: with both present the suite scored 3/6 against 5/6 on the pure text path,
# and every failure was a model describing the work instead of performing it.
# ADR-0139 Phase 0: the native section does not even NAME the JSON action it forbids;
# the native prompt speaks one protocol, and naming the other is how it leaked back in.
_JSON_TOOL_SECTION = _ap("tools-text") + "\n\n"
_NATIVE_TOOL_SECTION = _ap("tools-native") + "\n\n"


def build_output_schema(
    agent, system_tools: Optional[List[Dict]] = None,
    system_skills: Optional[List[Dict]] = None,
    allow_yield_subgoal: bool = True,
    native_tools: bool = False,
    allow_memory: bool = True,
) -> str:
    """Describe the closed action menu + the agent's tools for the LLM (v2).

    The menu lists BOTH the agent's intra-process ``@tool`` registry and the
    kernel-owned **system tools** the agent is granted (ADR-0039), fetched from the
    kernel via ``substrate.list_tools()``. Without the system-tool half the
    model would only learn about ``execute_python`` / ``execute_command`` from
    prose in its Role, and hallucinate names off the menu — so we surface them
    explicitly here as a closed list it can select from.

    v2 additions (see docs/research/agent-prompting/SUMMARY.md):
    - Actions are grouped (Memory / Tools / Skills / Answer / Delegation) for
      visual scanability.
    - Each action has a 1-line JSON worked example (ReAct 2022's primary
      mechanism; the literature shows this is the single biggest lift).
    - The 4 Rules are kept here (prompt-mid position) as well as in
      ``<Constraints>`` at the top (recency-priority).

    The text is the ``react-action-protocol`` fragment set; blocks are joined by one
    blank line.
    """
    tool_lines = []
    for spec in agent.tools.specs():
        tool_lines.append(_ap("tool-line-local", name=spec.name,
                              schema=json.dumps(spec.schema["properties"])))
    for t in system_tools or []:
        desc = (t.get("description") or "").strip()
        props = _schema_properties(t.get("schema_json"))
        suffix = _ap("tool-line-schema", schema=json.dumps(props)) if props else ""
        tool_lines.append(_ap("tool-line-system", name=t.get("name"), description=desc,
                              suffix=suffix))
    tools_block = "\n".join(tool_lines) if tool_lines else _ap("tools-empty")
    # ADR-0046: the [skills] menu section — authored procedures the agent may load.
    skill_lines = [_ap("skill-line", name=sk.get("name"),
                       description=(sk.get("description") or "").strip())
                   for sk in system_skills or []]
    skills_block = "\n".join(skill_lines) if skill_lines else _ap("skills-empty")
    # ADR-0139 Phase 0: one protocol per prompt. Under native tool-calling there is no
    # JSON tool_call action, so neither the opener nor the rules may name one; tools
    # are CALLED. Non-tool actions (memory_query, final_answer, ...) stay JSON actions
    # until Phase 2 makes every control verb a native tool.
    variant = "native" if native_tools else "text"
    # Skills are native tools when tools are native and there is a skill menu to load
    # from (build_tool_definitions(include_skills=...) follows the same condition).
    native_skills = bool(native_tools and system_skills)
    # yield_subgoal is only meaningful inside a coordinated plan; an opted-out agent
    # (e.g. the chat front desk) delegates whole work via a tool instead, so we omit
    # the action from its menu entirely rather than advertising a dead option.
    answer = _ap("answer")
    if allow_yield_subgoal:
        answer += "\n" + _ap("answer-yield")
    blocks = [
        _ap(f"opener-{variant}"),
        # Memory is advertised only when the agent may actually use it.
        #
        # Listing it unconditionally while `max_memory_queries` is 0 is the
        # same inconsistent middle that broke the tool path: the menu says "Do
        # this FIRST", the model obeys, the budget refuses it, the model obeys
        # the menu again, and the SECOND refusal returns working memory as the
        # final answer — so the caller receives "<note>memory query budget
        # exhausted</note>" as though it were the agent's considered reply.
        #
        # Advertise a capability and allow it, or advertise neither.
        _ap("memory") if allow_memory else _ap("memory-none"),
        _ap(f"tools-{variant}"),
        _ap("skills-native") if native_skills else _ap("skills-text"),
        answer,
        # ADR-0048 residual.
        _ap(f"resolution-{variant}"),
        _ap(f"rules-{variant}"),
        _ap("toollist-native") if native_tools else _ap("toollist-text", tools=tools_block),
        (_ap("skillmenu-native", skills=skills_block) if native_skills
         else _ap("skillmenu-text", skills=skills_block)),
    ]
    return "\n\n".join(blocks)


def _looks_like_truncated_action(text: str) -> bool:
    """True when the output BEGAN a JSON action envelope but didn't parse (#2).

    A generation cut off at the token limit mid-action leaves an unterminated JSON
    object — an ``"action"`` key right after the opening ``{`` (tolerating a code
    fence / language tag before it) but no balanced close. That is a TRUNCATION, not a
    prose answer; distinguishing the two stops the loop from silently finalizing a
    broken action (and never running the tool_call the agent intended)."""
    brace = text.find("{")
    if brace == -1:
        return False
    # The "action" key sits at the head of the object — an envelope, not prose that
    # merely happens to contain braces somewhere.
    return '"action"' in text[brace : brace + 40]


# Sent back when the model produces something that is not an action. Names both
# legitimate continuations so the model does not read it as "you must call a tool".
def _nudge(section: str, **values) -> str:
    """A mid-run loop note (prompts/react-nudges/v1.md). Notes are volatile: they go
    into the message stream, never into the system blocks (ADR-0139 D2)."""
    return fragment("react-nudges", section).render(values)


INFERRED_FINAL_ANSWER_NOTE = _nudge("inferred-answer")


# OpenAI (and every gateway that mirrors it) constrains a function name to
# ^[A-Za-z0-9_-]{1,64}$. Cambrian's kernel-owned tools are named
# "mcp:filesystem/write_file" — colons and slashes — so offering them verbatim is
# rejected with an opaque HTTP 400 ("Upstream request failed") that names nothing.
# Measured 2026-07-28: "write_file" 200, "mcp:filesystem/write_file" 400,
# "mcp_filesystem_write_file" 200.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def sanitize_tool_name(name: str, taken: Dict[str, str]) -> str:
    """Map a Cambrian tool name onto a provider-legal function name.

    ``taken`` is the sanitized→original map built so far; it is BOTH the collision
    check and the reverse map. Collisions are disambiguated deterministically, because
    two runs of the same agent must offer the same names or a provider-side cache and
    our own logs stop lining up.
    """
    candidate = name if _TOOL_NAME_RE.match(name) else re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64]
    if not candidate:
        candidate = "tool"
    if taken.get(candidate, name) != name:
        base, i = candidate[:60], 2
        while taken.get(f"{base}_{i}", name) != name:
            i += 1
        candidate = f"{base}_{i}"
    return candidate


def normalize_tool_parameters(params: Any) -> Dict[str, Any]:
    """Coerce a tool's argument schema into one a provider will accept.

    Cambrian's kernel-owned tool registry stores MCP schemas in a degenerate form —
    `{"properties": {"content": {}, "path": {}}}` — with no top-level "type". That is
    fine for the prompt-encoded menu, which only ever renders the property NAMES, and
    it is rejected outright by a provider, which validates the schema.

    Measured 2026-07-28 against the live endpoint, holding everything else constant:

        {"properties": {...}}                    -> HTTP 400
        {"type": "object", "properties": {...}}  -> 200, tool_calls

    The 400 body says only "Upstream request failed" and names no field, so this is
    worth stating precisely rather than leaving to be rediscovered.

    Deliberately does NOT invent property types: an empty `{}` property means "any",
    which is valid and which the provider accepts once the top-level type is present.
    Guessing "string" would be a lie the provider then enforces.
    """
    if not isinstance(params, dict) or not params:
        return {"type": "object", "properties": {}}
    out = dict(params)
    out.setdefault("type", "object")
    if out.get("type") == "object":
        out.setdefault("properties", {})
    return out


class _ConversationMirror:
    """Wraps WorkingMemory so its writes ALSO land in the provider conversation.

    Under native tool-calling the model's history is the message list, not the prompt —
    the prompt is built once and never rebuilt. Everything the loop would otherwise
    communicate by re-rendering working memory (nudges, ALREADY-DONE notes, discovery
    results, memory-query results) has to become a real turn or the model never sees it.

    A wrapper rather than ~30 edited call sites: the loop keeps writing to `wm` exactly
    as it does on the text path, and one place decides what that means for the
    conversation. Tool RESULTS are deliberately NOT mirrored here — they are appended as
    proper `role="tool"` turns keyed by tool_call_id, and mirroring them as prose too
    would tell the model the same thing twice in two formats.
    """

    _MIRRORED = ("add_text", "add_reflection")

    def __init__(self, wm, conversation: List[Dict[str, Any]]):
        self._wm = wm
        self._conv = conversation
        # A tool turn MUST immediately follow the assistant turn that requested it.
        # Measured against the live endpoint, everything else held constant:
        #   user, assistant(call), tool          -> 200
        #   user, assistant(call), NOTE, tool    -> 400
        #   user, assistant(call), tool, NOTE    -> 200
        # The loop writes notes (recurrence vetoes, nudges, discovery results) between
        # requesting a tool and executing it, so mirroring them immediately would land
        # them in exactly that gap. They are held and flushed after the tool turn.
        self._holding = False
        self._held: List[Dict[str, Any]] = []

    def hold(self) -> None:
        """Stop appending directly: a tool call is awaiting its result turn."""
        self._holding = True

    def release(self) -> None:
        """Tool turn is in place; flush anything written while holding."""
        self._holding = False
        if self._held:
            self._conv.extend(self._held)
            self._held.clear()

    def _emit(self, message: Dict[str, Any]) -> None:
        (self._held if self._holding else self._conv).append(message)

    def __getattr__(self, name):
        attr = getattr(self._wm, name)
        if name not in self._MIRRORED:
            return attr

        def wrapped(content, *a, **kw):
            result = attr(content, *a, **kw)
            if isinstance(content, str) and content.strip():
                self._emit({"role": "user", "content": content})
            return result

        return wrapped

    def add_skill(self, name: str, body: str, note: str = "") -> None:
        """A skill loaded by a TEXT envelope while native tools are active (no tool turn
        to answer): the conversation is the model's only history, so the full body
        must land there as a turn, besides being pinned in working memory (ADR-0139)."""
        self._wm.add_skill(name, body, note)
        content = f"<skill name={name!r}>\n{body}\n</skill>"
        if note:
            content += f"\n<note>{note}</note>"
        self._emit({"role": "user", "content": content})

    def add_step(self, kind: str, pinned: bool = False, **fields):
        """Memory-query steps carry retrieved content the model must see;
        tool_call steps are already represented by their tool turn."""
        result = self._wm.add_step(kind, pinned=pinned, **fields)
        if kind == "memory_query":
            rendered = _render_memory_results(fields.get("results") or [])
            self._emit({
                "role": "user",
                "content": ("<memory_query query=" + repr(fields.get("query")) + ">"
                            + chr(10) + rendered + chr(10) + "</memory_query>"),
            })
        return result


# Discovery actions, expressed as NATIVE tool definitions.
#
# D7.3 withdrew the JSON `tool_call` action under native tool-calling, correctly: two
# ways to invoke one thing means the model picks the one that does nothing. But it also
# withdrew find_tools and describe_tool, which are not invocation — they are how the
# model widens a menu that is only ever a top-k GUESS at what the task needs.
#
# Measured 2026-07-28 (runs/rt_native_final, 2/6 against the text path's 5/6): the seeded
# subset was `edit_file, read_file, fast_edit_blocks` — no write_file, for a suite whose
# whole job is creating files. The model called edit_file 14 times on files that did not
# exist, searched 11 times looking for a way through, and then declared success. On the
# text path the same bad subset is survivable because discovery is one action away.
#
# So discovery comes back — find_tools ONLY, as a tool rather than a competing action
# encoding.
#
# describe_tool is deliberately NOT here. It exists to fetch the ADR-0045 Tier-2 spec
# because the PROSE menu renders argument names without types. A native tool definition
# already carries the full schema the provider validates against, so describe_tool would
# spend a round fetching what is already in front of the model. It stays on the text
# path, where the abbreviation is real.
_DISCOVERY_TOOL_DEFS = [
    {
        "name": "find_tools",
        "description": (
            "Find additional tools by capability when your current tools cannot do the "
            "job. Describe the need verb-first, e.g. 'write a file to disk'."
        ),
        "parameters": {
            "type": "object",
            "properties": {"need": {"type": "string", "description": "capability needed"}},
            "required": ["need"],
        },
    },
]

# Names the loop handles itself rather than dispatching to the tool plane.
_DISCOVERY_TOOL_NAMES = {d["name"] for d in _DISCOVERY_TOOL_DEFS}

# ADR-0139 D3/D8: skill verbs as NATIVE tool definitions. The loop already
# RECOGNISED a native use_skill call (_CONTROL_ACTION_NAMES) but never OFFERED one,
# and the call's tool turn was then answered "call was not executed": the model
# loaded a skill and was told nothing happened. Offered only when the agent has a
# skill menu: a load tool with nothing to load is the same shape as offering
# find_tools to an opted-out agent (see build_tool_definitions). The use_skill turn
# is answered with the skill's FULL body (run_think, kind == "use_skill").
_SKILL_TOOL_DEFS = [
    {
        "name": "use_skill",
        "description": (
            "Load a skill (an authored, step-by-step procedure) by name. Its full "
            "instructions come back as the result and stay in force for the rest of "
            "the run. Name only a skill from the skill menu in the prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {"skill": {"type": "string", "description": "skill name from the menu"}},
            "required": ["skill"],
        },
    },
    {
        "name": "find_skills",
        "description": (
            "Find more skills (authored procedures) for a need the skill menu does not "
            "cover. Describe the procedure you want, e.g. 'investigate a Python codebase'."
        ),
        "parameters": {
            "type": "object",
            "properties": {"need": {"type": "string", "description": "procedure needed"}},
            "required": ["need"],
        },
    },
]

# The loop's OWN vocabulary, as the model may emit it under native tool-calling.
#
# <ActionProtocol> advertises these as actions. A model given native tools
# expresses an action the only way that interface offers — as a tool call — so
# `final_answer` arrives as `{"name": "final_answer"}` rather than as text. Only
# `find_tools` was recognised, so every other verb was forwarded to the kernel's
# tool registry and refused as an unknown tool.
#
# `final_answer` made that fatal rather than untidy: the model tried to FINISH,
# was told the tool does not exist, tried again, was refused again, and
# eventually summarised in prose — which is what a caller saw as "the planner
# answered in prose". The loop was refusing its own protocol.
#
# A kernel tool that happens to share one of these names is shadowed. That is the
# right way round: these words belong to the loop, and a deployment that wants a
# tool called `final_answer` can name it something else.
_CONTROL_ACTION_NAMES = _DISCOVERY_TOOL_NAMES | {
    "final_answer",
    "memory_query",
    "describe_tool",
    "use_skill",
    "find_skills",
    "yield_subgoal",
    "resolve_cid",
}


def build_tool_definitions(agent, system_tools, include_discovery: bool = True,
                           include_skills: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Render the agent's tool menu as native tool definitions (ADR-0097 Phase B).

    Returns ``(definitions, name_map)`` where ``name_map`` is sanitized→original. The
    map is not optional: the provider echoes back the name IT was given, and the
    kernel's tool registry only knows the original, so a call would be dispatched to a
    tool that does not exist.

    Same two sources as the prompt-encoded menu — local ``@tool`` functions and
    kernel-owned system tools — so the model is offered exactly the same set either
    way. Only the ENCODING differs: a schema the provider enforces, rather than a JSON
    example the model is asked to imitate.

    ``include_discovery=False`` drops ``find_tools``. It follows ``seed_system_tools``,
    because OFFERING a tool the loop will refuse to run is not a smaller version of
    granting it — it is worse. The execution side already refuses ``find_tools`` for an
    opted-out agent ("must not be able to pull them back in via discovery"), but this
    side advertised it anyway, so an opted-out agent was shown a tool it could only be
    denied for using. For an agent with a zero tool-round budget that is fatal rather
    than merely untidy: the model calls it, the loop will not execute it, and the next
    request carries an assistant turn with ``tool_calls`` and no tool turn answering
    them — which every OpenAI-compatible provider rejects with a 400.

    ``include_skills=True`` adds ``use_skill`` / ``find_skills`` (ADR-0139 D8); the
    loop passes it when the agent has a non-empty skill menu.
    """
    defs: List[Dict[str, Any]] = []
    name_map: Dict[str, str] = {}

    # Added FIRST so they keep their plain names; a real tool that happens to share one
    # is disambiguated by sanitize_tool_name instead.
    control_defs = (list(_DISCOVERY_TOOL_DEFS) if include_discovery else []) + (
        list(_SKILL_TOOL_DEFS) if include_skills else [])
    for d in control_defs:
        name_map[d["name"]] = d["name"]
        defs.append(dict(d))

    def _add(name: str, description: str, params: Any) -> None:
        if not name:
            return
        safe = sanitize_tool_name(name, name_map)
        name_map[safe] = name
        defs.append({"name": safe, "description": description,
                     "parameters": normalize_tool_parameters(params)})

    for spec in agent.tools.specs():
        _add(spec.name, (getattr(spec, "description", "") or "").strip(),
             spec.schema or {"type": "object", "properties": {}})

    for t in system_tools or []:
        raw = t.get("schema_json")
        params: Any = {"type": "object", "properties": {}}
        if isinstance(raw, str) and raw.strip():
            try:
                params = json.loads(raw)
            except ValueError:
                pass
        elif isinstance(raw, dict):
            params = raw
        _add(t.get("name", ""), (t.get("description") or "").strip(), params)

    return defs, name_map


def action_from_native_turn(text: str, calls: List[Dict[str, str]], stop_reason: str,
                            name_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Map a native tool-calling turn onto the loop's internal action dict.

    Both paths converge here so the loop keeps ONE shape and one set of guards —
    recurrence veto, tool-round budget, the inferred-answer re-prompt (ADR-0097 D5).

    Three cases, in the order that matters:

    1. A tool call is present -> act on it, WHATEVER the stop reason says. Some
       providers report "stop" while returning calls (opencode #14972), so the action
       outranks the narration.
    2. No call and an explicit "end_turn" -> a DECLARED final answer.
    3. Anything else — "max_tokens", "refusal", "unknown", a blank reason — is
       stopped-but-not-finished. Marked `_inferred` so the existing re-prompt handles
       it, which is the same treatment unparseable prose gets on the text path.
    """
    if calls:
        first = calls[0]
        args: Any = {}
        raw = first.get("arguments") or ""
        if raw:
            try:
                args = json.loads(raw)
            except ValueError:
                # A malformed argument blob is a failed action, not a final answer.
                return {"action": "_truncated", "raw": raw}
        if not isinstance(args, dict):
            args = {"value": args}
        provider_name = first.get("name", "")
        # The loop's own verbs are handled HERE, not dispatched to the tool plane.
        if provider_name in _CONTROL_ACTION_NAMES:
            act: Dict[str, Any] = {"action": provider_name}
            act.update(args)
            return act
        # Back to the name the kernel's tool registry knows. Dispatching the
        # sanitized name would look up a tool that does not exist.
        real_name = (name_map or {}).get(provider_name, provider_name)
        return {"action": "tool_call", "tool": real_name, "args": args}

    # The model may write a CONTROL ENVELOPE as plain text instead of emitting a native
    # tool call — `{"action": "memory_query", ...}` arriving as prose. Wrapping that as a
    # final answer hands the user raw JSON and silently drops the step the model asked for,
    # so honour the envelope instead. parse_action already knows this shape; reusing it
    # keeps the two paths from drifting apart.
    # NOT just a leading "{": the envelope usually TRAILS a sentence of the model narrating
    # itself -- a line of prose, then the action object. Requiring the brace at the START
    # is exactly what let that shape through. parse_action already scans for the object.
    if text and "{" in text and '"action"' in text:
        parsed = parse_action(text)
        if parsed.get("action") not in ("final_answer", "_truncated"):
            return parsed
        # A DECLARED final_answer is honoured as written, including its "answer".
        #
        # Falling through to the wrapper below discarded the envelope and made the
        # whole raw string the answer — so an agent whose output contract puts a
        # structured payload in "answer" (which is what _PER_TURN_OUTPUT_CONTRACT
        # instructs: "its answer field MUST follow this format") received its own
        # envelope back, JSON and all, instead of the payload. Every consumer then
        # had to re-parse a shape the loop had already parsed and thrown away.
        #
        # Nor is it "inferred": the model said final_answer. Marking a declared
        # answer as a guess triggers the re-prompt meant for models that merely
        # stopped talking.
        if parsed.get("action") == "final_answer" and "answer" in parsed:
            return parsed

    if stop_reason == "end_turn":
        return {"action": "final_answer", "answer": text, "type": "text"}

    return {"action": "final_answer", "answer": text, "type": "text", "_inferred": True}


def _answer_unexecuted(conversation: List[Dict[str, Any]], call_ids: List[str]) -> List[str]:
    """Close out tool calls the loop chose not to run, and return the emptied list.

    A model may emit SEVERAL tool calls in one turn. This loop runs one action per
    turn, so only the first is executed — but the assistant turn is echoed back
    verbatim carrying all of them, and every OpenAI-compatible provider requires a
    tool message per tool_call_id:

        An assistant message with 'tool_calls' must be followed by tool messages
        responding to each 'tool_call_id'. (insufficient tool messages following
        tool_calls message)

    Leaving the extras unanswered therefore 400s the NEXT request, which reads as
    the model being unavailable rather than as a malformed conversation we built.
    Answering them explicitly keeps the transcript valid AND tells the model why
    its second call produced nothing, so it retries deliberately instead of
    assuming the tool is broken.
    """
    for cid in call_ids:
        if not cid:
            continue
        conversation.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": json.dumps({
                "error": "not executed: one action per turn. Re-issue this call on its own turn if you still need it.",
            }),
        })
    return []


def parse_action(raw: str) -> Dict[str, Any]:
    """Parse the LLM's JSON action, tolerating surrounding prose/code fences.

    A response that cannot be parsed is returned as a final_answer envelope carrying
    ``_inferred: True``. The flag matters: an INFERRED answer is a guess that the
    model meant to stop, not a declaration that it did. The loop treats the two
    differently (see INFERRED_FINAL_ANSWER_NOTE); ``_reflect`` still only cares that
    prose parses to a final_answer at all, so its contract is unchanged.

    A truncated action (#2) is neither — the loop handles it as a recoverable error.
    """
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except (ValueError, TypeError):
            pass
    if _looks_like_truncated_action(text):
        return {"action": "_truncated", "raw": text}
    return {"action": "final_answer", "answer": text, "type": "text", "_inferred": True}


def run_think(
    agent,
    task: AgentTask,
    *,
    role: Optional[str] = None,
    output_schema: str = "",
    constraints: Optional[List[str]] = None,
    result_type: Optional[str] = None,
    seed_recall: bool = True,
    seed_system_tools: bool = True,
    allow_find_tools: Optional[bool] = None,
    allow_yield_subgoal: bool = True,
    max_memory_queries: int = _DEFAULT_MAX_MEMORY_QUERIES,
    max_tool_rounds: int = _DEFAULT_MAX_TOOL_ROUNDS,
    max_tokens: int = 4096,  # #1: 1024 truncated substantive outputs (e.g. a tool_call
    # whose `content` arg is a multi-KB file) mid-emit; 4096 gives single-shot headroom.
    # Past this, content must be chunked across turns (see the truncation guard, #2).
    temperature: float = 0.7,
    recurrence_enabled: bool = True,
    recurrence_threshold: float = 0.85,
    recurrence_veto_depth: int = 1,
    reflect_enabled: bool = True,
    max_reflections: int = _DEFAULT_MAX_REFLECTIONS,
    max_inline_resolves: int = _DEFAULT_MAX_INLINE_RESOLVES,
    cache_ttl: Optional[str] = None,
) -> AgentResult:
    """Drive the reason/retrieve/act loop for one task and return the final result.

    The retrieval loop always fires: the scratchpad is seeded with the
    server-assembled Global Workspace (``task.working_memory``, ADR-0022) and a
    **mandatory initial LTM recall** on the task (Path B kickoff), then the LLM may
    issue further capped ``memory_query`` / ``tool_call`` rounds before answering.

    ``result_type`` forces the final ``AgentResult.type`` (e.g. ``code`` → executor),
    overriding whatever ``type`` the LLM puts on its ``final_answer``.

    ``cache_ttl`` is the lifetime of the system prompt's cache breakpoint
    (``"short"`` | ``"long"``). ``None`` resolves to the agent's ``cache_ttl``
    attribute, then ``CAMBRIAN_AGENT_CACHE_TTL``, then ``"short"``: an agent run is
    machine-paced (ADR-0139 D1).
    """
    from . import assemble_context

    role = role or getattr(agent, "role", None) or agent.description
    # ADR-0041 D1/D2: typed working memory replaces the flat scratchpad. When the
    # substrate exposes the Embed RPC, entries are embedded for relevance-ranked,
    # bounded prompt assembly; otherwise assembly degrades to recency.
    embed_fn = _make_embed_fn(agent)
    # ADR-0048 D4/R7: wire the content offloader so a large working-memory text
    # block (a heavy recall, a big workspace seed) is replaced in the prompt by a
    # {gist + cid} pointer instead of being re-inlined every turn. The blob is
    # owner-stamped to this session (read-gated); drill-down via get_context_node.
    # Degrades to verbatim when the substrate has no write path (put returns None).
    _substrate = getattr(agent, "substrate", None)
    _offload_fn = None
    if _substrate is not None and hasattr(_substrate, "put_context_node"):
        _offload_fn = lambda text: _substrate.put_context_node(text, session_token_id=task.session_token_id)  # noqa: E731
    wm = WorkingMemory(embed_fn=embed_fn, offload_fn=_offload_fn)
    intent_vec = embed_fn(task.text) if (embed_fn and task.text.strip()) else None
    mem_queries = 0
    tool_rounds = 0
    find_tools_calls = 0  # ADR-0044: cap the discovery pull to avoid loops
    describe_calls = 0  # ADR-0045: cap the describe_tool Tier-2 fetch
    find_skills_calls = 0  # ADR-0046: cap the find_skills discovery pull
    skill_loads = 0  # ADR-0046: cap use_skill loads
    loaded_skills = set()  # ADR-0046: skills already loaded this run (idempotent)
    # ADR-0046 D6: tools a loaded skill conferred this run. Kept apart from
    # system_tools because a native->text latch RE-LISTS system_tools from the kernel,
    # whose ListTools serves standing grants only; without this they would silently
    # drop out of the menu again after the latch.
    skill_tools: List[Dict] = []
    mem_exhausted = False
    tool_exhausted = False
    veto_counts: Dict[str, int] = {}  # ADR-0041 D4: per-action hard-veto tally
    success_dedup_counts: Dict[str, int] = {}  # ADR-0041 D4: per-action already-succeeded tally
    truncation_retries = 0  # #2: consecutive token-limit truncations re-prompted
    inferred_answer_retries = 0  # non-action responses re-prompted (see note above)
    unknown_action_retries = 0  # unrecognised action kinds re-prompted (_MAX_UNKNOWN_ACTION_RETRIES)
    reflections: List[str] = []  # ADR-0052: verbal reflections extracted on hard-veto
    inline_resolves: int = 0   # ADR-0048 residual: budget for resolve_cid as="inline"

    # Seed A — the Global Workspace context the Substrate already assembled (given).
    # Wire fetch_fn so a high-precision prior-step result is resolved to its FULL body
    # from CAS (read-gated to this session), not served as a 500-char snippet — otherwise
    # a "write the previous step's output" task only sees the truncated head and the agent
    # regenerates a different summary. The cid is also surfaced for by-reference passing.
    given = ""
    if task.working_memory:
        _sub = getattr(agent, "substrate", None)
        fetch_fn = None
        if _sub is not None and hasattr(_sub, "get_context_node"):
            fetch_fn = lambda cid: _sub.get_context_node(cid, task.session_token_id)  # noqa: E731
        given = assemble_context(task.working_memory, fetch_fn=fetch_fn)
    if given:
        # The seed's CID is the canonical "workspace has content" pointer: the model
        # can pass it as ``{"$cid": "<seed_cid>"}`` in a tool_call to write the full
        # workspace content to a file in one call. (Surface this hint inside the
        # block so the model can't miss it.)
        wm.add_text(
            f"<workspace>\n{given}\n"
            f"<note>If the workspace content is offloaded (you see `cid='…'` on this "
            f"block), pass the cid as {{\"$cid\": \"<cid>\"}} in a tool_call arg — the "
            f"kernel resolves it; do NOT re-emit the body.</note>\n</workspace>"
        )

    # Seed B — a mandatory initial recall so every reasoning agent consults LTM first
    # (the agent-initiated retrieval loop, not just server-pushed context).
    # v2: emitted as a typed Step(kind="memory_query") so the query + status
    # are both visible in the trajectory (issue #1: "memory queries don't show
    # what was queried").
    if seed_recall and getattr(agent, "memory", None) is not None and task.text.strip():
        mem_queries += 1
        seed_q = task.text.strip()
        seed_results = _safe_recall(agent, seed_q, task.session_token_id)
        wm.add_step(
            kind="memory_query",
            query=seed_q,
            status="ok" if seed_results else "empty",
            results=seed_results,
        )

    # Seed C — a RESUMED yield (ADR-0037 D10 delegate-and-continue): the kernel
    # re-dispatched us with a delegated sub-goal's result. Surface it so the
    # Executive incorporates it and answers, rather than re-yielding the same intent.
    _ctx = getattr(task, "context", None) or {}
    if _ctx.get("_yield_result"):
        resumed_intent = _ctx.get("_yield_resumed_intent", "")
        wm.add_text(
            f"<delegated intent={resumed_intent!r}>{_ctx['_yield_result']}</delegated>\n"
            "<note>The sub-task you delegated has returned above. Use its result to "
            "answer — do NOT delegate it again.</note>"
        )

    # Fetch the granted kernel system tools ONCE (not per round — it is an RPC) so
    # the LLM sees a closed menu of the tools it may actually call (ADR-0039).
    # ADR-0044: pass the task as the relevance query so the kernel serves only the
    # top-k task-relevant tools (the push) instead of the whole registry — a
    # task-sized menu, not ~15k tokens of every granted tool.
    # An agent may opt into the FULL granted tool menu (seed_tools_full=True) instead of the
    # ADR-0044 task-relevant top-k. A domain agent (e.g. a customer-service session with a
    # fixed MCP toolset) wants every domain tool visible every turn so it calls them directly
    # rather than discovering them via find_tools — query="" ⇒ the kernel serves the full menu.
    # seed_system_tools=False opts the agent OUT of kernel-owned system tools entirely: the
    # menu then holds only the agent's in-process @tools (e.g. a chat front-desk whose only
    # "tool" is delegate_to_planner + memory recall — it must never execute tasks itself,
    # only hand them to the planner). find_tools is disabled too, so it cannot pull them back.
    _seed_tool_query = "" if getattr(agent, "seed_tools_full", False) else task.text
    # Seeded at Tier-2 because the native path is assumed available (it is latched off
    # on the first refusal, which also refetches Tier-1 and recomposes the menu). One
    # fetch, not two: under native tool-calling the prose menu no longer lists tools at
    # all, so Tier-1 has no consumer.
    # find_tools is separable from the seeded menu, and the distinction is the
    # difference between two legitimate agent shapes:
    #
    #   - A front desk with NO tools at all (chat): seed nothing, and disable
    #     discovery too, or it pulls back the menu it was denied.
    #   - A researcher that should DISCOVER what it needs (source discovery):
    #     seed nothing, because a preloaded top-k menu of every tool in the
    #     deployment is mostly noise — but keep find_tools, which is how it asks
    #     for a capability by name when it knows what it wants.
    #
    # Defaulting to seed_system_tools preserves the first shape for callers that
    # say nothing; the second asks for it explicitly.
    find_tools_enabled = seed_system_tools if allow_find_tools is None else bool(allow_find_tools)
    system_tools = _list_system_tools(agent, query=_seed_tool_query, full=True) if seed_system_tools else []
    # ADR-0046: the loadable skill menu — agent-local skills (always present) first,
    # then task-relevant system skills (the push), with same-name system skills
    # shadowed by agent-local ones (structural prioritization, no central ranking).
    system_skills = _assemble_skill_menu(agent, query=task.text)

    # #4: the ActionProtocol is loop-INVARIANT (agent, granted tools/skills, and the
    # domain answer-format are fixed for the task). Compose it ONCE so it is identical
    # bytes every turn — a stable prefix a provider can cache — instead of rebuilding
    # ~25 lines of menu+rules each round.
    # ADR-0097 Phase B. Assumed available and latched off on the first
    # ToolCallingUnsupported — optimistic because the common deployment HAS it, and a
    # capability probe per run would cost a round-trip to learn what the first real
    # call tells us anyway.
    native_tools = True
    # find_tools follows the same opt-out as the system menu it exists to reach.
    # ADR-0139 D8: skill verbs are native tools whenever there is a skill menu.
    tool_defs, tool_name_map = build_tool_definitions(
        agent, system_tools, include_discovery=find_tools_enabled,
        include_skills=bool(system_skills))
    skill_chars = 0  # ADR-0139 D3: total chars of skill bodies pinned this run
    # ADR-0097 D8: under native tool-calling the model's history is THIS list, not a
    # re-rendered prompt. It is seeded once with the composed prompt and thereafter
    # grows by real turns — assistant turns carrying tool_calls, tool turns carrying
    # their results. Empty on the text path, which keeps rebuilding the prompt.
    conversation: List[Dict[str, Any]] = []
    # The id of the call awaiting its tool turn. Providers correlate on it and
    # reject a synthesized one, so it is carried rather than regenerated.
    pending_tool_call_id = ""
    # The OTHER ids from the same turn, when the model emitted parallel calls.
    #
    # This loop runs one action per turn, so only the first is executed — but the
    # assistant turn is echoed back verbatim with ALL of them, and a provider
    # requires a tool message per tool_call_id:
    #
    #   An assistant message with 'tool_calls' must be followed by tool messages
    #   responding to each 'tool_call_id'.
    #
    # Answering only the first therefore 400s the NEXT request and takes the whole
    # run down. These ids get an explicit "not executed" reply instead, which
    # satisfies the invariant and tells the model why its other call did not run.
    pending_extra_call_ids: List[str] = []

    def _answer_native_turn(content: str) -> bool:
        """Answer the pending native tool call with ``content`` (ADR-0139 D8).

        A control verb that arrived as a NATIVE call (use_skill, find_skills) must get
        its result in the tool turn the provider correlates with the call. Otherwise
        the next round closes it with "call was not executed", which tells the model
        its skill load did nothing. Returns False on the text path (nothing pending).
        """
        nonlocal pending_tool_call_id, pending_extra_call_ids
        if not (conversation and pending_tool_call_id):
            return False
        conversation.append({"role": "tool", "tool_call_id": pending_tool_call_id,
                             "content": content})
        pending_tool_call_id = ""
        pending_extra_call_ids = _answer_unexecuted(conversation, pending_extra_call_ids)
        if isinstance(wm, _ConversationMirror):
            wm.release()
        return True

    def _control_reply(text: str) -> None:
        """Report a control verb's outcome: as the native tool result when the verb
        arrived as a native call, else as a working-memory note. On the native path
        the note goes to the UNDERLYING memory only, so the mirror does not repeat the
        tool result to the model as a second, prose user turn."""
        if _answer_native_turn(text):
            (wm._wm if isinstance(wm, _ConversationMirror) else wm).add_text(text)
        else:
            wm.add_text(text)

    # The menu follows the BUDGET, so it never advertises what the loop refuses.
    allow_memory = max_memory_queries > 0
    action_protocol = _compose_action_protocol(agent, output_schema, system_tools, system_skills,
                                                allow_yield_subgoal=allow_yield_subgoal,
                                                native_tools=bool(native_tools and tool_defs),
                                                allow_memory=allow_memory)

    # Host facts + the path rules that make them usable (ADR-0100 P3 residual #3).
    # Composed ONCE, before the loop: the values are cached and constant for the
    # process, so keeping this out of the per-round list preserves the byte-identical
    # cacheable prefix that the v2 prompt layout exists to protect.
    #
    # This is the executing half of the split with the kernel's planner prompt. The
    # planner is told the OS and told that paths are not its job; the agent runs on
    # the host, so it is told the absolute values and is the one that resolves "a
    # folder on the desktop" into a real path. Empty when nothing could be
    # determined, which stays silent rather than asserting a guessed path.
    host_constraint = host_block()

    # ADR-0139 D1/D2: the prompt is sent as a cached system tier plus messages. The
    # cache lifetime follows caller pacing: an agent run is machine-paced, so SHORT
    # by default; an agent serving a human-paced session may set `cache_ttl = "long"`
    # (class attribute), or a deployment CAMBRIAN_AGENT_CACHE_TTL.
    run_cache_ttl = _resolve_cache_ttl(agent, cache_ttl)
    # Latched per run, like native_tools:
    #   structured_text - the text path may use the structured call. Off after the
    #     kernel refused it (InvalidRequest) or reported "system" as ignored: the
    #     rest of the run sends the legacy single prompt.
    #   fold_system - the native path folds the system text into the first user turn
    #     because the kernel (or model) dropped the `system` field.
    structured_text = True
    fold_system = False
    native_system_kw = None  # None = unknown; resolved on the first native call
    native_system: List[Dict[str, Any]] = []
    _log_prompt_versions(agent)

    round_no = 0
    while True:
        extra = list(constraints or [])
        if host_constraint:
            extra.append(host_constraint)
        # Volatile notices go into the message stream, never the system tier: the
        # system text must be byte-identical across the turns of a run (D2).
        notes = [_nudge("memory-exhausted")] if mem_exhausted else []
        # ADR-0139 Phase 0: the prompt speaks ONE protocol (the same condition that
        # composed action_protocol) and states the budgets this loop enforces.
        native_prompt = bool(native_tools and tool_defs)
        parts = build_prompt_parts(
            role=role,
            task=task.text,
            # #3 trajectory reframe: number the bounded buffer as an ordered loop
            # history and stamp the current step, so the model reads its own progress
            # ("you are at step N; here is what you did") instead of a flat tag pile.
            context_str=wm.assemble(intent_vec, numbered=True),
            step_no=round_no + 1,
            # ADR-0048 D8: the action menu + behavioral rules go in <ActionProtocol>;
            # <OutputSchema> keeps only the per-turn action contract.
            action_protocol=action_protocol,  # #4: composed once above (loop-invariant)
            output_schema=(_PER_TURN_OUTPUT_CONTRACT_NATIVE if native_prompt
                           else _PER_TURN_OUTPUT_CONTRACT),
            constraints=extra or None,
            base_constraints=default_constraints(
                native_tools=native_prompt, max_memory_queries=max_memory_queries,
                max_tool_rounds=max_tool_rounds),
            base_anti_patterns=default_anti_patterns(
                native_tools=native_prompt,
                large_write_tool=(_LARGE_WRITE_TOOL
                                  if _menu_has_tool(agent, system_tools, _LARGE_WRITE_TOOL)
                                  else None)),
            notes=notes,
        )
        # No client-side timeout on the agent's LLM call: passing the inbound RPC's
        # remaining budget here makes the managed-generate gRPC deadline strangle a
        # slow model (esp. the graded interview, where there is no step deadline to
        # respect). timeout_ms=0 ⇒ the SDK sends no gRPC deadline; cancellation is
        # governed by the model client's ctx, not a fixed wall-clock cap.
        # ADR-0097 Phase B: prefer the provider's STRUCTURED signal when this
        # deployment has it. The text path below is the documented fallback, not a
        # deprecated branch — local and self-hosted models keep using it.
        action = None
        # getattr, not a direct call: a substrate predating this method (an older SDK,
        # or any custom transport) must take the fallback, not raise AttributeError
        # mid-loop. "The capability is absent" and "the object cannot express it" are
        # the same answer to the caller.
        # Every requested call must be answered before the next turn. A call can go
        # unexecuted — the recurrence veto blocks a repeat, a budget trips, the tool is
        # not granted — and leaving it unanswered dangles the assistant turn, which a
        # strict provider rejects. Close it explicitly, then release the mirror so held
        # notes land AFTER the tool turn rather than inside the assistant/tool pair.
        if pending_tool_call_id and conversation:
            conversation.append({
                "role": "tool",
                "tool_call_id": pending_tool_call_id,
                "content": json.dumps({"error": "call was not executed"}),
            })
            pending_tool_call_id = ""
            pending_extra_call_ids = _answer_unexecuted(conversation, pending_extra_call_ids)
            if isinstance(wm, _ConversationMirror):
                wm.release()

        native_fn = getattr(agent.substrate, "generate_with_tools", None) if native_tools else None
        if native_tools and native_fn is None:
            # The substrate cannot express native tool-calling at all (an older SDK, a
            # custom transport, a test double). That is the SAME answer as a refusal and
            # must latch identically — otherwise the prompt stays composed for native
            # mode, which OMITS the tool list, while the loop runs the text path that
            # depends on it. The model would then see no tools at all: attached to
            # nothing, listed nowhere.
            native_tools = False
            logger.info("react_native_tools_absent")
            if isinstance(wm, _ConversationMirror):
                wm = wm._wm
            conversation.clear()
            pending_tool_call_id = ""
            pending_extra_call_ids = []
            if seed_system_tools:
                system_tools = _list_system_tools(agent, query=_seed_tool_query, full=False)
            system_tools = _merge_tools(system_tools, skill_tools)
            action_protocol = _compose_action_protocol(
                agent, output_schema, system_tools, system_skills,
                allow_yield_subgoal=allow_yield_subgoal, native_tools=False,
                allow_memory=allow_memory)
            continue
        if native_fn is not None and tool_defs:
            if native_system_kw is None:
                # A custom substrate written before C1 cannot take `system=`; it gets
                # the system text inside the first user turn, as before (ADR-0139).
                native_system_kw = _accepts_kwarg(native_fn, "system")
                if not native_system_kw:
                    fold_system = True
            if not conversation:
                # Seed once. The system blocks carry role, rules and the action menu;
                # the first user turn carries the task and the seeded trajectory. From
                # here the conversation carries what happened, so neither is rebuilt —
                # re-narrating the model's own actions back at it as prose is what made
                # it summarise its history instead of continuing it.
                native_system = list(parts.system)
                first = parts.messages[0]["content"]
                conversation.append({"role": "user", "content": (
                    parts.legacy_prompt() if fold_system else first)})
                wm = _ConversationMirror(wm, conversation)
            kw: Dict[str, Any] = {}
            if native_system_kw and not fold_system:
                kw = {"system": native_system, "cache_ttl": run_cache_ttl}
            _log_llm_call("native", round_no + 1, native_system if not fold_system else [],
                          len(conversation))
            try:
                n_res = native_fn(
                    task.session_token_id,
                    conversation,
                    tool_defs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_ms=0,
                    **kw,
                )
                n_text, n_calls, n_stop = n_res
                n_ignored = list(getattr(n_res, "ignored", None) or [])
                if "system" in n_ignored and not fold_system:
                    # The kernel (an older build) or the allocated model dropped the
                    # system blocks: the model saw no role, rules or menu. Discard the
                    # turn and resend with the system text folded into the first user
                    # turn, for the rest of the run.
                    fold_system = True
                    logger.warning("react_system_ignored_folding", extra={
                        "path": "native", "legacy_kernel": bool(getattr(n_res, "legacy_kernel", False))})
                    if conversation and conversation[0].get("role") == "user":
                        conversation[0] = {"role": "user", "content": "\n\n".join(
                            p for p in ("\n\n".join(b["text"] for b in native_system),
                                        conversation[0].get("content") or "") if p)}
                    continue
                _log_llm_result("native", n_res)
                # Echo the assistant turn back VERBATIM, tool_calls included. This is
                # what lets the model see that the call was its own.
                assistant_turn: Dict[str, Any] = {"role": "assistant", "content": n_text or ""}
                if n_calls:
                    assistant_turn["tool_calls"] = n_calls
                conversation.append(assistant_turn)
                action = action_from_native_turn(n_text, n_calls, n_stop, tool_name_map)
                if n_calls:
                    pending_tool_call_id = n_calls[0].get("id", "")
                    # Parallel calls: the turn carries all of them, so all of them
                    # must be answered even though only the first is executed.
                    pending_extra_call_ids = [
                        c.get("id", "") for c in n_calls[1:] if c.get("id", "")
                    ]
                    if isinstance(wm, _ConversationMirror):
                        wm.hold()
            except ToolCallingUnsupported as exc:
                # Latch OFF for the rest of the run rather than re-asking every turn:
                # the answer cannot change mid-run, and retrying would add a failed
                # RPC per round. Logged at INFO because taking the fallback is a
                # normal deployment state, not a fault.
                native_tools = False
                logger.info("react_native_tools_unavailable", extra={"reason": str(exc)})
                # Back to the text path: the prompt is rebuilt from working memory each
                # turn, so the half-built conversation is dead weight and the mirror
                # would keep duplicating writes into it.
                if isinstance(wm, _ConversationMirror):
                    wm = wm._wm
                conversation.clear()
                pending_tool_call_id = ""
                pending_extra_call_ids = []
                # Back on the prose menu, which wants Tier-1: full schemas would bloat
                # every prompt with detail the menu never renders.
                if seed_system_tools:
                    system_tools = _list_system_tools(agent, query=_seed_tool_query, full=False)
                system_tools = _merge_tools(system_tools, skill_tools)
                # The action menu is composed ONCE before the loop and currently says
                # "your tools are attached to this request". That is now false, so
                # recompose and retake the turn — one extra call, only in the rare
                # latch case, versus a whole run prompted for tools it will never be
                # given.
                action_protocol = _compose_action_protocol(
                    agent, output_schema, system_tools, system_skills,
                    allow_yield_subgoal=allow_yield_subgoal,
                    native_tools=False, allow_memory=allow_memory)
                continue

        if action is None:
            raw, structured_text = _generate_text_turn(
                agent, task, parts, structured=structured_text, cache_ttl=run_cache_ttl,
                max_tokens=max_tokens, temperature=temperature, round_no=round_no + 1)
            action = parse_action(raw)
        kind = action.get("action")

        round_no += 1
        logger.info(
            "react_round",
            extra={
                "round": round_no,
                "action": kind,
                "mem_queries": mem_queries,
                "tool_rounds": tool_rounds,
                "mem_exhausted": mem_exhausted,
            },
        )

        if kind == "_truncated":
            # #2: the model's action was cut off at the token limit — it was NOT a
            # final answer and the intended tool_call never ran. Surface it as a
            # RECOVERABLE observation that steers toward chunked authoring, so the
            # agent gets the second turn it needs to decompose, instead of the loop
            # silently finalizing a broken blob.
            truncation_retries += 1
            if truncation_retries > _MAX_TRUNCATION_RETRIES:
                logger.warning("react_truncation_exhausted", extra={"round": round_no})
                return AgentResult.from_text(
                    "Could not complete the task: the response kept exceeding the output "
                    "limit. Break the output into smaller pieces (e.g. write a large file "
                    "in sections with append) and retry.",
                    type="error",
                )
            logger.info("react_truncated_action", extra={"round": round_no, "retry": truncation_retries})
            # ADR-0139 Phase 0: name the sectioned-write tool only when it is in this
            # agent's menu. Recommending a tool the model cannot call sends it hunting
            # for one it will never find, on the turn it most needs a usable plan.
            if _menu_has_tool(agent, system_tools, _LARGE_WRITE_TOOL):
                how = _nudge("truncated-how-tool", tool=_LARGE_WRITE_TOOL)
            else:
                how = _nudge("truncated-how-none")
            wm.add_text(_nudge("truncated", how=how))
            continue

        if kind == "memory_query":
            query = action.get("query", "")
            if mem_queries >= max_memory_queries:
                # Graceful degradation. First time over budget: nudge the LLM to
                # answer best-effort. If it *still* insists on querying, terminate
                # with a best-effort answer rather than loop forever (never crash).
                if mem_exhausted:
                    logger.warning("react_memory_budget_exceeded", extra={"round": round_no})
                    return AgentResult.from_text(_best_effort(wm), type=result_type or "text")
                mem_exhausted = True
                wm.add_text(_nudge("memory-budget"))
                continue
            mem_queries += 1
            results = _safe_recall(agent, query, task.session_token_id)
            logger.info(
                "react_memory_query",
                extra={"round": round_no, "query": query, "results_count": len(results)},
            )
            # v2: emit as a typed Step so the query is visible in the trajectory.
            wm.add_step(
                kind="memory_query",
                query=query,
                status="ok" if results else "empty",
                results=results,
            )
            continue

        if kind == "tool_call":
            name = action.get("tool", "")
            args = action.get("args", {})

            # ADR-0041 D4: recurrent reconciliation gate. Before committing the
            # action, reconcile it against the episodic buffer — a local escalation
            # ladder (mirrors ADR-0037 D3) that stops the loop re-issuing a failing
            # call. wm.cards is the FULL episodic buffer (even cards 0041-03 evicted
            # from the prompt), so the gate's memory is complete.
            if recurrence_enabled:
                avec = embed_fn(action_text(name, args)) if embed_fn else None
                failed_dups = count_failed_duplicates(name, args, avec, wm.cards, recurrence_threshold)
                if failed_dups >= 2:
                    sig = action_text(name, args)
                    veto_counts[sig] = veto_counts.get(sig, 0) + 1
                    if veto_counts[sig] > recurrence_veto_depth:
                        # Escalate (→ Fail): honest failure, never spin.
                        logger.warning("react_recurrence_escalate",
                                       extra={"round": round_no, "tool": name})
                        return AgentResult.from_text(
                            f"Could not complete the task: the action {name!r} failed repeatedly "
                            f"and further retries were vetoed to avoid a loop.",
                            type="error",
                        )
                    # Hard veto (→ ReFrame): block, instruct, re-prompt without executing.
                    logger.info("react_recurrence_veto",
                                extra={"round": round_no, "tool": name, "failed_dups": failed_dups})
                    # ADR-0052: verbal self-reflection (Reflexion). The structural gate
                    # has caught the repeated failure; on the FIRST hard-veto for this
                    # action, ask the LLM *why* it failed and *what* to change, and pin
                    # the lesson into working memory so every later attempt reads it
                    # (the gate counts; the reflection learns). Bounded by veto depth +
                    # max_reflections; a reflection failure degrades to the terse note.
                    if reflect_enabled and veto_counts[sig] == 1 and len(reflections) < max_reflections:
                        lesson = _request_reflection(
                            agent, role, task, name, args, wm, reflections, temperature
                        )
                        if lesson:
                            reflections.append(lesson)
                            wm.add_reflection(lesson)
                            logger.info("react_reflection",
                                        extra={"round": round_no, "tool": name, "lesson": lesson[:200]})
                    wm.add_text(_nudge("vetoed", tool=repr(name), count=failed_dups))
                    continue
                if failed_dups == 1:
                    # Soft nudge (→ ReBind): warn, but allow this one retry (transient failures recover).
                    wm.add_text(_nudge("retry-once", tool=repr(name)))
                elif count_successful_duplicates(name, args, avec, wm.cards, recurrence_threshold) >= 1:
                    # ADR-0041 D4 (symmetric guard): this exact action ALREADY
                    # SUCCEEDED this run. Re-running it is an idempotent no-op spin
                    # (e.g. writing the same file with the same content turn after
                    # turn). Block re-execution and steer to the next step / final
                    # answer. Content-keyed: distinct successful calls (a real
                    # multi-step plan) never trip this — only the same call does.
                    sig = action_text(name, args)
                    success_dedup_counts[sig] = success_dedup_counts.get(sig, 0) + 1
                    if success_dedup_counts[sig] > recurrence_veto_depth:
                        # The model keeps re-proposing an action that already
                        # succeeded and won't advance. Finish gracefully with what we
                        # have — the work IS done (unlike the failure ladder, this is
                        # a success, not an error) — rather than spin (the dedup path
                        # skips the tool-round cap, so without this it could loop).
                        logger.info("react_success_dedup_complete",
                                    extra={"round": round_no, "tool": name})
                        return AgentResult.from_text(_best_effort(wm), type=result_type or "text")
                    logger.info("react_success_dedup", extra={"round": round_no, "tool": name})
                    wm.add_text(_nudge("already-done", tool=repr(name)))
                    continue

            tool_rounds += 1
            if tool_rounds > max_tool_rounds:
                # Graceful first, fatal second — the same shape the memory budget
                # already uses, and for the same reason.
                #
                # Raising immediately threw away a whole run's gathered evidence
                # because the model had not been TOLD it was nearly out. It cannot
                # ration a budget it cannot see: a research agent will keep
                # searching until something stops it, and what stopped it was an
                # exception the caller could only report as a failure.
                #
                # One explicit chance to conclude turns that into an answer built
                # from what it actually found. If it spends that chance on another
                # tool call, the runaway guard still fires.
                #
                # `continue` is safe with a tool call outstanding: the top of the
                # loop closes an unanswered tool_call_id before the next request,
                # which is what keeps the conversation valid for the provider.
                if tool_exhausted:
                    raise ReActLoopError(f"tool-round budget ({max_tool_rounds}) exceeded")
                tool_exhausted = True
                logger.info("react_tool_budget_exhausted",
                            extra={"round": round_no, "budget": max_tool_rounds})
                wm.add_text(_nudge("tool-budget", max_tool_rounds=max_tool_rounds))
                continue
            # Local @tool → run in-process. Otherwise it is a kernel-owned system
            # tool (ADR-0039): the agent marshalled the args here; the kernel
            # authorizes and executes it in a confined process via ExecuteTool.
            if name in agent.tools:
                out = agent.tools.call(name, **args)
            else:
                # No client-side gRPC deadline on the tool call (timeout_ms=0),
                # mirroring the generate() rationale above: passing the whole-task
                # step budget here strangled cold-start tool spawns (interpreter
                # start + module import, esp. first call on Windows) with an opaque
                # DEADLINE_EXCEEDED *before* the kernel's own ProcessHandler timeout
                # (DefaultTimeout, which also kills the child) could apply. The
                # kernel side is the authority on tool runtime bounds, not the agent.
                # Pass the task's session token so the kernel can recognize a
                # sandboxed evaluation (interview) session and auto-approve
                # dangerous tools within it (ADR-0037/0039).
                out = _call_system_tool(agent, name, args, task.session_token_id, task.task_id)
            logger.info(
                "react_tool_call",
                # NOTE: "args" is a RESERVED LogRecord attribute — using it as an
                # extra= key raises KeyError("Attempt to overwrite 'args'"). Use tool_args.
                extra={"round": round_no, "tool": name, "tool_args": args},
            )
            # ADR-0041 D1/D3: record an invocation card (tool + args + status +
            # summary). A heavy system-tool result the kernel offloaded keeps
            # {summary + cid}; otherwise it is summarized inline (heuristic by
            # default; the agent's opt-in `tool_summarizer` for heavy results).
            # The ToolCard is what the recurrence gate (wm.cards) reads.
            card = ToolCard.from_result(
                name, args, out, summarizer=getattr(agent, "tool_summarizer", None),
            )
            wm.add_tool_card(card)
            # ADR-0097 D8: close the tool-calling loop. The provider correlates this
            # against the call it made, under ITS id — a synthesized one is rejected.
            # Without this turn the model never learns the call happened, which is what
            # made the first cut re-explore every round and never write a file.
            #
            # The result is sent as JSON where possible; a provider expects a string,
            # and a Python repr would be a shape the model has to guess at.
            if conversation and pending_tool_call_id:
                try:
                    tool_content = json.dumps(out, default=str)
                except (TypeError, ValueError):
                    tool_content = str(out)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": pending_tool_call_id,
                    "content": tool_content,
                })
                pending_tool_call_id = ""
                pending_extra_call_ids = _answer_unexecuted(conversation, pending_extra_call_ids)
                if isinstance(wm, _ConversationMirror):
                    wm.release()
            # v2: also emit a typed Step so the trajectory has the rich XML
            # (call site as named children, offloaded result with summary, etc.).
            #
            # Issue seen in the field: the model re-issued the same read 4 times
            # because ``card.summary`` was a 400-char preview of a 3475-char file.
            # The model never saw the full content, so it kept reading. Fix: pass
            # the FULL body to the v2 step. The renderer decides whether to
            # show it inline (small enough) or offload it via substrate (too
            # large) — see the v2 step renderer for the threshold.
            full_body_str = out if isinstance(out, str) else json.dumps(out, default=str)
            wm.add_step(
                kind="tool_call",
                tool=name,
                args=args,
                status=card.status,
                summary=card.summary,            # the one-line gist (for the offloaded branch)
                body=full_body_str,              # the full body (for the inline branch)
                cid=card.cid or "",               # the kernel-offloaded cid (if any)
                chars=len(full_body_str),
            )
            continue

        if kind == "find_tools":
            # Replies go through _control_reply so a NATIVE find_tools call is
            # answered in its own tool turn, not "call was not executed" (ADR-0139 D8).
            # An agent that opted out of DISCOVERY must not be able to pull tools
            # back in through it. Keyed on find_tools_enabled rather than on the
            # seeded menu: an agent may legitimately want no preloaded menu and
            # still want to ask for a capability by name.
            if not find_tools_enabled:
                _control_reply("<note>Tools are not available to you. Either answer from what you "
                               "have, or use delegate_to_planner to hand the task to the planner.</note>")
                continue
            # ADR-0044 pull: the agent describes a capability need; the kernel
            # returns the matching tools, which we merge into the menu so the next
            # turn can tool_call one. The analogue of memory_query, for tools.
            need = (action.get("need") or action.get("query") or "").strip()
            if not need:
                _control_reply("<note>find_tools needs a non-empty capability 'need' "
                               "(verb-first, e.g. 'search the web for a person').</note>")
                continue
            find_tools_calls += 1
            if find_tools_calls > 3:
                _control_reply("<note>tool-discovery budget exhausted; use a tool already "
                               "in your menu, or answer/yield.</note>")
                continue
            found = _list_system_tools(agent, query=need, full=native_tools)
            seen = {t.get("name") for t in system_tools}
            added = [t for t in found if t.get("name") not in seen]
            system_tools = system_tools + added
            # Under native tool-calling the menu the model can ACT on is tool_defs, not
            # the prose list — so a discovery that does not rebuild it teaches the model
            # names it still cannot call. Rebuild both the definitions and the
            # sanitized→original map together; they must stay in step or a discovered
            # tool dispatches to a name the registry has never heard of.
            if native_tools:
                tool_defs, tool_name_map = build_tool_definitions(
                    agent, system_tools, include_skills=bool(system_skills))
            logger.info("react_find_tools", extra={"round": round_no, "need": need,
                                                   "added": [t.get("name") for t in added],
                                                   "offered": len(tool_defs)})
            if added:
                # One protocol per prompt (ADR-0139): natively the tool is just called.
                _control_reply("<note>Found tools for {!r}: {}. {}</note>".format(
                    need, ", ".join(t.get("name") for t in added),
                    "Call one." if native_tools else "Call one with tool_call."))
            else:
                _control_reply(f"<note>No tool matched {need!r}. Proceed without it — "
                               "answer from what you have, or yield_subgoal.</note>")
            continue

        if kind == "find_skills":
            # ADR-0046 pull: discover authored procedures (skills) for a need not in
            # the menu. The analogue of find_tools, for skills. Returns Tier-1 short
            # forms, merged into the menu; then the agent loads one with use_skill.
            # Replies go through _control_reply so a NATIVE find_skills call is
            # answered in its own tool turn (ADR-0139 D8).
            need = (action.get("need") or action.get("query") or "").strip()
            if not need:
                _control_reply("<note>find_skills needs a non-empty 'need' describing the "
                               "procedure/capability you want.</note>")
                continue
            find_skills_calls += 1
            if find_skills_calls > 3:
                _control_reply("<note>skill-discovery budget exhausted; use a listed skill, "
                               "or proceed/answer.</note>")
                continue
            found = _list_system_skills(agent, query=need)
            seen = {sk.get("name") for sk in system_skills}
            added = [sk for sk in found if sk.get("name") not in seen]
            system_skills = system_skills + added
            logger.info("react_find_skills", extra={"round": round_no, "need": need,
                                                    "added": [sk.get("name") for sk in added]})
            if added:
                _control_reply("<note>Found skills for {!r}: {}. Load one with use_skill.</note>".format(
                    need, "; ".join("{} ({})".format(sk.get("name"), (sk.get("description") or "").strip())
                                    for sk in added)))
            else:
                _control_reply(f"<note>No skill matched {need!r}. Proceed without it.</note>")
            continue

        if kind == "use_skill":
            # ADR-0046 D3: load a skill — inject its instructions into working memory
            # and (for system skills) activate its bundled tool grants for the run.
            # An agent-local skill loads purely locally (its grants are already within
            # the agent's envelope); a system skill is fetched Tier-2 and the kernel
            # confers + activates its grants run-scoped during the fetch.
            #
            # ADR-0139 D3 (Phase 0): the body is PINNED in full via wm.add_skill, not
            # put through add_text, whose R7 offloader cut any body over 1000 chars to
            # a 240-char head and whose entries the 10-entry cap later evicted. The
            # bound is the run's skill budget instead: the load cap (5) plus a total
            # body budget; an overflow is refused with a note, never truncated.
            # Every reply goes through _control_reply, so a NATIVE use_skill call is
            # answered in its own tool turn with the full body (ADR-0139 D8).
            name = (action.get("skill") or action.get("name") or "").strip()
            if not name:
                _control_reply("<note>use_skill needs a 'skill' name from the skill menu.</note>")
                continue
            if name in loaded_skills:
                _control_reply(f"<note>Skill {name!r} is already loaded — follow its instructions.</note>")
                continue
            skill_loads += 1
            if skill_loads > _DEFAULT_MAX_SKILL_LOADS:
                _control_reply(f"<note>skill-load budget exhausted ({_DEFAULT_MAX_SKILL_LOADS} loads "
                               "per run); act on a loaded skill or answer.</note>")
                continue
            sk = _load_skill(agent, name, task.session_token_id)
            logger.info("react_use_skill", extra={"round": round_no, "skill": name,
                                                  "loaded": sk is not None})
            if not sk:
                _control_reply(f"<note>Skill {name!r} is not available to you. "
                               "Use a listed skill, or proceed.</note>")
                continue
            body = (sk.get("instructions") or "").strip()
            if skill_chars + len(body) > _DEFAULT_MAX_SKILL_CHARS:
                # Refuse whole, never truncate: a partial procedure is a different one.
                logger.info("react_use_skill_over_budget",
                            extra={"round": round_no, "skill": name, "chars": len(body),
                                   "used": skill_chars, "budget": _DEFAULT_MAX_SKILL_CHARS})
                _control_reply(
                    f"<note>Skill {name!r} was NOT loaded: its instructions ({len(body)} chars) "
                    f"do not fit the remaining skill budget ({_DEFAULT_MAX_SKILL_CHARS - skill_chars} "
                    f"of {_DEFAULT_MAX_SKILL_CHARS} chars left). Continue with the skills already "
                    "loaded, or proceed without it.</note>")
                continue
            skill_chars += len(body)
            loaded_skills.add(name)
            grants = sk.get("tool_grants") or []
            # ADR-0046 D6: the load conferred these grants (kernel-side, run-scoped), so
            # they must also become CALLABLE: added to the native tool definitions and
            # to the prose menu, or the note below promises tools the model cannot name.
            # Append-only, so the cached prefix of tool_defs is unchanged.
            new_tools = _skill_granted_tools(
                agent, name, grants, system_tools, full=native_tools,
                kernel_confirmed=not sk.get("_local"),
                session_token_id=task.session_token_id)
            if new_tools:
                skill_tools = skill_tools + new_tools
                system_tools = _merge_tools(system_tools, new_tools)
                if native_tools:
                    tool_defs, tool_name_map = build_tool_definitions(
                        agent, system_tools, include_discovery=find_tools_enabled,
                        include_skills=bool(system_skills))
                else:
                    action_protocol = _compose_action_protocol(
                        agent, output_schema, system_tools, system_skills,
                        allow_yield_subgoal=allow_yield_subgoal,
                        native_tools=False, allow_memory=allow_memory)
                logger.info("react_skill_tools_added",
                            extra={"round": round_no, "skill": name,
                                   "added": [t.get("name") for t in new_tools]})
            grant_note = (" Its tools ({}) are available for this task.".format(", ".join(grants))
                          if grants else "")
            note = f"Loaded skill {name!r}; follow its instructions for the rest of this task.{grant_note}"
            if _answer_native_turn(
                    "<skill name={!r}>\n{}\n</skill>\n<note>{}</note>".format(name, body, note)):
                # Native: the tool turn carries the body and the conversation keeps it
                # for the run. Pin it in the underlying memory too, so a later latch to
                # the text path still renders it.
                (wm._wm if isinstance(wm, _ConversationMirror) else wm).add_skill(name, body, note)
            else:
                wm.add_skill(name, body, note)
            continue

        if kind == "describe_tool":
            # ADR-0045 Tier-2 fetch: the menu lists tools tersely (summary + arg
            # names); before calling one, the agent fetches its full description +
            # arg schema so it can pass correct arguments. Grant-gated kernel-side:
            # an ungranted/unknown name comes back absent (no existence leak).
            name = (action.get("tool") or action.get("name") or "").strip()
            if not name:
                wm.add_text("<note>describe_tool needs a 'tool' name to fetch its full spec.</note>")
                continue
            describe_calls += 1
            if describe_calls > _DEFAULT_MAX_DESCRIBE_CALLS:
                wm.add_text("<note>describe budget exhausted; call a tool you've already "
                            "described, or answer/yield.</note>")
                continue
            spec = _describe_tool(agent, name)
            logger.info("react_describe_tool", extra={"round": round_no, "tool": name,
                                                      "found": spec is not None})
            if spec:
                wm.add_text(
                    "<tool_spec name={!r}>\n  <description>{}</description>\n"
                    "  <args_schema>{}</args_schema>\n</tool_spec>".format(
                        spec.get("name"), (spec.get("description") or "").strip(),
                        spec.get("schema_json") or "{}"))
            else:
                wm.add_text(f"<note>Tool {name!r} is not available to you. Use a tool in "
                            "your menu, or answer/yield.</note>")
            continue

        if kind == "resolve_cid":
            # ADR-0048 residual: the agent asks the kernel to fetch a
            # previously-offloaded body (a workspace seed, a tool result, a
            # recalled fact). Two modes:
            #   - "offload" (default): body is re-offloaded and a NEW cid is
            #     returned. Net cost: 1 round-trip + 0 prompt tokens.
            #   - "inline": body is inlined into the trajectory so the model
            #     can read it. Net cost: 1 round-trip + N prompt tokens.
            #     Per-run budget enforced here so a runaway model can't
            #     fill the context.
            cid = (action.get("cid") or "").strip()
            as_mode = (action.get("as") or "offload").strip()
            if not cid:
                wm.add_step(
                    kind="note", n=round_no,
                    body="resolve_cid needs a 'cid' from an offloaded block (or a recalled fact with [full content cid:…]).",
                )
                continue
            if as_mode not in ("offload", "inline"):
                wm.add_step(
                    kind="note", n=round_no,
                    body=f"resolve_cid: 'as' must be 'offload' or 'inline', got {as_mode!r}.",
                )
                continue
            # Fetch the body via the substrate.
            body: Optional[str] = None
            sub = getattr(agent, "substrate", None)
            if sub is not None and hasattr(sub, "get_context_node"):
                try:
                    node = sub.get_context_node(cid, session_token_id=task.session_token_id)
                except Exception as exc:  # noqa: BLE001 — degrade, never crash
                    logger.warning("react_resolve_cid_fetch_failed",
                                   extra={"cid": cid, "err": str(exc)})
                    node = None
                if node is not None:
                    data = getattr(node, "data", None)
                    if data is not None:
                        body = (data.decode("utf-8", "replace")
                                if isinstance(data, (bytes, bytearray)) else str(data))
            if body is None:
                wm.add_step(
                    kind="note", n=round_no,
                    body=f"resolve_cid: cid {cid!r} not found in ContentStore (or fetch failed).",
                )
                continue
            if as_mode == "inline":
                # Per-run budget (mitigation for runaway context fill).
                if inline_resolves >= max_inline_resolves:
                    wm.add_step(
                        kind="note", n=round_no,
                        body=f"resolve_cid: inline budget exhausted ({max_inline_resolves}). "
                             f"Use as='offload' instead — body stays out of your context; the next "
                             f"tool_call can use the new cid.",
                    )
                    continue
                inline_resolves += 1
                wm.add_step(
                    kind="resolve_cid", cid=cid, resolved_from=cid,
                    mode="inline", body=body,
                    summary=f"{len(body)} chars inlined; full body in your context (inline resolve {inline_resolves}/{max_inline_resolves}).",
                )
                logger.info("react_resolve_cid_inline",
                            extra={"round": round_no, "cid": cid, "chars": len(body),
                                   "inline_count": inline_resolves, "max": max_inline_resolves})
                continue
            # as_mode == "offload": re-offload via the substrate so the body
            # gets a fresh cid the next tool_call can use. Fall back to inline
            # if the substrate has no put_context_node.
            new_cid: Optional[str] = None
            if sub is not None and hasattr(sub, "put_context_node"):
                try:
                    new_cid = sub.put_context_node(body, session_token_id=task.session_token_id)
                except Exception as exc:  # noqa: BLE001 — degrade, never crash
                    logger.warning("react_resolve_cid_put_failed",
                                   extra={"cid": cid, "err": str(exc)})
                    new_cid = None
            if not new_cid:
                # No write path — fall back to inline. This is the safe default;
                # the model can still get the body, just with the inline cost.
                wm.add_step(
                    kind="resolve_cid", cid=cid, resolved_from=cid,
                    mode="inline", body=body,
                    summary=f"{len(body)} chars; substrate has no put_context_node — "
                             f"inlined as fallback (resolve {inline_resolves + 1}/{max_inline_resolves}).",
                )
                inline_resolves += 1
                logger.info("react_resolve_cid_inline_fallback",
                            extra={"round": round_no, "cid": cid, "chars": len(body)})
                continue
            wm.add_step(
                kind="resolve_cid", cid=new_cid, resolved_from=cid,
                mode="offload", chars=len(body),
                summary=f"re-offloaded as new cid ({len(body)} chars; body is NOT in your context).",
            )
            logger.info("react_resolve_cid_offload",
                        extra={"round": round_no, "from_cid": cid, "to_cid": new_cid,
                               "chars": len(body)})
            continue

        if kind == "yield_subgoal":
            # ADR-0041 D5: the agent delegates an independent sub-task to the kernel
            # (Global RP) rather than scheduling it in-process. We just return the
            # yield — the YieldCoordinator (ADR-0037 D10) binds + dispatches it. No
            # in-agent scheduler; the loop stays single-action-per-turn.
            # A subgoal is a sub-task WITHIN a plan, bound by a kernel YieldCoordinator
            # (ADR-0037 D10) — only meaningful for an agent already running inside a
            # coordinated plan. An agent that opted out (allow_yield_subgoal=False, e.g.
            # the chat front desk, which is dispatched directly and has no coordinator)
            # must not yield: it delegates whole work to the planner via its own tool,
            # which creates a fresh top-level plan. Steer it there instead of returning
            # a sentinel nobody binds.
            if not allow_yield_subgoal:
                wm.add_text("<note>yield_subgoal is not available to you — a subgoal only "
                            "exists inside a plan. To hand real work off, tool_call your "
                            "delegation tool (it asks the planner to create a plan), then "
                            "report its result. Otherwise answer directly.</note>")
                continue
            intent = (action.get("intent") or "").strip()
            if not intent:
                wm.add_text("<note>yield_subgoal needs a non-empty intent describing the "
                            "sub-task; provide one or answer directly.</note>")
                continue
            logger.info("react_yield_subgoal", extra={"round": round_no, "intent": intent})
            return yield_subgoal(intent, capability_hint=(action.get("capability_hint") or None))

        # INVERTED DEFAULT (2026-07-28). "Finished" must be DECLARED, not inferred
        # from a parse failure.
        #
        # Previously any output without a parseable action became a final_answer and
        # returned immediately, so a model narrating its next step ("I need to write a
        # one-line summary to the output file") or dumping its working-memory markup
        # ended the task REPORTING SUCCESS with the work undone — silent, and strictly
        # worse than the loud ReActLoopError it displaced.
        #
        # Every mature agent loop routes on a structured signal instead: Anthropic's
        # rule is "when stop_reason is NOT end_turn, treat the response as incomplete";
        # LangChain feeds a parse failure BACK to the model (handle_parsing_errors /
        # RetryWithErrorOutputParser) rather than accepting it as an answer. This is
        # that contract, bounded: re-prompt once, then accept so the loop still always
        # terminates. The real fix is native tool-calling, which removes the guess.
        # Every handled action kind `continue`d above; only final_answer (declared or
        # inferred) may reach the answer tail. Anything else is an unknown verb.
        if kind != "final_answer" and not action.get("_inferred"):
            if unknown_action_retries < _MAX_UNKNOWN_ACTION_RETRIES:
                unknown_action_retries += 1
                logger.info("react_unknown_action_reprompt",
                            extra={"round": round_no, "action_kind": str(kind)[:80],
                                   "retry": unknown_action_retries})
                wm.add_text(_nudge("unknown-action", action=repr(kind)))
                continue
            raise ReActLoopError(
                f"model kept emitting an unknown action {kind!r} "
                f"({_MAX_UNKNOWN_ACTION_RETRIES} re-prompts)")

        if action.get("_inferred") and inferred_answer_retries < _MAX_INFERRED_ANSWER_RETRIES:
            inferred_answer_retries += 1
            logger.info(
                "react_inferred_answer_reprompt",
                extra={"round": round_no, "retry": inferred_answer_retries,
                       "text_preview": str(action.get("answer", ""))[:200]},
            )
            wm.add_text(INFERRED_FINAL_ANSWER_NOTE)
            continue

        # final_answer. result_type (the agent's output contract) wins over the
        # LLM-declared type when set. An inferred answer that survived the re-prompt
        # above is accepted here — logged distinctly so it stays visible.
        if action.get("_inferred"):
            logger.warning("react_inferred_answer_accepted", extra={"round": round_no})
        answer = action.get("answer", "")
        # An LLM may return a structured (dict/list/number) answer despite the
        # string contract; serialize it rather than crashing on answer[:200] or
        # AgentResult.from_text (which calls .encode on a str).
        if not isinstance(answer, str):
            answer = json.dumps(answer, default=str)
        logger.info(
            "react_final_answer",
            extra={"round": round_no, "answer_preview": answer[:200]},
        )
        return AgentResult.from_text(answer, type=result_type or action.get("type", "text"))


# ADR-0048 D8: the per-turn output contract — the LAST (recency-anchored) section.
# It steers the model to pick an action each turn, NOT to emit a final answer; the
# final_answer body format lives in <ActionProtocol> (the final_answer action's
# description), present but no longer the closing instruction that biased premature
# termination.
_PER_TURN_OUTPUT_CONTRACT = fragment("react-turn", "contract-text").render()
# The last sentence above: T-20261104-agent-final-answers-leak-loop-narration-perfect
# (E2E E5: a final answer opened with "Perfect! I now have all the information...",
# breaking a first-line format rule). ADR-0139 D3's native final_answer tool removes
# the failure by construction; this is the interim prompt fix.

# ADR-0139 Phase 0: the native-tool-calling variant. The text contract above says
# "Output EXACTLY ONE JSON object", which under native tool-calling contradicts the
# attached tool schemas and _NATIVE_TOOL_SECTION ("there is no such action"): two
# ways to act, and the model takes the one that does nothing (react.py, the
# _JSON_TOOL_SECTION comment: 3/6 against 5/6). One protocol per prompt.
_PER_TURN_OUTPUT_CONTRACT_NATIVE = fragment("react-turn", "contract-native").render()
# The last sentence above: T-20261104-agent-final-answers-leak-loop-narration-perfect
# (E2E E5: a final answer opened with "Perfect! I now have all the information...",
# breaking a first-line format rule). ADR-0139 D3's native final_answer tool removes
# the failure by construction; this is the interim prompt fix.


def _menu_has_tool(agent, system_tools: Optional[List[Dict]], name: str) -> bool:
    """True when ``name`` is a tool this agent can call right now: one of its local
    ``@tool`` functions, or a granted system tool in the current menu (matched on the
    full name or its last path segment, since system tools are namespaced, e.g.
    ``mcp:filesystem/fast_large_write_file``)."""
    try:
        if name in agent.tools:
            return True
    except Exception:  # noqa: BLE001 — a test double without a registry
        pass
    for t in system_tools or []:
        full = t.get("name") or ""
        if full == name or full.rsplit("/", 1)[-1] == name:
            return True
    return False


def _compose_action_protocol(agent, domain_schema: str, system_tools: Optional[List[Dict]] = None,
                             system_skills: Optional[List[Dict]] = None,
                             allow_yield_subgoal: bool = True,
                             native_tools: bool = False,
                             allow_memory: bool = True) -> str:
    """The agent-loop action menu + behavioral rules (the <ActionProtocol> body),
    with the agent's domain final-answer format folded into the final_answer action
    so it is documented but not the recency-anchored closer (ADR-0048 D8)."""
    base = build_output_schema(agent, system_tools, system_skills,
                               allow_yield_subgoal=allow_yield_subgoal,
                               native_tools=native_tools,
                               allow_memory=allow_memory)
    if domain_schema and domain_schema.strip():
        base += "\n\n" + _ap("final-format", format=domain_schema.strip())
    return base


# ── ADR-0139 C1 call plumbing for the ReAct loop (wave 1d) ───────────────────

_CACHE_TTL_ENV = "CAMBRIAN_AGENT_CACHE_TTL"
_DEFAULT_RUN_CACHE_TTL = "short"


def _resolve_cache_ttl(agent, explicit: Optional[str]) -> str:
    """run_think argument > agent ``cache_ttl`` attribute > env > ``"short"``.
    An unrecognised value is logged and replaced by the default, never sent."""
    from .llm import coerce_cache_ttl
    from .errors import InvalidRequest
    for value in (explicit, getattr(agent, "cache_ttl", None), os.getenv(_CACHE_TTL_ENV)):
        if value is None or value == "":
            continue
        try:
            return coerce_cache_ttl(value)
        except InvalidRequest:
            logger.warning("react_bad_cache_ttl", extra={"value": str(value)})
    return _DEFAULT_RUN_CACHE_TTL


_KWARG_CACHE: Dict[Any, bool] = {}


def _accepts_kwarg(fn, name: str) -> bool:
    """True when ``fn`` takes ``name`` as a keyword (explicitly or via ``**kw``).
    Lets the loop keep serving a substrate written before the C1 keywords existed."""
    import inspect
    key = (getattr(fn, "__func__", fn), name)
    if key in _KWARG_CACHE:
        return _KWARG_CACHE[key]
    try:
        params = inspect.signature(fn).parameters.values()
        ok = any(p.kind is p.VAR_KEYWORD or (p.name == name and p.kind in (
            p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)) for p in params)
    except (TypeError, ValueError):
        ok = False
    _KWARG_CACHE[key] = ok
    return ok


def _log_prompt_versions(agent) -> None:
    """Debug: which prompt-library versions this run's ReAct prompt is built from."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("react_prompt_versions", extra={
            "agent_id": getattr(agent, "agent_id", ""), "prompts": prompt_versions()})


def _log_llm_call(path: str, round_no: int, system: List[Dict[str, Any]], n_messages: int) -> None:
    """Per-call request metadata, at INFO: it is the prompt-provenance record.
    ``system_sha256`` is the hash of the system text sent; it must be the same on
    every turn of a run (the cache depends on it). ``prompts`` names each
    prompt-library id, version and static hash."""
    if not logger.isEnabledFor(logging.INFO):
        return
    import hashlib
    text = "\n\n".join(b.get("text", "") for b in system)
    logger.info("react_llm_request", extra={
        "path": path,
        "round": round_no,
        "system_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if system else "",
        "system_chars": len(text),
        "messages": n_messages,
        "prompts": prompt_versions(),
    })


def _log_llm_result(path: str, res) -> None:
    usage = getattr(res, "usage", None)
    ignored = list(getattr(res, "ignored", None) or [])
    if usage is None and not ignored:
        return
    logger.info("react_llm_usage", extra={
        "path": path,
        "model": getattr(res, "model", ""),
        "ignored": ignored,
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
        "cache_write_tokens": getattr(usage, "cache_write_tokens", 0),
        "cache_write_known": getattr(usage, "cache_write_known", False),
        "estimated": getattr(usage, "estimated", False),
    })


def _generate_text_turn(agent, task, parts, *, structured: bool, cache_ttl: str,
                        max_tokens, temperature, round_no: int) -> Tuple[str, bool]:
    """One text-protocol turn. Returns ``(raw_text, structured_still_ok)``.

    Preferred: the structured call (``substrate.generate_request``) with the cached
    system tier and the volatile user turn. Degrades, latched for the run, to the
    legacy single prompt when:
      - the substrate has no ``generate_request`` (an older SDK object, a custom
        transport, a test double): it gets ``generate(token, prompt)`` as before;
      - the kernel refuses the structured shape (``InvalidRequest``);
      - the kernel reports ``"system"`` as ignored (the model saw no system text).
    A kernel that predates C1 needs no retry: the client also sends the flattened
    prompt in the deprecated field, which such a kernel reads.
    """
    from .errors import InvalidRequest
    sub = agent.substrate
    req_fn = getattr(sub, "generate_request", None) if structured else None
    if req_fn is not None:
        _log_llm_call("text", round_no, parts.system, len(parts.messages))
        try:
            res = req_fn(task.session_token_id, system=parts.system, messages=parts.messages,
                         cache_ttl=cache_ttl, max_tokens=max_tokens, temperature=temperature,
                         timeout_ms=0)
        except InvalidRequest as exc:
            logger.warning("react_structured_refused_legacy_prompt", extra={"reason": str(exc)})
            structured = False
        else:
            if "system" in (res.ignored or []):
                logger.warning("react_system_ignored_legacy_prompt",
                               extra={"path": "text", "model": res.model})
                structured = False
            else:
                _log_llm_result("text", res)
                return res.text, True
    _log_llm_call("text-legacy", round_no, [], 1)
    raw = sub.generate(
        task.session_token_id,
        parts.legacy_prompt(),
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_ms=0,
    )
    return raw, structured and req_fn is not None


def _best_effort(wm: WorkingMemory) -> str:
    """A degraded final answer synthesized from whatever the loop gathered."""
    return wm.assemble()


def _latest_failure_summary(wm: WorkingMemory, tool: str) -> str:
    """The summary of the most recent FAILED card for ``tool`` (ADR-0052) — the
    concrete failure the verbal reflection should explain. ``''`` when none found."""
    for c in reversed(wm.cards):
        if c.tool == tool and c.status in ("error", "denied"):
            return c.summary or ""
    return ""


def _request_reflection(agent, role, task, tool, args, wm, prior_reflections, temperature) -> str:
    """ADR-0052: one bounded LLM call asking *why* a repeatedly-failed action failed
    and *what* to change, returning the verbal reflection (or ``''``).

    Best-effort by construction: a missing substrate, an RPC failure, an empty
    response, or a model that ignored the instruction and emitted another action
    (rather than prose) all degrade to ``''`` — the loop then falls back to the terse
    structural veto note alone. Never raises into the loop."""
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "generate"):
        return ""
    prompt = build_reflection_prompt(
        role, task.text, tool, args, _latest_failure_summary(wm, tool), prior_reflections
    )
    try:
        raw = substrate.generate(
            task.session_token_id, prompt,
            max_tokens=DEFAULT_MAX_REFLECTION_TOKENS, temperature=temperature, timeout_ms=0,
        )
    except Exception:  # noqa: BLE001 — reflection is optional; degrade to the terse note
        return ""
    text = (raw or "").strip()
    if not text:
        return ""
    # Plain prose parses to a final_answer envelope; a real action (tool_call, etc.)
    # or a truncation means the model emitted an action, NOT a reflection — discard it
    # rather than pin a bogus lesson the next attempt would read as guidance.
    parsed = parse_action(text)
    if parsed.get("action") != "final_answer":
        return ""
    answer = parsed.get("answer")
    lesson = answer if isinstance(answer, str) else text
    return " ".join(lesson.split())


def _call_system_tool(agent, name: str, args: dict, session_token_id: str = "", task_id: str = ""):
    """Route a non-local tool call to the kernel's ExecuteTool (ADR-0039). A
    missing substrate or any RPC failure degrades to a structured error the loop
    can reason about — never an exception. ``session_token_id`` is forwarded so
    the kernel can recognize a sandboxed evaluation session and auto-approve
    dangerous tools within it (ADR-0037). ``task_id`` is the per-step correlation key
    (ADR-0049 D3) so the kernel can stamp action records for the synthesis dedup. No
    client-side gRPC deadline is set (timeout_ms=0): the kernel's ProcessHandler
    bounds and kills the child."""
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "execute_tool"):
        return {"error": f"system tool {name!r} unavailable (no substrate)", "tool": name}
    try:
        resp = substrate.execute_tool(
            name, args_json=json.dumps(args),
            session_token_id=session_token_id or "", timeout_ms=0, task_id=task_id or "",
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never crash the loop
        return {"error": f"system tool {name!r} failed: {exc}", "tool": name}
    # Interpret the raw kernel response into a result the loop can reason about.
    if resp.get("denied"):
        return {"error": f"denied: {resp.get('deny_reason', '')}", "tool": name}
    if resp.get("error"):
        return {"error": resp["error"], "tool": name}
    if resp.get("result_cid"):
        return {"cid": resp["result_cid"], "tool": name}
    rj = resp.get("result_json") or ""
    try:
        return json.loads(rj) if rj else {}
    except (ValueError, TypeError):
        return {"result": rj, "tool": name}


def _list_system_tools(agent, query: str = "", full: bool = False) -> List[Dict]:
    """Fetch the agent's granted kernel system tools for the prompt menu (ADR-0039).

    ADR-0044: when ``query`` is given, the kernel returns only the top-k
    task-relevant tools (semantic retrieval) instead of the full granted set.

    ``full`` selects the ADR-0045 disclosure tier. Tier-1 (default) is an
    arg-names-only schema — right for the PROSE menu, which renders property names and
    nothing else. Tier-2 is the real JSON Schema, which is what a native tool
    definition needs: the provider VALIDATES against it, so types and ``required`` are
    load-bearing rather than decorative. Asking for Tier-1 and handing it to a provider
    is what produced the opaque 400s (no top-level ``type``) and type-less arguments.
    Degrades to an empty list when the substrate is absent or has no tool plane —
    a cognitive agent with only local @tools, or a kernel without a registry,
    simply gets the @tool menu. Never raises into the loop."""
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "list_tools"):
        return []
    try:
        return substrate.list_tools(query=query, k=_DEFAULT_TOOL_MENU_K, full=full)
    except TypeError:
        # Substrate/test fake without the ADR-0044 query parameter — full menu.
        try:
            return substrate.list_tools()
        except Exception:  # noqa: BLE001
            return []
    except Exception:  # noqa: BLE001 — degrade to no system tools, never crash
        return []


def _describe_tool(agent, name: str) -> Optional[Dict]:
    """Fetch a tool's Tier-2 full spec (ADR-0045 describe_tool): the full
    description + full arg schema for one named tool. Returns the descriptor
    dict, or None when the tool is unavailable to the agent (ungranted/unknown)
    or there is no tool plane. Never raises into the loop."""
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "list_tools"):
        return None
    try:
        specs = substrate.list_tools(names=[name], full=True)
    except TypeError:
        return None  # substrate/test fake without the ADR-0045 names/full params
    except Exception:  # noqa: BLE001 — degrade, never crash the loop
        return None
    return specs[0] if specs else None


def _merge_tools(base: List[Dict], extra: List[Dict]) -> List[Dict]:
    """``base`` followed by the entries of ``extra`` whose names it lacks. Order is
    kept (append-only), so a native tool-definition list rebuilt from the result
    keeps its existing prefix byte-identical."""
    seen = {t.get("name") for t in base}
    out = list(base)
    for t in extra:
        if t.get("name") not in seen:
            seen.add(t.get("name"))
            out.append(t)
    return out


# The schema a skill-granted tool is offered with when the kernel confirmed the grant
# but could not describe the tool (see _skill_granted_tools). Top-level "type" is what
# a provider requires; no properties means "any arguments".
_GRANTED_TOOL_FALLBACK_SCHEMA = '{"type": "object", "properties": {}}'


def _skill_granted_tools(agent, skill: str, grants: List[str], menu: List[Dict],
                         full: bool, kernel_confirmed: bool,
                         session_token_id: str = "") -> List[Dict]:
    """The menu entries for the tools a just-loaded skill grants (ADR-0046 D6) that
    the agent's menu does not already carry.

    The kernel conferred the grants run-scoped while serving the skill (ListSkills
    -> ToolExecutor.ConferSkillGrants), and ``ExecuteTool`` honours them through the
    run overlay — but nothing put the tools in front of the model, so it was told
    they were available and could not call them. Schemas come from the existing
    disclosure path (``list_tools(names=..., full=...)``, as describe_tool uses).

    The run's lease (``session_token_id``) rides the call so the kernel also
    describes tools the skill conferred under THIS run's overlay
    (``AvailableToolsNamedFor``); an older kernel ignores it and serves standing
    grants only, so an overlay-only tool can still come back absent. For a system
    skill the returned ``tool_grants`` is exactly the post-clip set of registry-known
    tools the kernel just conferred, so such a tool is still offered, with a
    permissive schema and a description pointing at the skill. For an agent-local
    skill (``kernel_confirmed=False``) nothing vouches for an undescribed name, so it
    is not invented. Local ``@tool`` names are already in the menu and are skipped.
    Never raises."""
    have = {t.get("name") for t in menu or []}
    tools = getattr(agent, "tools", None)
    missing = [g for g in grants or []
               if g and g not in have and not (tools is not None and g in tools)]
    if not missing:
        return []
    described: Dict[str, Dict] = {}
    substrate = getattr(agent, "substrate", None)
    if substrate is not None and hasattr(substrate, "list_tools"):
        try:
            try:
                listed = substrate.list_tools(names=missing, full=full,
                                              session_token_id=session_token_id)
            except TypeError:  # a substrate (or fake) without the lease kwarg
                listed = substrate.list_tools(names=missing, full=full)
            for t in listed or []:
                if t.get("name") in missing:
                    described[t["name"]] = t
        except Exception:  # noqa: BLE001 — TypeError (old fake) or RPC failure: degrade
            pass
    out: List[Dict] = []
    for g in missing:
        if g in described:
            out.append(described[g])
        elif kernel_confirmed:
            out.append({
                "name": g,
                "description": f"Granted by skill {skill!r} for this task; its instructions "
                               "say how to call it.",
                "schema_json": _GRANTED_TOOL_FALLBACK_SCHEMA,
            })
    return out


def _list_system_skills(agent, query: str = "") -> List[Dict]:
    """Fetch the agent's scope-permitted kernel system skills (ADR-0046). Degrades
    to an empty list when the substrate has no skill plane. Never raises."""
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "list_skills"):
        return []
    try:
        return substrate.list_skills(query=query, k=_DEFAULT_TOOL_MENU_K)
    except TypeError:
        try:
            return substrate.list_skills()
        except Exception:  # noqa: BLE001
            return []
    except Exception:  # noqa: BLE001 — degrade to no system skills, never crash
        return []


# The Tier-1 one-liner cap — the SAME constant as the kernel's toolSummaryMaxChars
# (cambrian-core domain/tool_retriever.go, ADR-0045 D3).
SUMMARY_MAX_CHARS = 120


def derive_summary(name: str, description: str) -> str:
    """Port of the kernel's ``deriveSummary`` (domain/tool_retriever.go, ADR-0045 D3,
    reused for skills by ADR-0046): strip leading markdown, take the first sentence or
    line, collapse whitespace, cap at a word boundary with an ellipsis, and fall back
    to the humanized name when nothing is left. Pure, no LLM.

    It exists so an agent-local skill's menu line is derived exactly as a system
    skill's is (the kernel applies it in ``SkillDisclosure`` before serving). Held to
    the Go implementation by ``tests/test_skill_summary_parity.py`` against the fixture
    the Go test writes (internal/skill/discovery/testdata/summary_cases.json)."""
    # stripLeadingMarkup -> firstSentenceOrLine -> collapseSpaces, in the Go order.
    seg = (description or "").lstrip("#*`> \t\r\n").strip()
    if seg:
        ends = [i for i in (seg.find("\n"), seg.find("\r")) if i >= 0]
        if ends:
            seg = seg[:min(ends)].strip()
        i = seg.find(". ")
        if i >= 0:
            seg = seg[:i + 1]  # include the period
        seg = seg.strip()
    seg = " ".join(seg.split())
    if len(seg) > SUMMARY_MAX_CHARS:
        cut_s = seg[:SUMMARY_MAX_CHARS]
        j = cut_s.rfind(" ")
        if j > 0:
            cut_s = cut_s[:j]
        seg = cut_s.rstrip(" ") + "…"
    if not seg:
        n = name or ""
        k = max(n.rfind("/"), n.rfind(":"))
        if k >= 0 and k + 1 < len(n):
            n = n[k + 1:]
        return " ".join(n.replace("_", " ").replace("-", " ").split())
    return seg


def _agent_local_skills(agent) -> List[Dict]:
    """The agent's own SDK-local skills (ADR-0046 D2/D5) — shipped with the agent,
    never in the kernel index. The agent exposes them via a ``local_skills``
    attribute (list of dicts or Skill-like objects), normalized to the menu shape."""
    out = []
    for s in getattr(agent, "local_skills", None) or []:
        if isinstance(s, dict):
            name, get = s.get("name"), s.get
        else:  # a Skill-like object
            name, get = getattr(s, "name", ""), lambda k, _s=s: getattr(_s, k, None)
        if not name:
            continue
        out.append({
            "name": name,
            # One deriver for both loci: a local skill's menu line is capped and cut
            # exactly as the kernel cuts a system skill's (SkillDisclosure), instead of
            # rendering an author's whole multi-paragraph description into the menu.
            "description": derive_summary(name, get("description") or ""),
            "instructions": get("instructions") or "",
            "tool_grants": list(get("tool_grants") or []),
        })
    return out


def _assemble_skill_menu(agent, query: str = "") -> List[Dict]:
    """Build the loadable skill menu (ADR-0046 D5): agent-local skills are always
    listed FIRST (structural prioritization), then task-relevant system skills,
    with same-name system skills SHADOWED by agent-local ones."""
    local = _agent_local_skills(agent)
    local_names = {s["name"] for s in local}
    system = [s for s in _list_system_skills(agent, query=query) if s.get("name") not in local_names]
    return local + system


def _load_skill(agent, name: str, session_token_id: str = "") -> Optional[Dict]:
    """Load a skill's Tier-2 form for use_skill (ADR-0046 D3). An agent-local skill
    loads purely LOCALLY — its instructions are in hand and its grants are already
    within the agent's envelope (no kernel activation). A system skill is fetched
    Tier-2 from the kernel, which confers + activates its bundled grants run-scoped
    during that fetch (keyed by the session token). Returns the skill dict, or None
    when unavailable. Never raises."""
    for s in _agent_local_skills(agent):
        if s["name"] == name:
            # Marked so the caller knows no kernel confirmed (or conferred) its grants.
            return dict(s, _local=True)
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "list_skills"):
        return None
    try:
        specs = substrate.list_skills(names=[name], full=True, session_token_id=session_token_id)
    except TypeError:
        try:
            specs = substrate.list_skills(names=[name], full=True)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001 — degrade, never crash the loop
        return None
    return specs[0] if specs else None


def _make_embed_fn(agent):
    """Return a ``text -> vector`` embedder backed by ``substrate.embed`` (ADR-0041
    D2), or ``None`` when the substrate has no Embed plane — in which case bounded
    assembly degrades to recency ranking. Never raises into the loop."""
    substrate = getattr(agent, "substrate", None)
    if substrate is None or not hasattr(substrate, "embed"):
        return None

    def fn(text):
        try:
            v = substrate.embed(text)
            return v or None
        except Exception:  # noqa: BLE001 — degrade to recency, never crash
            return None

    return fn


def _schema_properties(schema_json):
    """Extract the ``properties`` map from a tool's JSON-Schema string for the menu.

    Tolerates an empty/invalid schema (returns ``{}``) so a malformed manifest
    degrades to a name-only menu entry rather than breaking prompt assembly."""
    if not schema_json:
        return {}
    try:
        schema = json.loads(schema_json)
    except (ValueError, TypeError):
        return {}
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return props
    return {}


def _safe_recall(agent, query: str, session_token_id: str = ""):
    """Call the agent's memory.recall if present; degrade to empty on any failure.

    ``session_token_id`` (the per-step BudgetLease) is forwarded so recall carries the
    lease header, the kernel can resolve it to this run's session, and the same-session
    step-record filter (ADR-0048 D1) can actually fire — without it D1 no-ops and the
    agent's own step output is recalled back into the same run."""
    memory = getattr(agent, "memory", None)
    if memory is None:
        return []
    try:
        return memory.recall(query, session_token_id=session_token_id)
    except TypeError:
        # A memory client that predates the session_token_id param — degrade rather
        # than break (the filter just stays off for that client).
        return memory.recall(query)
    except Exception:
        return []


def _content_cid(r) -> str:
    """The full-body CID a recalled fact carries (ADR-0048 #1), parsed from its
    metadata. Recall returns the fact's one-line SUMMARY as the text and the full
    content behind this cid — so a large LTM fact no longer ships its whole body into
    context. Returns '' when the fact has no offloaded body."""
    meta = r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", None)
    if isinstance(meta, str) and meta:
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return ""
    if isinstance(meta, dict):
        return str(meta.get("content_cid", "") or "")
    return ""


def _format_memory(results) -> str:
    """Legacy v1 memory format — used by tests and by code that pre-dates the v2
    typed-Step API. Returns the body that the old ``<memory>`` block used to carry
    (a pipe-separated list of summary[+cid])."""
    if not results:
        return "(no results)"
    out = []
    for r in results:
        text = r.get("text", r) if isinstance(r, dict) else getattr(r, "text", r)
        cid = _content_cid(r)
        # Summary is the gist; the cid points at the full body (resolve to read it).
        out.append(f"{text} [full content cid:{cid}]" if cid else str(text))
    return " | ".join(out)


# v2 typed-Step renderer for memory_query — surfaces the query as an XML attribute
# (issue #1 in the trajectory critique: "memory queries don't show what was
# queried"). Used by the loop via WorkingMemory.add_step(kind="memory_query", ...).
def _render_memory_step(query: str, status: str, results) -> str:
    """Build a typed Step for a memory_query round (v2 trajectory).

    The query is surfaced as an XML attribute so the next round of the loop can
    see WHAT WAS ASKED, not just what was returned.
    """
    return (
        f'<step type="memory_query" query="{_esc(query)}" status="{_esc(status)}">\n'
        + _render_memory_children(results)
        + "\n</step>"
    )


def _render_memory_children(results) -> str:
    """Render a memory_query's results as ``<memory>`` children with provenance +
    freshness attributes (ADR-0048 A1: source/author/session/written/age/freshness/
    content_cid). Attribute assembly is shared with the WorkingMemory renderer via
    :func:`memory_provenance_attrs` so every recall surface tags facts identically."""
    if not results:
        return "  <note>no relevant memory found</note>"
    items = []
    for r in results:
        text = r.get("text", r) if isinstance(r, dict) else getattr(r, "text", r)
        items.append(f"  <memory {memory_provenance_attrs(r)}>{_esc(text)}</memory>")
    return "\n".join(items)


# ADR-0048 #1: the recall block carried into the prompt. When the kernel's
# relevance floor drops everything (an unrelated query), results is EMPTY — and the
# agent must KNOW that, not be handed a hit that quietly isn't there. An explicit
# "no relevant memory" block is the signal the ActionProtocol already licenses
# ("if memory returns nothing relevant, you may answer from your own knowledge — but
# say so"). A silent or bland block is exactly how a prior task's junk used to pose
# as grounding.
#
# v2 — this function is now a thin wrapper around ``_render_memory_step``. The old
# one-arg signature ``_render_recall(results)`` is preserved for backward compat
# (some callers in the loop's seed-recall path still use it). New code should
# prefer ``_render_memory_step(query, status, results)`` directly.
def _render_recall(query_or_results, results=None) -> str:
    """Render a memory_query block for the prompt.

    Backward-compat form: ``_render_recall(results)`` — the v1 call shape, used
    in tests and in the seed-recall path. Renders the old ``<memory>`` block.

    v2 form: ``_render_recall(query, results)`` — the new call shape, used by
    the loop when an explicit memory_query action is emitted. Renders a typed
    ``<step type="memory_query" query="..." status="...">`` block.
    """
    if results is None:
        # v1 call: _render_recall(results)
        results = query_or_results
        return _render_recall_v1(results)
    # v2 call: _render_recall(query, results)
    query = query_or_results
    status = "ok" if results else "empty"
    return _render_memory_step(query, status, results)


def _render_recall_v1(results) -> str:
    """The legacy v1 ``<memory>`` block — kept for the seed-recall path and for
    backward compat with the test suite. The new typed memory_query Step is the
    preferred format for the per-round memory_query action."""
    if not results:
        return (
            "<memory status='empty'>no relevant memory found — answer from your own "
            "knowledge and SAY you are doing so (the knowledge base had no grounded "
            "facts for this query)</memory>"
        )
    return f"<memory>{_format_memory(results)}</memory>"
