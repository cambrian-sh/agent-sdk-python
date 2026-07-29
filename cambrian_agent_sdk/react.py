"""The think() ReAct loop (ADR-0036 issue 0036-04).

A reason/retrieve/act loop with three branches:

- **memory-query** — capped re-queries with *graceful degradation*: once the budget
  is spent the loop stops retrieving and asks the LLM to answer best-effort; it never
  crashes at the cap;
- **tool-call** — capped rounds; exceeding the cap raises :class:`ReActLoopError`,
  a *typed* error the default ``run()`` catches and turns into ``type="error"``;
- **final-answer** — parsed and returned as an :class:`AgentResult`.

The prompt is assembled with the canonical PROMPTREQ 4-section builder
(:func:`build_prompt`), with the agent's ``@tool`` registry injected into the
``<OutputSchema>`` so the LLM only ever sees the closed tool menu.

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

from .helpers import _esc, build_prompt
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
# ADR-0052: how many verbal reflections the loop will extract per run. Each costs one
# small LLM call on a hard-veto; bounded so a thrashing run cannot reflect endlessly.
_DEFAULT_MAX_REFLECTIONS = 3
# resolve_cid (ADR-0048 residual): how many INLINE resolves per run. Inline
# resolves put the full body in the prompt, so they cost N tokens. Offload
# resolves are not capped (they just create a new cid; no token cost). The cap
# prevents a runaway model from filling the context with full bodies.
_DEFAULT_MAX_INLINE_RESOLVES = 3


# The Tool-actions half of the menu, in its two encodings.
#
# Under native tool-calling the JSON tool_call action must NOT be advertised. Telling a
# model to emit `{"action":"tool_call",...}` while ALSO attaching real tool schemas
# gives it two ways to do one thing, and it takes the one that does nothing. Measured
# 2026-07-28: with both present the suite scored 3/6 against 5/6 on the pure text path,
# and every failure was a model describing the work instead of performing it.
_JSON_TOOL_SECTION = (
    "## Tool actions\n"
    '- `{"action": "tool_call", "tool": "<name>", "args": {<json>}}` — invoke a granted tool.\n'
    '  Example: `{"action": "tool_call", "tool": "web_search", "args": {"query": "Paris population 2024"}}`\n'
    "- **cid handoff (ADR-0048 #1)**: for an arg value you already have a cid for\n"
    "  (a recalled fact shows `[full content cid:…]`, OR a workspace block / tool\n"
    '  result shows `offloaded_cid="…"` in the trajectory), pass `{"$cid": "<cid>"}`\n'
    "  instead of pasting the whole content — the kernel resolves it. The tool\n"
    "  sees the full body; you don't have to re-emit it. This is how you write\n"
    "  large offloaded workspace content to a file in a single tool_call.\n"
    '- `{"action": "find_tools", "need": "<capability>"}` — discover more tools (verb-first).\n'
    '  Example: `{"action": "find_tools", "need": "search the web for a person"}`\n'
    '- `{"action": "describe_tool", "tool": "<name>"}` — fetch the FULL arg schema.\n'
    '  Example: `{"action": "describe_tool", "tool": "mcp:pdf-reader/read_pdf"}`\n'
    "\n"
)

_NATIVE_TOOL_SECTION = """## Tools
- Your tools are attached to this request directly. CALL them — do not describe
  calling them, and do not emit a JSON tool_call action; there is no such action.
- Use an action from the menu below ONLY when you are not calling a tool.
- Describing what you are about to do is not doing it. If the task needs a tool,
  call it now; if the task is finished, say so as a final answer.

"""


def build_output_schema(
    agent, system_tools: Optional[List[Dict]] = None,
    system_skills: Optional[List[Dict]] = None,
    allow_yield_subgoal: bool = True,
    native_tools: bool = False,
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
    """
    tool_lines = []
    for spec in agent.tools.specs():
        tool_lines.append(f'  - {spec.name}: args schema = {json.dumps(spec.schema["properties"])}')
    for t in system_tools or []:
        desc = (t.get("description") or "").strip()
        props = _schema_properties(t.get("schema_json"))
        suffix = f" args schema = {json.dumps(props)}" if props else ""
        tool_lines.append(f'  - {t.get("name")} (system tool): {desc}{suffix}')
    tools_block = "\n".join(tool_lines) if tool_lines else "  (no tools registered)"
    # ADR-0046: the [skills] menu section — authored procedures the agent may load.
    skill_lines = [f'  - {sk.get("name")}: {(sk.get("description") or "").strip()}'
                   for sk in system_skills or []]
    skills_block = "\n".join(skill_lines) if skill_lines else "  (no skills available)"
    # yield_subgoal is only meaningful inside a coordinated plan; an opted-out agent
    # (e.g. the chat front desk) delegates whole work via a tool instead, so we omit
    # the action from its menu entirely rather than advertising a dead option.
    yield_line = (
        '- `{"action": "yield_subgoal", "intent": "<task>", "capability_hint": "<opt>"}` — delegate.\n'
        '  Example: `{"action": "yield_subgoal", "intent": "summarise the risk factors section", "capability_hint": "summarisation"}`\n'
        if allow_yield_subgoal else ""
    )
    return (
        'You emit ONE JSON action per turn — {"action": "<name>", ...} — and are '
        "called again with each result. Prefer gathering evidence over guessing.\n"
        "\n"
        "## Memory actions\n"
        '- `{"action": "memory_query", "query": "<text>"}` — retrieve from org long-term memory (knowledge base). Do this FIRST; ground claims in retrieved facts before answering from your own training.\n'
        '  Example: `{"action": "memory_query", "query": "Q4 2024 revenue"}`\n'
        "\n"
        + (_NATIVE_TOOL_SECTION if native_tools else _JSON_TOOL_SECTION) +
        "## Skill actions\n"
        '- `{"action": "use_skill", "skill": "<name>"}` — load a skill\'s steps + tools.\n'
        '  Example: `{"action": "use_skill", "skill": "codebase-investigation"}`\n'
        '- `{"action": "find_skills", "need": "<capability>"}` — discover authored procedures.\n'
        '  Example: `{"action": "find_skills", "need": "investigate a Python codebase"}`\n'
        "\n"
        "## Answer / delegation\n"
        '- `{"action": "final_answer", "answer": "<text>", "type": "text"}` — emit the answer.\n'
        '  Example: `{"action": "final_answer", "answer": "Q4 2024 revenue was $4.2B (verified via PDF p.5).", "type": "text"}`\n'
        f"{yield_line}"
        "\n"
        "## Resolution actions (ADR-0048 residual)\n"
        "- `{\"action\": \"resolve_cid\", \"cid\": \"<cid>\", \"as\": \"offload\"|\"inline\"}` — fetch a previously-offloaded body for the next step.\n"
        "  - `as: \"offload\"` (default): body is re-offloaded and a NEW cid is returned; the body is NOT in your context, but the next tool_call can use the new cid. Use this for chained operations: resolve_cid → tool_call.\n"
        "  - `as: \"inline\"`: body is inlined so you can read it. Use ONLY when you need to reason about the body (quote, summarize, answer a question). Per-run cap: 3 inline resolves.\n"
        "  Default: prefer passing the cid as `{\"$cid\": \"<cid>\"}` to a tool_call (the kernel resolves it). resolve_cid is the escape hatch.\n"
        '  Example: `{"action": "resolve_cid", "cid": "2e0093...", "as": "offload"}`\n'
        "\n"
        "## Rules\n"
        "- Ground first: issue at least one memory_query before answering, unless the "
        "task is trivial.\n"
        "- Do trivial math / multi-step reasoning yourself in final_answer; don't "
        "tool_call each step.\n"
        "- Stale-failure rule: a recalled PAST tool failure ('unavailable', timed out, "
        "'not found') is history under conditions that may no longer hold — NOT proof it "
        "fails now. Attempt it; judge by the CURRENT result.\n"
        "- Capability-gap rule: **Tools** is a task-relevant SUBSET, not everything you "
        "can use. Missing a capability means find_tools (and find_skills for a "
        "procedure) — never conclude a task is impossible or a tool 'missing' before "
        "trying find_tools.\n"
        "\n"
        + ("## Tools\n  (attached directly to this request - call them)\n"
           if native_tools else
           "## Tools (tool_call may name only these)\n" + tools_block + "\n")
        + "\n"
        + "## Skills (use_skill may name only these)\n"
        + skills_block
    )


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
INFERRED_FINAL_ANSWER_NOTE = (
    "<note>That response was not a valid action, so nothing was executed. If the task "
    "still needs work, emit the next action. If the task is genuinely complete, emit "
    '{"action": "final_answer", "answer": "..."} explicitly. Do not describe what you '
    "are about to do — either do it, or declare the answer.</note>"
)


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

    def add_step(self, kind: str, pinned: bool = False, **fields):
        """Memory-query steps carry retrieved facts the model must see; tool_call steps
        are already represented by their tool turn."""
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


def build_tool_definitions(agent, system_tools) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Render the agent's tool menu as native tool definitions (ADR-0097 Phase B).

    Returns ``(definitions, name_map)`` where ``name_map`` is sanitized→original. The
    map is not optional: the provider echoes back the name IT was given, and the
    kernel's tool registry only knows the original, so a call would be dispatched to a
    tool that does not exist.

    Same two sources as the prompt-encoded menu — local ``@tool`` functions and
    kernel-owned system tools — so the model is offered exactly the same set either
    way. Only the ENCODING differs: a schema the provider enforces, rather than a JSON
    example the model is asked to imitate.
    """
    defs: List[Dict[str, Any]] = []
    name_map: Dict[str, str] = {}

    # Added FIRST so they keep their plain names; a real tool that happens to share one
    # is disambiguated by sanitize_tool_name instead.
    for d in _DISCOVERY_TOOL_DEFS:
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
        # Discovery is handled by the loop, not dispatched to the tool plane.
        if provider_name in _DISCOVERY_TOOL_NAMES:
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

    if stop_reason == "end_turn":
        return {"action": "final_answer", "answer": text, "type": "text"}

    return {"action": "final_answer", "answer": text, "type": "text", "_inferred": True}


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
) -> AgentResult:
    """Drive the reason/retrieve/act loop for one task and return the final result.

    The retrieval loop always fires: the scratchpad is seeded with the
    server-assembled Global Workspace (``task.working_memory``, ADR-0022) and a
    **mandatory initial LTM recall** on the task (Path B kickoff), then the LLM may
    issue further capped ``memory_query`` / ``tool_call`` rounds before answering.

    ``result_type`` forces the final ``AgentResult.type`` (e.g. ``code`` → executor),
    overriding whatever ``type`` the LLM puts on its ``final_answer``.
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
    mem_exhausted = False
    veto_counts: Dict[str, int] = {}  # ADR-0041 D4: per-action hard-veto tally
    success_dedup_counts: Dict[str, int] = {}  # ADR-0041 D4: per-action already-succeeded tally
    truncation_retries = 0  # #2: consecutive token-limit truncations re-prompted
    inferred_answer_retries = 0  # non-action responses re-prompted (see note above)
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
    tool_defs, tool_name_map = build_tool_definitions(agent, system_tools)
    # ADR-0097 D8: under native tool-calling the model's history is THIS list, not a
    # re-rendered prompt. It is seeded once with the composed prompt and thereafter
    # grows by real turns — assistant turns carrying tool_calls, tool turns carrying
    # their results. Empty on the text path, which keeps rebuilding the prompt.
    conversation: List[Dict[str, Any]] = []
    # The id of the call awaiting its tool turn. Providers correlate on it and
    # reject a synthesized one, so it is carried rather than regenerated.
    pending_tool_call_id = ""

    action_protocol = _compose_action_protocol(agent, output_schema, system_tools, system_skills,
                                                allow_yield_subgoal=allow_yield_subgoal,
                                                native_tools=bool(native_tools and tool_defs))

    round_no = 0
    while True:
        extra = list(constraints or [])
        if mem_exhausted:
            extra.append("Memory budget is exhausted — answer best-effort from what you have.")
        prompt = build_prompt(
            role=role,
            task=task.text,
            # #3 trajectory reframe: number the bounded buffer as an ordered loop
            # history and stamp the current step, so the model reads its own progress
            # ("you are at step N; here is what you did") instead of a flat tag pile.
            context_str=wm.assemble(intent_vec, numbered=True),
            step_no=round_no + 1,
            # ADR-0048 D8: the action menu + behavioral rules go in <ActionProtocol>;
            # <OutputSchema> keeps only the per-turn action contract, so the recency-
            # anchored last line steers toward "emit one action", not a final-answer body.
            action_protocol=action_protocol,  # #4: composed once above (loop-invariant)
            output_schema=_PER_TURN_OUTPUT_CONTRACT,
            constraints=extra or None,
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
            if seed_system_tools:
                system_tools = _list_system_tools(agent, query=_seed_tool_query, full=False)
            action_protocol = _compose_action_protocol(
                agent, output_schema, system_tools, system_skills,
                allow_yield_subgoal=allow_yield_subgoal, native_tools=False)
            continue
        if native_fn is not None and tool_defs:
            if not conversation:
                # Seed once. The prompt carries role, task and the action menu; from
                # here the conversation carries what happened, so the prompt is never
                # rebuilt — re-narrating the model's own actions back at it as prose is
                # what made it summarise its history instead of continuing it.
                conversation.append({"role": "user", "content": prompt})
                wm = _ConversationMirror(wm, conversation)
            try:
                n_text, n_calls, n_stop = native_fn(
                    task.session_token_id,
                    conversation,
                    tool_defs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_ms=0,
                )
                # Echo the assistant turn back VERBATIM, tool_calls included. This is
                # what lets the model see that the call was its own.
                assistant_turn: Dict[str, Any] = {"role": "assistant", "content": n_text or ""}
                if n_calls:
                    assistant_turn["tool_calls"] = n_calls
                conversation.append(assistant_turn)
                action = action_from_native_turn(n_text, n_calls, n_stop, tool_name_map)
                if n_calls:
                    pending_tool_call_id = n_calls[0].get("id", "")
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
                # Back on the prose menu, which wants Tier-1: full schemas would bloat
                # every prompt with detail the menu never renders.
                if seed_system_tools:
                    system_tools = _list_system_tools(agent, query=_seed_tool_query, full=False)
                # The action menu is composed ONCE before the loop and currently says
                # "your tools are attached to this request". That is now false, so
                # recompose and retake the turn — one extra call, only in the rare
                # latch case, versus a whole run prompted for tools it will never be
                # given.
                action_protocol = _compose_action_protocol(
                    agent, output_schema, system_tools, system_skills,
                    allow_yield_subgoal=allow_yield_subgoal,
                    native_tools=False)
                continue

        if action is None:
            raw = agent.substrate.generate(
                task.session_token_id,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_ms=0,
            )
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
            wm.add_text(
                "<note>Your last action was TRUNCATED at the output-token limit before it "
                "finished — it was NOT executed. Do NOT emit a large payload in one action. "
                "To write a large file, call fast_large_write_file in SECTIONS: the first "
                "call with append=false (create/overwrite), then append=true for each "
                "subsequent section. Otherwise emit a smaller action.</note>"
            )
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
                wm.add_text("<note>memory query budget exhausted; answer best-effort</note>")
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
                    wm.add_text(
                        f"<note>VETOED: {name!r} already failed {failed_dups} times with similar "
                        f"arguments. Do NOT retry it — change your approach or give a final answer.</note>"
                    )
                    continue
                if failed_dups == 1:
                    # Soft nudge (→ ReBind): warn, but allow this one retry (transient failures recover).
                    wm.add_text(
                        f"<note>{name!r} already failed once. Retrying once more — if it fails "
                        f"again, change your approach.</note>"
                    )
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
                    wm.add_text(
                        f"<note>ALREADY DONE: {name!r} already completed successfully this run "
                        f"with these arguments — its result is above. Do NOT repeat it; continue "
                        f"to the next step, or if the task is now complete, emit final_answer.</note>"
                    )
                    continue

            tool_rounds += 1
            if tool_rounds > max_tool_rounds:
                raise ReActLoopError(f"tool-round budget ({max_tool_rounds}) exceeded")
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
            # An agent that opted out of system tools (seed_system_tools=False) must not be
            # able to pull them back in via discovery.
            if not seed_system_tools:
                wm.add_text("<note>Tools are not available to you. Either answer from what you "
                            "have, or use delegate_to_planner to hand the task to the planner.</note>")
                continue
            # ADR-0044 pull: the agent describes a capability need; the kernel
            # returns the matching tools, which we merge into the menu so the next
            # turn can tool_call one. The analogue of memory_query, for tools.
            need = (action.get("need") or action.get("query") or "").strip()
            if not need:
                wm.add_text("<note>find_tools needs a non-empty capability 'need' "
                            "(verb-first, e.g. 'search the web for a person').</note>")
                continue
            find_tools_calls += 1
            if find_tools_calls > 3:
                wm.add_text("<note>tool-discovery budget exhausted; use a tool already "
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
                tool_defs, tool_name_map = build_tool_definitions(agent, system_tools)
            logger.info("react_find_tools", extra={"round": round_no, "need": need,
                                                   "added": [t.get("name") for t in added],
                                                   "offered": len(tool_defs)})
            if added:
                wm.add_text("<note>Found tools for {!r}: {}. Call one with tool_call.</note>".format(
                    need, ", ".join(t.get("name") for t in added)))
            else:
                wm.add_text(f"<note>No tool matched {need!r}. Proceed without it — "
                            "answer from what you have, or yield_subgoal.</note>")
            continue

        if kind == "find_skills":
            # ADR-0046 pull: discover authored procedures (skills) for a need not in
            # the menu. The analogue of find_tools, for skills. Returns Tier-1 short
            # forms, merged into the menu; then the agent loads one with use_skill.
            need = (action.get("need") or action.get("query") or "").strip()
            if not need:
                wm.add_text("<note>find_skills needs a non-empty 'need' describing the "
                            "procedure/capability you want.</note>")
                continue
            find_skills_calls += 1
            if find_skills_calls > 3:
                wm.add_text("<note>skill-discovery budget exhausted; use a listed skill, "
                            "or proceed/answer.</note>")
                continue
            found = _list_system_skills(agent, query=need)
            seen = {sk.get("name") for sk in system_skills}
            added = [sk for sk in found if sk.get("name") not in seen]
            system_skills = system_skills + added
            logger.info("react_find_skills", extra={"round": round_no, "need": need,
                                                    "added": [sk.get("name") for sk in added]})
            if added:
                wm.add_text("<note>Found skills for {!r}: {}. Load one with use_skill.</note>".format(
                    need, ", ".join(sk.get("name") for sk in added)))
            else:
                wm.add_text(f"<note>No skill matched {need!r}. Proceed without it.</note>")
            continue

        if kind == "use_skill":
            # ADR-0046 D3: load a skill — inject its instructions into working memory
            # and (for system skills) activate its bundled tool grants for the run.
            # An agent-local skill loads purely locally (its grants are already within
            # the agent's envelope); a system skill is fetched Tier-2 and the kernel
            # confers + activates its grants run-scoped during the fetch.
            name = (action.get("skill") or action.get("name") or "").strip()
            if not name:
                wm.add_text("<note>use_skill needs a 'skill' name from the [skills] list.</note>")
                continue
            if name in loaded_skills:
                wm.add_text(f"<note>Skill {name!r} is already loaded — follow its instructions.</note>")
                continue
            skill_loads += 1
            if skill_loads > _DEFAULT_MAX_SKILL_LOADS:
                wm.add_text("<note>skill-load budget exhausted; act on a loaded skill or answer.</note>")
                continue
            sk = _load_skill(agent, name, task.session_token_id)
            logger.info("react_use_skill", extra={"round": round_no, "skill": name,
                                                  "loaded": sk is not None})
            if sk:
                loaded_skills.add(name)
                grants = sk.get("tool_grants") or []
                grant_note = (" Its tools ({}) are available for this task.".format(", ".join(grants))
                              if grants else "")
                wm.add_text(
                    "<skill name={!r}>\n{}\n</skill>\n".format(name, (sk.get("instructions") or "").strip())
                    + f"<note>Loaded skill {name!r}; follow its instructions.{grant_note}</note>")
            else:
                wm.add_text(f"<note>Skill {name!r} is not available to you. "
                            "Use a listed skill, or proceed.</note>")
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
_PER_TURN_OUTPUT_CONTRACT = (
    "Output EXACTLY ONE JSON object this turn — a single action chosen from "
    "<ActionProtocol>:\n"
    '  {"action": "<name>", ...}\n'
    "Pick the action that makes progress. Do NOT emit final_answer until you have "
    "gathered the evidence it requires (query memory, discover/call tools); when you "
    "do, its \"answer\" field must follow the format given for the final_answer "
    "action in <ActionProtocol>."
)


def _compose_action_protocol(agent, domain_schema: str, system_tools: Optional[List[Dict]] = None,
                             system_skills: Optional[List[Dict]] = None,
                             allow_yield_subgoal: bool = True,
                             native_tools: bool = False) -> str:
    """The agent-loop action menu + behavioral rules (the <ActionProtocol> body),
    with the agent's domain final-answer format folded into the final_answer action
    so it is documented but not the recency-anchored closer (ADR-0048 D8)."""
    base = build_output_schema(agent, system_tools, system_skills,
                               allow_yield_subgoal=allow_yield_subgoal,
                               native_tools=native_tools)
    if domain_schema and domain_schema.strip():
        base += (
            "\n\nWhen you emit final_answer, the \"answer\" field MUST follow this format:\n"
            + domain_schema.strip()
        )
    return base


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
            "description": get("description") or "",
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
            return s
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
