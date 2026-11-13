"""The context engine (ADR-0139 D3): what one agent run puts in front of the model.

It replaces the entry-count, cosine-evicted :class:`working_memory.WorkingMemory`,
which treated instructions like evidence: a loaded skill was cut to 240 chars and
evicted after ~4 tool calls, while every tool call was rendered twice (a ToolCard
and a Step). The engine is a pluggable interface (:class:`ContextEngine`); the
default implementation (:class:`DefaultContextEngine`) keeps four token-budgeted
regions plus the D14 memory-notes slot:

1. **Instructions** — loaded skill bodies, their grant notes and verbal
   reflections. PINNED: never offloaded, evicted or summarised away. Role and rules
   are already in the stable system tier. The region has its own budget; a skill
   load that would overflow it is refused by the loop (never truncated).
2. **Task** — the step objective, expected output and acceptance check. Rendered
   LAST in the user turn (D2: the task comes after the data).
3. **Evidence** — run-start documents (workspace seed, the seed recall, a delegated
   sub-goal's result), in tagged ``<documents>`` marked as data.
4. **History** — ONE ``<step>`` per action round. A tool call is one step whose
   output sits inside ``<evidence>`` (data); mid-run memory hits sit inside
   ``<documents>``.

The volatile-tier **memory-notes snapshot** (D14) is read once at run start and
rendered first, byte-identical for the whole run.

**Two-phase compaction** (adopted from Hermes Agent), on both protocol paths — the
text path's history items and the native path's conversation:

- *Phase 1, no LLM.* Tool outputs over ``tool_output_compact_chars`` outside the
  protected tail become a placeholder plus a cid. Re-opening a compacted cid with
  ``resolve_cid`` has no per-run cap. Pinned skill turns are never touched.
- *Phase 2, only when still over budget.* The middle turns (outside the head and
  the protected tail) are replaced by a structured summary (Goal, Constraints,
  Done/In progress/Blocked, Decisions, Artifacts, Next steps), updated iteratively
  on later compactions. Tool-call groups are kept intact (an assistant turn and its
  tool replies are summarised, or kept, together).
- *Guard.* No summariser, a summariser failure, or content larger than the
  summariser's input budget makes compaction fail LOUDLY (an error log plus a
  ``compaction_failed`` event) and fall back to phase 1 only. Turns are never
  dropped silently.
- *Raw history stays retrievable:* every compacted output is kept under its cid
  (:meth:`DefaultContextEngine.resolve_local`) and every summarised turn in
  :attr:`DefaultContextEngine.archive`. Compaction is an event
  (:attr:`DefaultContextEngine.events`), not a loss.

**Token estimate.** The SDK has no tokenizer, so tokens are estimated as
``ceil(chars / 4)`` (:func:`estimate_tokens`). It is a budget heuristic, not a
billing number: it over-counts dense code slightly and under-counts CJK text.

Every budget and threshold is a :class:`ContextConfig` field, resolved per run as
``run_think(context_config=...)`` > the agent's ``context_config`` attribute >
``CAMBRIAN_CONTEXT_<FIELD>`` env vars > the defaults below.

The previous working memory stays reachable for one release behind
``CAMBRIAN_CONTEXT_ENGINE=legacy`` (or ``agent.context_engine = "legacy"``), via
:class:`LegacyWorkingMemoryEngine`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from ._logging import safe_extra
from .helpers import _esc
from .prompt_library import fragment
from .working_memory import (
    EMPTY_RECALL_NOTE,
    Step,
    ToolCard,
    WorkingMemory,
    _condense_args,
    _render_call_xml,
    _render_memory_results,
    render_step_xml,
    short_query,
)

logger = logging.getLogger("cambrian.context")

CHARS_PER_TOKEN = 4

ENGINE_ENV = "CAMBRIAN_CONTEXT_ENGINE"
ENGINE_DEFAULT = "default"
ENGINE_LEGACY = "legacy"


def estimate_tokens(text: Any) -> int:
    """``ceil(len(text) / 4)``: the SDK's token heuristic (no tokenizer)."""
    if not text:
        return 0
    n = len(text) if isinstance(text, str) else len(str(text))
    return (n + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _cp(section: str, **values) -> str:
    return fragment("react-compaction", section).render(values)


def _turn(section: str, **values) -> str:
    return fragment("react-turn", section).render(values)


_CLOSING_TAG = re.compile(r"</\s*(documents?|evidence|Evidence|Instructions|MemoryNotes|Task|Trajectory|turns|previous_summary)\b",
                          re.IGNORECASE)


def neutralise_closing_tags(text: str) -> str:
    """Escape closing tags of the engine's own regions inside data, so a document
    cannot close its region and speak as instructions (D2 data regions)."""
    return _CLOSING_TAG.sub(lambda m: "&lt;/" + m.group(1), text or "")


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

_BOOL_FIELDS = frozenset({"phase2_enabled"})


@dataclass
class ContextConfig:
    """Every context budget and threshold (tokens unless the name says chars)."""

    # The whole prompt (system tier + user turn, or system + conversation + tool
    # definitions). Compaction runs when the estimate exceeds it.
    budget_tokens: int = 32000
    # The pinned instructions region (loaded skill bodies). None = the loop's
    # Phase 0 skill budget (react._DEFAULT_MAX_SKILL_CHARS / 4, i.e. 8000).
    instructions_budget_tokens: Optional[int] = None
    # The D14 memory-notes slot. A snapshot over it is shown cut, with an explicit
    # marker and a warning log (never silently).
    memory_notes_budget_tokens: int = 2000
    # Phase 1: a tool output longer than this, outside the protected tail, is
    # replaced by a placeholder plus cid.
    tool_output_compact_chars: int = 1200
    # A single tool output longer than this is compacted at once, even in the
    # tail (its head stays as a preview): one output must not blow the budget alone.
    tool_output_max_chars: int = 24000
    tool_output_preview_chars: int = 600
    # One window of a re-opened body (resolve_cid on a compacted or content-store
    # cid): a longer body is read page by page or searched, never inlined whole
    # (owner decision A3, T-20261112-worker-resolve-cid-inline-has-no-size-cap-a).
    # The kernel serves this, tool_output_max_chars, tool_output_preview_chars and
    # budget_tokens from execution.content as CAMBRIAN_CONTEXT_* env vars on every
    # agent it spawns, so the kernel's limits and the SDK's agree (A5); a led_ id is
    # paged by the kernel's ledger_get at the same size.
    resolve_page_chars: int = 16000
    # Protected tail: the last N history steps (text) / assistant-led turn groups
    # (native) are never compacted.
    protect_tail: int = 3
    # ADR-0150 D7 age trigger: a tool output older than this many tool rounds is
    # cleared to its placeholder even when the prompt is under budget (no LLM;
    # Anthropic's "tool-result clearing"). 0 = off. Served by the kernel as
    # CAMBRIAN_CONTEXT_KEEP_RECENT_TOOL_RESULTS (execution.content).
    keep_recent_tool_results: int = 0
    # Phase 2.
    phase2_enabled: bool = True
    summary_max_tokens: int = 1200
    # Guard: content above this is not sent to the summariser (compaction fails
    # loudly and falls back to phase 1 only).
    summariser_input_max_tokens: int = 48000

    @classmethod
    def from_env(cls, base: Optional["ContextConfig"] = None) -> "ContextConfig":
        cfg = dataclasses.replace(base) if base is not None else cls()
        for f in dataclasses.fields(cls):
            raw = os.getenv(f"CAMBRIAN_CONTEXT_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            try:
                if f.name in _BOOL_FIELDS:
                    val: Any = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    val = int(raw)
            except ValueError:
                logger.warning("context_bad_env", extra={"field": f.name, "value": raw})
                continue
            setattr(cfg, f.name, val)
        return cfg

    @classmethod
    def resolve(cls, agent: Any = None, explicit: Any = None) -> "ContextConfig":
        """``explicit`` > ``agent.context_config`` > env > defaults. Either source may
        be a :class:`ContextConfig` or a dict of field overrides."""
        cfg = cls.from_env()
        for src in (getattr(agent, "context_config", None), explicit):
            if src is None:
                continue
            if isinstance(src, ContextConfig):
                cfg = dataclasses.replace(src)
                continue
            if isinstance(src, dict):
                names = {f.name for f in dataclasses.fields(cls)}
                for k, v in src.items():
                    if k in names:
                        setattr(cfg, k, v)
                    else:
                        logger.warning("context_unknown_config_key", extra={"key": k})
        return cfg


# ──────────────────────────────────────────────────────────────────────
# The interface
# ──────────────────────────────────────────────────────────────────────

Summariser = Callable[[str, str], str]  # (previous_summary, turns_text) -> summary


@runtime_checkable
class ContextEngine(Protocol):
    """What ``react.run_think`` needs from a context engine.

    The write side is the loop's vocabulary; the read side renders the volatile user
    turn (text path) and compacts the conversation (native path).
    """

    # write side
    def add_text(self, content: str) -> None: ...
    def add_evidence(self, content: str, source: str = "") -> None: ...
    def add_seed_recall(self, query: str, results: list) -> None: ...
    def add_skill(self, name: str, body: str, note: str = "") -> None: ...
    def add_reflection(self, text: str) -> None: ...
    def add_step(self, kind: str, pinned: bool = False, **fields) -> None: ...
    def record_tool_call(self, card: ToolCard, body: str) -> None: ...
    def set_memory_notes(self, text: str) -> None: ...

    @property
    def cards(self) -> List[ToolCard]: ...

    # instructions budget
    def instructions_tokens(self) -> int: ...

    # read side
    def render_user_turn(self, *, task_text: str, step_no: int, notes: List[str],
                         expected_output: str = "", acceptance: str = "") -> Optional[str]: ...
    def assemble(self, intent_vec: Optional[List[float]] = None, numbered: bool = False) -> str: ...

    # compaction
    def compact_prompt(self, prompt_tokens: int) -> bool: ...
    def compact_conversation(self, conversation: List[Dict[str, Any]], fixed_tokens: int) -> bool: ...
    def pin_tool_call(self, tool_call_id: str) -> None: ...
    def resolve_local(self, cid: str) -> Optional[str]: ...


# ──────────────────────────────────────────────────────────────────────
# Legacy adapter (one release, behind CAMBRIAN_CONTEXT_ENGINE=legacy)
# ──────────────────────────────────────────────────────────────────────

class LegacyWorkingMemoryEngine(WorkingMemory):
    """The pre-D3 working memory behind the :class:`ContextEngine` interface.

    Behaviour is unchanged: the 10-entry cosine cap, the R7 offloader, a ToolCard
    plus a Step per tool call, no compaction, no memory-notes slot.
    ``render_user_turn`` returns None, which tells ``build_prompt_parts`` to keep
    the old Task + Trajectory layout.
    """

    def add_evidence(self, content: str, source: str = "") -> None:
        self.add_text(content)

    def add_seed_recall(self, query: str, results: list) -> None:
        # Same rule as the default engine: an empty seed recall adds nothing, and
        # the step's query attribute is capped by render_step_xml (short_query).
        if not results:
            return
        self.add_step(kind="memory_query", query=query, status="ok", results=results)

    def record_tool_call(self, card: ToolCard, body: str) -> None:
        self.add_tool_card(card)
        self.add_step(kind="tool_call", tool=card.tool, args=card.args, status=card.status,
                      summary=card.summary, body=body, cid=card.cid or "", chars=len(body))

    def set_memory_notes(self, text: str) -> None:
        return None  # the legacy engine has no D14 slot

    def instructions_tokens(self) -> int:
        return sum(estimate_tokens(e.body) for e in self._entries
                   if isinstance(e, Step) and e.kind == "use_skill")

    def render_user_turn(self, **_kw) -> Optional[str]:
        return None

    def compact_prompt(self, prompt_tokens: int) -> bool:
        return False

    def compact_conversation(self, conversation, fixed_tokens: int) -> bool:
        return False

    def pin_tool_call(self, tool_call_id: str) -> None:
        return None

    def resolve_local(self, cid: str) -> Optional[str]:
        return None


# ──────────────────────────────────────────────────────────────────────
# The default engine
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _Item:
    """One history entry: a typed Step, or a loop note (``step is None``)."""

    step: Optional[Step] = None
    note: str = ""
    compacted_cid: str = ""   # phase 1: the output is behind this cid
    compacted_chars: int = 0
    preview: str = ""         # oversize compaction keeps a head preview

    def tool_body(self) -> str:
        s = self.step
        return (s.body or s.summary or "") if s is not None else ""


@dataclass
class _Doc:
    source: str
    content: str
    attrs: str = ""
    raw: bool = False  # content is already markup (workspace / recall children)


class DefaultContextEngine:
    """Token-budgeted regions plus two-phase compaction (module docstring)."""

    def __init__(self, config: Optional[ContextConfig] = None, *,
                 offload_fn: Optional[Callable[[str], Optional[str]]] = None,
                 summariser: Optional[Summariser] = None,
                 embed_fn: Optional[Callable[[str], Optional[List[float]]]] = None) -> None:
        self.config = config or ContextConfig()
        self._offload_fn = offload_fn
        self._summariser = summariser
        self._embed_fn = embed_fn
        self._cards: List[ToolCard] = []
        self._docs: List[_Doc] = []
        self._instructions: List[Tuple[str, str]] = []   # (kind, rendered) in load order
        self._skill_bodies: List[Tuple[str, str, str]] = []  # (name, body, note)
        self._history: List[_Item] = []
        self._notes_snapshot: str = ""
        self._summary: str = ""
        self._summarised_turns = 0
        self._next_n = 1
        self._local: Dict[str, str] = {}      # cid -> raw body (every compacted output)
        self._cid_by_digest: Dict[str, str] = {}  # sha256(body) -> its cid (stable re-compaction)
        self._pinned_ids: set = set()         # native tool_call ids that carry a skill
        self._compacted_ids: set = set()      # native tool_call ids already placeholdered
        # Native tool turns that are a resolve_cid WINDOW of a cid: tool_call id ->
        # (cid, page, pages, total chars). Compacting one points back at that cid and
        # page; storing it under a new cid made the model re-open the new cid, which
        # was compacted again on the next round (the p12 ping-pong,
        # T-20261112-worker-resolve-cid-inline-has-no-size-cap-a).
        self._window_by_tid: Dict[str, Tuple[str, int, int, int]] = {}
        # A call whose output was compacted may legitimately be re-issued: the model
        # needs the content back and re-calls the tool instead of resolve_cid. The
        # loop serves such a re-read from here (no re-execution) instead of letting
        # the success-dedup gate end the run (T-20261105-after-context-compaction-
        # re-reading-a-compacted). action signature -> cid; tool_call id -> signature.
        self._sig_by_tid: Dict[str, str] = {}
        self._compacted_by_sig: Dict[str, str] = {}
        self._native_head: Optional[str] = None
        self.archive: List[Any] = []          # raw history replaced by a summary
        self.events: List[Dict[str, Any]] = []

    # ── write side ────────────────────────────────────────────────────

    def add_text(self, content: str) -> None:
        if content and content.strip():
            self._history.append(_Item(note=content.strip()))

    def add_evidence(self, content: str, source: str = "") -> None:
        if content and content.strip():
            self._docs.append(_Doc(source=source or "context", content=content.strip(), raw=True))

    def add_seed_recall(self, query: str, results: list) -> None:
        """The run-start recall as an Evidence document, or nothing.

        The seed query IS the task text. It used to ride along as a ``query=``
        attribute, and an empty recall still added a document; a child whose recall
        came back empty then saw its own whole objective inside the data-marked
        Evidence region, read it as instructions planted in evidence, and refused its
        task (T-20261106-agent-refuses-its-whole-legitimate-task-when). So: no hits,
        no document; hits, a document with no query echo (the Task region already
        says what was asked)."""
        if not results:
            return
        self._docs.append(_Doc(source="memory", attrs=" kind='recall'",
                               content=_render_memory_results(results), raw=True))

    def add_skill(self, name: str, body: str, note: str = "") -> None:
        body = (body or "").strip()
        self._skill_bodies.append((name, body, (note or "").strip()))
        block = f"<skill name={_esc(name)!r}>\n{body}\n</skill>"
        if note and note.strip():
            block += f"\n<note>{_esc(note.strip())}</note>"
        self._instructions.append(("skill", block))

    def add_reflection(self, text: str) -> None:
        clean = " ".join((text or "").split())
        if clean:
            self._instructions.append(("reflection", f"<reflection>\n  {_esc(clean)}\n</reflection>"))

    def add_step(self, kind: str, pinned: bool = False, **fields) -> None:
        fields.pop("n", None)
        if kind == "reflection":
            self.add_reflection(fields.get("body", ""))
            return
        step = Step(kind=kind, n=self._next_n, pinned=pinned, **fields)
        self._next_n += 1
        self._history.append(_Item(step=step))

    def record_tool_call(self, card: ToolCard, body: str) -> None:
        """ONE history step per tool call; the card is kept for the recurrence gate
        (``cards``) but never rendered."""
        if card.status in ("error", "denied") and card.action_vec is None and self._embed_fn:
            from .working_memory import action_text
            try:
                card.action_vec = self._embed_fn(action_text(card.tool, card.args)) or None
            except Exception:  # noqa: BLE001 — degrade to exact matching
                card.action_vec = None
        self._cards.append(card)
        self.add_step(kind="tool_call", tool=card.tool, args=card.args, status=card.status,
                      summary=card.summary, body=body, cid=card.cid or "", chars=len(body))
        item = self._history[-1]
        if len(body) > self.config.tool_output_max_chars and not card.cid:
            self._compact_item(item, keep_preview=True, reason="oversize")

    def tool_turn_content(self, content: str, tool: str) -> str:
        """The native tool turn for one output: verbatim, unless it alone is over
        ``tool_output_max_chars``, in which case a head preview plus a cid (kept
        retrievable) replaces it even in the protected tail."""
        if len(content) <= self.config.tool_output_max_chars:
            return content
        cid = self._store(content)
        preview = content[: self.config.tool_output_preview_chars]
        self._event(event="compaction", phase=1, path="native", reason="oversize", cids=[cid],
                    chars=len(content))
        return _cp("placeholder-preview", tool=tool, chars=len(content), cid=cid, preview=preview)

    def add_tool_card(self, card: ToolCard) -> None:
        """Compatibility: a card without a history step (recurrence evidence only)."""
        self._cards.append(card)

    def set_memory_notes(self, text: str) -> None:
        """The D14 frozen snapshot. Set once at run start; later calls are ignored so
        the region stays byte-identical for the run."""
        if self._notes_snapshot or not (text and text.strip()):
            return
        text = text.strip()
        budget_chars = self.config.memory_notes_budget_tokens * CHARS_PER_TOKEN
        if len(text) > budget_chars:
            logger.warning("context_memory_notes_over_budget", extra={
                "chars": len(text), "budget_chars": budget_chars})
            text = text[:budget_chars] + "\n" + _turn(
                "memory-notes-cut", shown=budget_chars, total=len(text))
        self._notes_snapshot = text

    @property
    def cards(self) -> List[ToolCard]:
        return list(self._cards)

    def instructions_tokens(self) -> int:
        return sum(estimate_tokens(b) for _, b, _ in self._skill_bodies)

    def pin_tool_call(self, tool_call_id: str) -> None:
        if tool_call_id:
            self._pinned_ids.add(tool_call_id)

    def resolve_local(self, cid: str) -> Optional[str]:
        return self._local.get(cid)

    def note_resolve_window(self, tool_call_id: str, cid: str, page: int, pages: int,
                            total_chars: int) -> None:
        """Mark a native tool turn as page ``page`` of ``cid`` (see _window_by_tid)."""
        if tool_call_id and cid:
            self._window_by_tid[tool_call_id] = (cid, page, pages, total_chars)

    def is_compaction_cid(self, cid: str) -> bool:
        return cid in self._local

    def note_tool_call(self, tool_call_id: str, tool: str, args: Any) -> None:
        """Native path: remember which action a tool_call id answered, so compacting
        its reply can be matched to a later re-issue of the same action."""
        if tool_call_id:
            from .working_memory import action_text
            self._sig_by_tid[tool_call_id] = action_text(tool, args)

    def take_compacted_output(self, tool: str, args: Any) -> Optional[str]:
        """The raw output of an earlier identical call that compaction placeholdered,
        or None. Taken once: the body is back in context after this, so a further
        identical call with nothing compacted in between is a real duplicate."""
        from .working_memory import action_text
        cid = self._compacted_by_sig.pop(action_text(tool, args), None)
        return self._local.get(cid) if cid else None

    def best_effort_text(self) -> str:
        """The loop's degraded answer: what the run gathered (history, else
        evidence), NEVER the instructions region - a pinned skill rendered as the
        deliverable is the defect this replaced."""
        return self.render_history() or self.render_evidence()

    def __len__(self) -> int:
        return len(self._history) + len(self._docs) + len(self._instructions)

    # ── rendering ─────────────────────────────────────────────────────

    def render_memory_notes(self) -> str:
        if not self._notes_snapshot:
            return ""
        return ("<MemoryNotes source='principal' frozen='run-start'>\n"
                f"{_turn('memory-notes-header')}\n\n"
                f"{neutralise_closing_tags(self._notes_snapshot)}\n</MemoryNotes>")

    def render_instructions(self) -> str:
        if not self._instructions:
            return ""
        return ("<Instructions>\n" + _turn("instructions-header") + "\n\n"
                + "\n".join(b for _, b in self._instructions) + "\n</Instructions>")

    def render_evidence(self) -> str:
        if not self._docs:
            return ""
        docs = []
        for d in self._docs:
            body = neutralise_closing_tags(d.content) if d.raw else _esc(d.content)
            docs.append(f"<document source={_esc(d.source)!r}{d.attrs}>\n{body}\n</document>")
        return ("<Evidence>\n" + _turn("evidence-header") + "\n<documents>\n"
                + "\n".join(docs) + "\n</documents>\n</Evidence>")

    def _render_item(self, it: _Item) -> str:
        if it.step is None:
            return it.note
        s = it.step
        if s.kind == "tool_call":
            call_xml = _render_call_xml(s.tool, _condense_args(s.args or {}))
            src = f"tool:{s.tool}"
            if it.compacted_cid:
                if it.preview:
                    inner = _cp("placeholder-preview", tool=s.tool, chars=it.compacted_chars,
                                cid=it.compacted_cid, preview=_esc(it.preview))
                else:
                    inner = _cp("placeholder", tool=s.tool, chars=it.compacted_chars,
                                cid=it.compacted_cid)
                ev = (f"  <evidence source={_esc(src)!r} status={_esc(s.status)!r}"
                      f" compacted_cid={_esc(it.compacted_cid)!r} chars={it.compacted_chars!r}>\n"
                      f"    {inner}\n  </evidence>")
            elif s.cid:
                ev = (f"  <evidence source={_esc(src)!r} status={_esc(s.status)!r}"
                      f" offloaded_cid={_esc(s.cid)!r}>\n"
                      f"    {_esc(s.summary or '')}\n"
                      f"    {_cp('kernel-offloaded', cid=s.cid)}\n  </evidence>")
            else:
                ev = (f"  <evidence source={_esc(src)!r} status={_esc(s.status)!r}>\n"
                      f"{_esc(it.tool_body())}\n  </evidence>")
            return (f"<step n={s.n!r} type='tool_call' tool={_esc(s.tool)!r} status={_esc(s.status)!r}>\n"
                    f"  <call>{call_xml}\n  </call>\n{ev}\n</step>")
        if s.kind == "memory_query":
            head = (f"<step n={s.n!r} type='memory_query' query={_esc(short_query(s.query))!r}"
                    f" status={_esc(s.status)!r}>\n")
            if not s.results:
                # No hits: a note in the model's own step, not an empty <documents>.
                return f"{head}  {EMPTY_RECALL_NOTE}\n</step>"
            return f"{head}<documents>\n{_render_memory_results(s.results)}\n</documents>\n</step>"
        return render_step_xml(s)

    def render_history(self) -> str:
        parts = []
        if self._summary:
            parts.append(self._summary_block())
        parts.extend(self._render_item(it) for it in self._history)
        return "\n".join(p for p in parts if p)

    def _summary_block(self) -> str:
        return _cp("summary-block", turn_count=self._summarised_turns,
                   summary=neutralise_closing_tags(self._summary))

    def render_user_turn(self, *, task_text: str, step_no: int, notes: List[str],
                         expected_output: str = "", acceptance: str = "") -> str:
        regions = [self.render_memory_notes(), self.render_instructions(), self.render_evidence()]
        hist = self.render_history()
        if hist:
            regions.append(f"<Trajectory>\n{_turn('trajectory-header', round=step_no)}\n"
                           f"{_turn('data-note')}\n\n{hist}\n</Trajectory>")
        task = [task_text.strip()]
        if expected_output and expected_output.strip():
            task.append(_turn("expected-output", expected_output=expected_output.strip()))
        if acceptance and acceptance.strip():
            task.append(_turn("acceptance", acceptance=acceptance.strip()))
        regions.append("<Task>\n" + "\n\n".join(task) + "\n</Task>")
        regions.extend(n.strip() for n in notes or [] if n and n.strip())
        return "\n\n".join(r for r in regions if r)

    def assemble(self, intent_vec: Optional[List[float]] = None, numbered: bool = False) -> str:
        """Everything gathered so far (instructions, evidence, history) without the
        task: the loop's degraded best-effort answer."""
        return "\n\n".join(p for p in (self.render_instructions(), self.render_evidence(),
                                       self.render_history()) if p)

    # ── compaction: shared pieces ────────────────────────────────────

    def _store(self, body: str) -> str:
        """Keep a compacted body retrievable; returns its cid. The kernel content
        store when an offloader is wired (the cid then also works as a ``$cid`` tool
        argument), else a deterministic local id. Raw text is ALWAYS kept locally.
        The same body always gets the same cid: compacting it again (after it was
        re-opened) must not hand the model a new id to re-open."""
        digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()
        known = self._cid_by_digest.get(digest)
        if known is not None:
            return known
        cid = None
        if self._offload_fn is not None:
            try:
                cid = self._offload_fn(body)
            except Exception:  # noqa: BLE001 — degrade to a local cid
                cid = None
        if not cid:
            cid = "ctx-" + digest[:16]
        self._local[cid] = body
        self._cid_by_digest[digest] = cid
        return cid

    def _event(self, **ev) -> None:
        self.events.append(ev)
        level = logging.ERROR if ev.get("event") == "compaction_failed" else logging.INFO
        logger.log(level, "context_" + ev.get("event", "compaction"), extra=safe_extra(
            {k: v for k, v in ev.items() if k != "event"}))

    def _summarise(self, turns_text: str, path: str) -> Optional[str]:
        """Phase 2 with the guard. Returns the summary, or None after a LOUD failure."""
        if self._summariser is None:
            self._event(event="compaction_failed", phase=2, path=path, reason="no summariser")
            return None
        need = estimate_tokens(turns_text) + estimate_tokens(self._summary)
        if need > self.config.summariser_input_max_tokens:
            self._event(event="compaction_failed", phase=2, path=path,
                        reason="content larger than the summariser input budget",
                        tokens=need, budget=self.config.summariser_input_max_tokens)
            return None
        try:
            out = self._summariser(self._summary, turns_text)
        except Exception as exc:  # noqa: BLE001 — loud fallback, never drop turns
            self._event(event="compaction_failed", phase=2, path=path, reason=f"summariser error: {exc}")
            return None
        out = (out or "").strip()
        if not out:
            self._event(event="compaction_failed", phase=2, path=path, reason="empty summary")
            return None
        return out

    # ── compaction: text path ────────────────────────────────────────

    def _compact_item(self, it: _Item, *, keep_preview: bool = False, reason: str = "phase1") -> int:
        body = it.tool_body()
        cid = self._store(body)
        it.compacted_cid, it.compacted_chars = cid, len(body)
        it.preview = body[: self.config.tool_output_preview_chars] if keep_preview else ""
        if it.step is not None and it.step.kind == "tool_call" and not keep_preview:
            from .working_memory import action_text
            self._compacted_by_sig[action_text(it.step.tool, it.step.args)] = cid
        saved = estimate_tokens(body) - estimate_tokens(it.preview) - 40
        if reason == "oversize":
            self._event(event="compaction", phase=1, path="text", reason=reason, cids=[cid],
                        chars=len(body))
        return max(saved, 0)

    def _placeholder(self, tool: str, body: str, cid: str) -> str:
        """The placeholder of a cleared tool output: its run-ledger id when the kernel
        recorded it (readable with ledger_get), else the SDK cid (ADR-0150 D7)."""
        ref = ledger_ref_of(body)
        if ref:
            return _cp("placeholder-ledger", tool=tool, chars=len(body), ref=ref)
        return _cp("placeholder", tool=tool, chars=len(body), cid=cid)

    def _clear_aged_native(self, conversation: List[Dict[str, Any]]) -> bool:
        """ADR-0150 D7 age trigger (native path): every tool output but the newest
        ``keep_recent_tool_results`` becomes its placeholder, under budget or not.
        Pinned (skill) turns and resolve windows are left to the budget pass."""
        keep = self.config.keep_recent_tool_results
        if keep <= 0:
            return False
        tool_idx = [i for i, m in enumerate(conversation) if m.get("role") == "tool"]
        old = tool_idx[:max(len(tool_idx) - keep, 0)]
        if not old:
            return False
        names = {c.get("id"): c.get("name", "") for m in conversation
                 for c in (m.get("tool_calls") or [])}
        cids = []
        for i in old:
            m = conversation[i]
            tid = m.get("tool_call_id", "")
            body = m.get("content") or ""
            if (tid in self._pinned_ids or tid in self._compacted_ids or tid in self._window_by_tid
                    or len(body) <= self.config.tool_output_compact_chars):
                continue
            cid = self._store(body)
            m["content"] = self._placeholder(names.get(tid) or "the tool", body, cid)
            self._compacted_ids.add(tid)
            sig = self._sig_by_tid.get(tid)
            if sig:
                self._compacted_by_sig[sig] = cid
            cids.append(cid)
        if cids:
            self._event(event="compaction", phase=1, path="native", reason="age", items=len(cids),
                        cids=cids, keep_recent=keep)
        return bool(cids)

    def _clear_aged_text(self) -> bool:
        """ADR-0150 D7 age trigger (text path), as ``_clear_aged_native``."""
        keep = self.config.keep_recent_tool_results
        if keep <= 0:
            return False
        calls = [it for it in self._history if it.step is not None and it.step.kind == "tool_call"]
        cids = []
        for it in calls[:max(len(calls) - keep, 0)]:
            if it.compacted_cid or it.step.cid or len(it.tool_body()) <= self.config.tool_output_compact_chars:
                continue
            self._compact_item(it)
            cids.append(it.compacted_cid)
        if cids:
            self._event(event="compaction", phase=1, path="text", reason="age", items=len(cids),
                        cids=cids, keep_recent=keep)
        return bool(cids)

    def compact_prompt(self, prompt_tokens: int) -> bool:
        """Bring the text-path prompt under budget. ``prompt_tokens`` is the estimate
        of the full prompt just rendered. Returns True when anything changed."""
        aged = self._clear_aged_text()
        excess = prompt_tokens - self.config.budget_tokens
        if excess <= 0:
            return aged
        tail = max(self.config.protect_tail, 0)
        cut = max(len(self._history) - tail, 0)
        saved, cids = 0, []
        for it in self._history[:cut]:
            s = it.step
            if (s is not None and s.kind == "tool_call" and not it.compacted_cid and not s.cid
                    and len(it.tool_body()) > self.config.tool_output_compact_chars):
                saved += self._compact_item(it)
                cids.append(it.compacted_cid)
        changed = bool(cids) or aged
        if cids:
            self._event(event="compaction", phase=1, path="text", items=len(cids), cids=cids,
                        tokens_before=prompt_tokens, tokens_saved=saved)
        if saved >= excess or not self.config.phase2_enabled:
            return changed
        middle = self._history[:cut]
        if not middle:
            self._event(event="compaction_failed", phase=2, path="text",
                        reason="nothing outside the protected tail to summarise",
                        tokens=prompt_tokens - saved, budget=self.config.budget_tokens)
            return changed
        turns = "\n".join(self._render_item(it) for it in middle)
        summary = self._summarise(turns, "text")
        if summary is None:
            return changed
        self.archive.extend(middle)
        del self._history[:cut]
        self._summary = summary
        self._summarised_turns += len(middle)
        self._event(event="compaction", phase=2, path="text", items=len(middle),
                    tokens_before=prompt_tokens, summary_tokens=estimate_tokens(summary))
        return True

    # ── compaction: native path ──────────────────────────────────────

    @staticmethod
    def conversation_tokens(conversation: List[Dict[str, Any]]) -> int:
        n = 0
        for m in conversation:
            n += estimate_tokens(m.get("content") or "")
            for c in m.get("tool_calls") or []:
                n += estimate_tokens(c.get("name", "")) + estimate_tokens(c.get("arguments", ""))
        return n

    @staticmethod
    def _groups(conversation: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Assistant-led groups after the head (message 0): an assistant turn, its
        tool replies and any notes up to the next assistant turn. Messages between
        the head and the first assistant turn form a group of their own. A group is
        the unit compaction keeps or summarises, so a tool call is never separated
        from its reply."""
        groups: List[Tuple[int, int]] = []
        start = 1
        for i in range(1, len(conversation)):
            if conversation[i].get("role") == "assistant" and i > start:
                groups.append((start, i))
                start = i
        if start < len(conversation):
            groups.append((start, len(conversation)))
        return groups

    def compact_conversation(self, conversation: List[Dict[str, Any]], fixed_tokens: int) -> bool:
        """Bring the native conversation under budget, in place. ``fixed_tokens`` is
        the system tier plus the tool definitions."""
        if len(conversation) < 2:
            return False
        aged = self._clear_aged_native(conversation)
        before = fixed_tokens + self.conversation_tokens(conversation)
        excess = before - self.config.budget_tokens
        if excess <= 0:
            return aged
        if self._native_head is None:
            self._native_head = conversation[0].get("content") or ""
        groups = self._groups(conversation)
        tail = max(self.config.protect_tail, 0)
        middle = groups[:max(len(groups) - tail, 0)]
        names = {c.get("id"): c.get("name", "") for m in conversation
                 for c in (m.get("tool_calls") or [])}
        saved, cids = 0, []
        for a, b in middle:
            for m in conversation[a:b]:
                tid = m.get("tool_call_id", "")
                body = m.get("content") or ""
                if (m.get("role") != "tool" or tid in self._pinned_ids or tid in self._compacted_ids
                        or len(body) <= self.config.tool_output_compact_chars):
                    continue
                window = self._window_by_tid.get(tid)
                if window is not None:
                    # A resolve window: point back at the original cid and page; no
                    # new cid, so nothing new to re-open.
                    cid, page, pages, total = window
                    m["content"] = _cp("placeholder-window", cid=cid, page=page, pages=pages,
                                       chars=total or len(body))
                    self._compacted_ids.add(tid)
                    saved += estimate_tokens(body) - estimate_tokens(m["content"])
                    cids.append(cid)
                    continue
                cid = self._store(body)
                m["content"] = self._placeholder(names.get(tid) or "the tool", body, cid)
                self._compacted_ids.add(tid)
                sig = self._sig_by_tid.get(tid)
                if sig:
                    self._compacted_by_sig[sig] = cid
                saved += estimate_tokens(body) - estimate_tokens(m["content"])
                cids.append(cid)
        changed = bool(cids) or aged
        if cids:
            self._event(event="compaction", phase=1, path="native", items=len(cids), cids=cids,
                        tokens_before=before, tokens_saved=saved)
        if saved >= excess or not self.config.phase2_enabled:
            return changed
        if not middle:
            self._event(event="compaction_failed", phase=2, path="native",
                        reason="nothing outside the protected tail to summarise",
                        tokens=before - saved, budget=self.config.budget_tokens)
            return changed
        lo, hi = middle[0][0], middle[-1][1]
        raw = conversation[lo:hi]
        summary = self._summarise(_render_turns(raw), "native")
        if summary is None:
            return changed
        self.archive.extend(dict(m) for m in raw)
        del conversation[lo:hi]
        self._summary = summary
        self._summarised_turns += len(middle)
        # The head keeps the original first turn; the summary and the pinned skills
        # (whose tool turns may just have been summarised) are folded into it, so the
        # conversation still alternates user -> assistant and no instruction is lost.
        head = [self._native_head, self._summary_block()]
        if self._skill_bodies:
            head.append(self.render_instructions())
        conversation[0] = dict(conversation[0], content="\n\n".join(p for p in head if p))
        self._event(event="compaction", phase=2, path="native", items=len(middle),
                    messages=len(raw), tokens_before=before,
                    tokens_after=fixed_tokens + self.conversation_tokens(conversation))
        return True


_LEDGER_REF_RE = re.compile(r'"(?:_ledger_ref|cid)"\s*:\s*"(led_[0-9a-f]+)"')


def ledger_ref_of(body: str) -> str:
    """The run-ledger id a tool output carries (the kernel's ``_ledger_ref`` of an
    inline result, or an offloaded result's ``cid``), or "" (ADR-0150 D7)."""
    m = _LEDGER_REF_RE.search(body or "")
    return m.group(1) if m else ""


def _render_turns(messages: List[Dict[str, Any]]) -> str:
    """Serialise conversation turns for the summariser (data, closing tags escaped)."""
    out = []
    for m in messages:
        role = m.get("role", "")
        body = neutralise_closing_tags(m.get("content") or "")
        calls = m.get("tool_calls") or []
        if calls:
            names = "; ".join("{}({})".format(c.get("name", ""), (c.get("arguments") or "")[:300])
                              for c in calls)
            body = (body + "\n" if body else "") + f"[tool calls: {neutralise_closing_tags(names)}]"
        tid = m.get("tool_call_id")
        attr = f" tool_call_id={tid!r}" if tid else ""
        out.append(f"<turn role={role!r}{attr}>\n{body}\n</turn>")
    return "\n".join(out)


def make_engine(kind: Any, *, config: ContextConfig, embed_fn=None, offload_fn=None,
                summariser: Optional[Summariser] = None):
    """Build the engine a run uses. ``kind`` is an engine instance, a factory
    ``(config, embed_fn, offload_fn, summariser) -> engine``, ``"default"`` or
    ``"legacy"``; ``None`` reads ``CAMBRIAN_CONTEXT_ENGINE`` (default ``"default"``)."""
    if kind is None:
        kind = os.getenv(ENGINE_ENV) or ENGINE_DEFAULT
    if isinstance(kind, str):
        k = kind.strip().lower()
        if k == ENGINE_LEGACY:
            return LegacyWorkingMemoryEngine(embed_fn=embed_fn, offload_fn=offload_fn)
        if k not in (ENGINE_DEFAULT, ""):
            logger.warning("context_unknown_engine", extra={"engine": kind})
        return DefaultContextEngine(config, offload_fn=offload_fn, summariser=summariser,
                                    embed_fn=embed_fn)
    if callable(kind) and not isinstance(kind, ContextEngine):
        return kind(config, embed_fn, offload_fn, summariser)
    return kind


def _render_notes_documents(docs) -> str:
    """ADR-0139 D14: the GetMemoryNotes documents, one labelled section per scope,
    PERSONAL first, each marked as data. A closing ``</notes`` inside a body is
    neutralised, so a note cannot end its own section."""
    order = {"personal": 0, "org": 1}
    out = []
    for d in sorted((d for d in docs if isinstance(d, dict)),
                    key=lambda d: order.get(str(d.get("scope", "")), 2)):
        body = str(d.get("content") or "").strip()
        if not body:
            continue
        body = re.sub(r"</\s*notes\b", "&lt;/notes", body, flags=re.IGNORECASE)
        attrs = f" scope={_esc(str(d.get('scope') or 'unknown'))!r} kind='data'"
        if d.get("truncated"):
            attrs += " truncated='true'"
        out.append(f"<notes{attrs}>\n{body}\n</notes>")
    return "\n".join(out)


def memory_notes_text(value: Any) -> str:
    """Normalise whatever ``substrate.get_memory_notes`` returns into text: a str, an
    object with ``.text``/``.notes``/``.content``, a dict with one of those keys, or a
    list of notes (joined one per line)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, dict):
        if isinstance(value.get("documents"), (list, tuple)):
            return _render_notes_documents(value["documents"])
        for k in ("text", "notes", "content"):
            if k in value:
                return memory_notes_text(value[k])
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (memory_notes_text(v) for v in value) if t)
    for k in ("text", "notes", "content"):
        v = getattr(value, k, None)
        # Only plain values are followed: an arbitrary object (a test double, a
        # proxy) that answers every attribute would otherwise recurse forever.
        if isinstance(v, (str, bytes, bytearray, list, tuple, dict)):
            return memory_notes_text(v)
    return ""


__all__ = [
    "CHARS_PER_TOKEN", "ContextConfig", "ContextEngine", "DefaultContextEngine",
    "LegacyWorkingMemoryEngine", "estimate_tokens", "make_engine", "memory_notes_text",
    "neutralise_closing_tags",
]

