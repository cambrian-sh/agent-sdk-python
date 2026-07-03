"""The agent's Local Recurrent Workspace buffers (ADR-0041 D1).

This module replaces ``run_think``'s flat ``scratch: List[str]`` with a *typed*
working memory. The unit of episodic memory is the :class:`ToolCard` — a
**provenance record** of one tool invocation: what the agent intended, what it
called, *with what args*, and the *outcome status*. The old loop appended only
the raw result (``<tool name=…>{result}</tool>``), so the model could see an
error but not *what it ran* — the cause of the dumb-retry failures
(``docs/requirements/AGENTS_RUNNING_CONTEXT.md`` flaw #3).

This slice (0041-01) is **behavior-preserving**: the same content reaches the
prompt, sourced from the structured buffers, with tool entries now carrying
provenance. Bounding/relevance (0041-03) and the recurrence gate (0041-04) build
on the :class:`ToolCard` episodic buffer introduced here.

The LLM remains a storage-less Central Executive (Baddeley); this is the memory
it directs.
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .helpers import _esc  # v2: XML-escape for the rich trajectory renderer

# Default working-memory capacity: the max entries assembled into the prompt each
# turn (ADR-0041 D2). Bounding here is what turns O(N²) prompt growth into ~O(k).
DEFAULT_CAP = 10

# R7 (WORKING-MEMORY-CONDENSATION): a TEXT block larger than this is offloaded to
# the ContentStore — the prompt then carries a {gist + cid} pointer instead of the
# full payload (the agent's own large step output, a heavy recalled fact). The full
# text is one resolve_content(cid) drill-down away. Mirrors the ToolCard offload
# (ADR-0041 D3) generalized from tool results to ALL large working-memory text.
DEFAULT_TEXT_INLINE_CAP = 1000

# A tool-call ARGUMENT value longer than this is replaced by a "<N chars>" marker
# when the card renders (#2). A write_file's multi-KB `content` is provenance the
# agent already acted on — re-emitting it in the card every turn is the same raw-
# payload bloat we fixed for results. Small values (a path, a query, a number)
# render verbatim so the card still shows WHAT was called with.
ARG_VALUE_INLINE_CAP = 120


def _status_of(result: Any) -> str:
    """Derive an outcome status for a tool result.

    A tool result is, by the loop's convention, a dict that may carry an
    ``error``/``denied`` marker (see ``react._call_system_tool``). Anything else
    is treated as a successful return.
    """
    if isinstance(result, dict):
        if result.get("error"):
            return "error"
        if result.get("denied"):
            return "denied"
    return "ok"


def _condense_dict(result: dict) -> Optional[str]:
    """Content-aware one-line condensation of the structured tool-result shapes the
    loop produces (ADR-0041 D3). Returns ``None`` when the dict is not a recognized
    shape, so the caller falls back to generic JSON truncation.

    The shapes worth collapsing — observed bloating ``<LTMContext>`` in practice:
    - **error / denied markers** (``react._call_system_tool`` results): keep only the
      actionable message, drop the verbose payload (e.g. a repeated ENOENT blob).
    - **list-bearing results** (search / list / file tools): a count plus a small
      head, not the entire list dumped as JSON (e.g. a 7900-char directory listing).
    """
    if result.get("error"):
        msg = " ".join(str(result["error"]).split())
        tool = result.get("tool")
        line = f"error: {msg[:300]}"
        return f"{line} (tool {tool})" if tool else line
    if result.get("denied"):
        reason = result.get("deny_reason") or result.get("denied")
        return f"denied: {' '.join(str(reason).split())[:300]}"
    for key in ("matches", "results", "items", "files", "entries", "tools"):
        v = result.get(key)
        if isinstance(v, list):
            head = ", ".join(str(x) for x in v[:5])
            more = f" …(+{len(v) - 5} more)" if len(v) > 5 else ""
            return f"{len(v)} {key}: {head[:300]}{more}"
    return None


def _summarize(result: Any, max_chars: int = 400, summarizer: Optional[Any] = None) -> str:
    """Summary of a tool result for the card (ADR-0041 D3).

    **Content-aware first:** the common structured shapes (error/denied markers,
    list-bearing results) collapse to a single informative line via
    :func:`_condense_dict` — instead of a full-JSON dump or a blind head-truncation
    that keeps 400 chars of noise. **Heuristic fallback, no LLM call:** an
    unrecognized small result is kept verbatim; a heavy one is truncated with a
    length marker. **Opt-in LLM path:** when a ``summarizer`` callable is supplied
    (the agent's ``tool_summarizer``) and the result still exceeds ``max_chars``, it
    is summarized by that callable; a summarizer failure degrades to truncation.
    """
    if isinstance(result, dict):
        condensed = _condense_dict(result)
        if condensed is not None:
            return condensed
    s = result if isinstance(result, str) else json.dumps(result, default=str)
    if len(s) <= max_chars:
        return s
    if summarizer is not None:
        try:
            return str(summarizer(s))
        except Exception:  # noqa: BLE001 — degrade to heuristic, never crash the loop
            pass
    return s[:max_chars] + f"… ({len(s)} chars total)"


def _condense_args(args: dict) -> dict:
    """Shrink large argument VALUES for the prompt, keeping keys and small values
    intact (#2). The twin of :func:`_summarize` for results: a long string value
    collapses to a ``<N chars>`` marker, a heavy nested value to ``<N chars dict|list>``;
    short scalars (a path, a query, a number) render verbatim so the card still shows
    WHAT it was called with. Non-destructive — operates on a copy for rendering only;
    the card's real ``args`` (the recurrence gate's evidence) are untouched.
    """
    out: dict = {}
    for k, v in args.items():
        if isinstance(v, str):
            out[k] = f"<{len(v)} chars>" if len(v) > ARG_VALUE_INLINE_CAP else v
        elif isinstance(v, (dict, list)):
            blob = json.dumps(v, default=str)
            out[k] = v if len(blob) <= ARG_VALUE_INLINE_CAP else f"<{len(blob)} chars {type(v).__name__}>"
        else:
            out[k] = v
    return out


@dataclass
class ToolCard:
    """One episodic invocation card (the provenance record of a tool call)."""

    tool: str
    args: dict
    status: str
    summary: str
    intent: str = ""
    cid: Optional[str] = None
    ts: float = field(default_factory=time.time)
    vec: Optional[List[float]] = None  # cached embedding for relevance ranking (D2)
    action_vec: Optional[List[float]] = None  # cached embedding of the ACTION (tool+args) for recurrence (D4)

    @classmethod
    def from_result(cls, tool: str, args: Any, result: Any, intent: str = "",
                    summarizer: Optional[Any] = None) -> "ToolCard":
        """Build a card from a raw tool call + result, deriving status + summary.

        When the kernel **offloaded** a heavy system-tool result (ADR-0039 returns
        a ``result_cid`` and clears the payload — surfaced here as a result dict with
        a ``cid`` key), the card keeps the ``cid`` and a short marker instead of the
        (absent) payload; the full content is one :func:`resolve_content` drill-down
        away. Otherwise the result is summarized inline (heuristic or opt-in LLM).
        """
        cid = result.get("cid") if isinstance(result, dict) else None
        if cid:
            summary = "(large result offloaded to content store; drill down by cid)"
        else:
            summary = _summarize(result, summarizer=summarizer)
        return cls(
            tool=tool,
            args=args if isinstance(args, dict) else {"value": args},
            status=_status_of(result),
            summary=summary,
            intent=intent,
            cid=cid,
        )

    def render(self) -> str:
        """Render the card as a COMPLETED step (#2).

        Three deliberate choices, all to fix the bloat seen in Langfuse:
        - ``<step …>`` (not ``<tool …>``) frames it as an action ALREADY executed
          this run — provenance/history, not a call being proposed now.
        - the **result leads** — the outcome (``ok: wrote cambrian1.txt``) is the
          signal; it must not be buried under a wall of args.
        - the **args are condensed** — a multi-KB ``content`` payload collapses to a
          ``<N chars>`` marker instead of re-rendering verbatim every turn.
        """
        intent = f" intent={self.intent!r}" if self.intent else ""
        cid = f"\n  <cid>{self.cid}</cid>" if self.cid else ""
        args = json.dumps(_condense_args(self.args), default=str)
        return (
            f"<step action={self.tool!r} status={self.status!r}{intent}>\n"
            f"  <result>{self.summary}</result>\n"
            f"  <args>{args}</args>{cid}\n"
            f"</step>"
        )


@dataclass
class TextEntry:
    """A non-tool working-memory entry (workspace, memory, note).

    Small blocks render verbatim. A large block (R7) is *offloaded*: ``content``
    keeps the full text in-process for drill-down, ``cid`` points at its
    ContentStore copy, and ``summary`` is the gist the prompt carries instead of
    the full payload — so a re-injected definition does not re-bloat every step.

    ``pinned`` entries are always kept by :meth:`WorkingMemory.assemble` regardless
    of relevance/recency — the channel for a **verbal reflection** (ADR-0052), a
    Reflexion-style lesson the agent must keep in front of it across every later
    attempt, not lose to bounding the way an ordinary note would.
    """

    content: str
    vec: Optional[List[float]] = None  # cached embedding for relevance ranking (D2)
    cid: Optional[str] = None          # set when offloaded (R7); drill down via resolve_content
    summary: Optional[str] = None      # gist rendered in place of a large offloaded payload
    pinned: bool = False               # ADR-0052: always kept in assembly (reflections)

    def render(self) -> str:
        if self.summary is None:
            return self.content
        # Offloaded form: a compact pointer the agent can resolve by cid. It is told
        # the content exists ("here is the gist; full text at <cid>") instead of
        # carrying the whole block every turn.
        ref = f" cid={self.cid!r}" if self.cid else ""
        return f"<offloaded chars={len(self.content)}{ref}>\n{self.summary}\n</offloaded>"


# ──────────────────────────────────────────────────────────────────────
# v2 trajectory: a typed Step entry (ADR-0052 + agent-prompting/SUMMARY.md)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """A typed trajectory entry — the unit the v2 rich XML renderer consumes.

    The ``kind`` field maps to the agent action. Other fields are optional and
    depend on the kind. The renderer is forgiving: missing fields render as empty.

    Kinds:
        memory_query   — ``query``, ``results`` (list of LTM hit dicts)
        tool_call      — ``tool``, ``args``, ``body``/``summary``,
                         optional ``cid``+``chars`` (offload),
                         ``status`` ('ok'|'error'|'denied')
        describe_tool  — ``tool``, ``body`` (the tool spec)
        use_skill      — ``skill``, ``body`` (the skill instructions)
        find_tools     — ``body`` (the matched tools list)
        find_skills    — ``body`` (the matched skills list)
        resolve_cid    — fetch a previously-offloaded body.
                         ``cid`` = the cid the model should USE NEXT
                                  (the new cid for offload; the original
                                  cid for inline)
                         ``resolved_from`` = the cid the model ASKED to resolve
                                            (the original cid, for both modes)
                         ``mode`` = 'offload' (default) | 'inline'
                         For offload: ``chars`` = the body size
                         For inline: ``body`` = the inlined content
        reflection     — ``body`` (the verbal lesson; ADR-0052)
        note           — ``body`` (a runtime note: veto, dedup, truncation)
        vetoed         — ``body`` (a hard-veto marker)
        truncated      — ``body`` (a truncation marker)
    """
    kind: str
    n: int = 0  # 1-based ordinal in the trajectory (assigned by add_step)
    # memory_query fields
    query: str = ""
    results: list = field(default_factory=list)
    # tool_call / describe_tool / use_skill fields
    tool: str = ""
    skill: str = ""
    args: dict = field(default_factory=dict)
    body: str = ""       # inline result body / tool spec / skill instructions / inlined resolve_cid body
    summary: str = ""    # tool result summary (the gist)
    intent: str = ""
    # offload fields
    cid: str = ""
    chars: int = 0
    # status
    status: str = "ok"   # 'ok' | 'error' | 'denied' | 'empty' | 'vetoed' | 'truncated'
    # resolve_cid fields
    resolved_from: str = ""  # the cid the model asked to resolve (original)
    mode: str = ""          # for resolve_cid: 'offload' (default) | 'inline'


def render_step_xml(s) -> str:
    """Render one Step as a clear, self-describing XML block (v2).

    Addresses the 4 trajectory issues:
      1. memory_query steps include ``query="..."`` as an XML attribute — the
         agent can see WHAT was asked, not just what was returned.
      2. tool_call steps wrap args in a ``<call>`` block with one child per
         named arg — the call-site is parseable, not a raw JSON dump.
      3. Offloaded content surfaces ``<result offloaded_cid="..." chars="N">``
         with a ``<summary>`` child — the agent knows what's behind the cid
         without re-fetching.
      4. Each step is a separate ``<step>`` with explicit ``n``, ``type``,
         ``status`` attributes. Reflections and notes are their own tags.
    """
    n = getattr(s, "n", 0) or 0
    kind = s.kind

    if kind == "memory_query":
        attrs = (
            f' type="memory_query"'
            f' query="{_esc(s.query)}"'
            f' status="{_esc(s.status)}"'
        )
        body = _render_memory_results(s.results or [])
        return f"<step n={n!r}{attrs}>\n{body}\n</step>"

    if kind == "tool_call":
        attrs = (
            f' type="tool_call"'
            f' tool={_esc(s.tool)!r}'
            f' status={_esc(s.status)!r}'
        )
        call_xml = _render_call_xml(s.tool, s.args or {})
        # Threshold above which we offload a body instead of inlining it. The
        # rule seen in the field: a 3475-char file (a small HTML page) was
        # truncated to 400 chars by ``_summarize``; the model kept re-reading
        # because it never saw the full content. We now show the full body
        # inline up to 4000 chars; above that, the model is expected to use
        # ``{"$cid": "..."}`` for tool calls or ``resolve_cid`` for reading.
        _INLINE_BODY_CAP = 4000
        body = s.body or s.summary
        if s.cid:
            # Kernel offloaded the body (rare — the kernel's Read tool returns
            # the full body inline; offload is a kernel-side decision).
            cid_hint = (
                f"    <cid_hint>use this cid as {{\"$cid\": \"{s.cid}\"}} in the next "
                f"tool_call arg — the kernel resolves it; do NOT re-emit the body. "
                f"If you need the body in YOUR context (to quote, summarize, or "
                f"reason about it), use {{\"action\": \"resolve_cid\", \"cid\": \"{s.cid}\", "
                f"\"as\": \"offload\"|\"inline\"}}</cid_hint>\n"
            )
            result_xml = (
                f"  <result status={_esc(s.status)!r}"
                f" offloaded_cid={_esc(s.cid)!r}"
                f" chars={s.chars!r}>\n"
                f"    <summary>{_esc(s.summary or '(no summary — re-call or describe_tool)')}</summary>\n"
                + cid_hint
                + f"  </result>"
            )
        elif body and len(body) > _INLINE_BODY_CAP:
            # Body is large but no kernel offload. Show a truncated summary +
            # an offload hint (the model can read the full body via resolve_cid
            # or pass it as $cid — but we don't have a kernel cid yet, so the
            # hint is the ACTION the model can take). This is the path that
            # closed the field's "model re-reads 4 times" gap: when the body is
            # small enough, the model sees it all; when it's not, the model is
            # told it's been truncated and has an escape hatch.
            preview = body[:200] + "…"
            offload_hint = (
                f"    <note>body truncated for the prompt ({s.chars or len(body)} chars total). "
                f"Options: (a) use the body as-is for reasoning in this turn; (b) if you need "
                f"to ACT on the full body, emit a tool_call whose args include the content; "
                f"the kernel will accept it inline. The recurrence gate blocks re-reading "
                f"the same path with the same args.</note>\n"
            )
            result_xml = (
                f"  <result status={_esc(s.status)!r}"
                f" chars={s.chars or len(body)!r}>\n"
                f"    <summary>{_esc(s.summary or preview)}</summary>\n"
                f"    <body_preview>{_esc(preview)}</body_preview>\n"
                + offload_hint
                + f"  </result>"
            )
        else:
            # Inline the full body. This is the default — and the path the
            # field's re-read loop came from missing. Small results (the
            # common case) go here: the model sees everything and does NOT
            # need to call again.
            result_xml = (
                f"  <result status={_esc(s.status)!r}>\n"
                f"    <body>{_esc(body)}</body>\n"
                f"  </result>"
            )
        return (
            f"<step n={n!r}{attrs}>\n"
            f"  <call>{call_xml}\n  </call>\n"
            f"{result_xml}\n"
            f"</step>"
        )

    if kind == "describe_tool":
        return (
            f"<step n={n!r} type='describe_tool' tool={_esc(s.tool)!r}>\n"
            f"  <tool_spec>{_esc(s.body)}</tool_spec>\n"
            f"</step>"
        )

    if kind == "use_skill":
        return (
            f"<step n={n!r} type='use_skill' skill={_esc(s.skill)!r}>\n"
            f"  <skill_instructions>{_esc(s.body)}</skill_instructions>\n"
            f"</step>"
        )

    if kind in ("find_tools", "find_skills"):
        return (
            f"<step n={n!r} type={_esc(kind)!r}>\n"
            f"  <body>{_esc(s.body)}</body>\n"
            f"</step>"
        )

    if kind == "resolve_cid":
        # v2 resolve_cid (ADR-0048 residual): the agent asks the kernel to
        # fetch a previously-offloaded body. Two modes:
        #  - 'offload' (default): the body is re-offloaded and a NEW cid is
        #    returned. The Step.cid is the NEW cid (the one to use next);
        #    Step.resolved_from is the original.
        #  - 'inline': the body is inlined into the trajectory so the model
        #    can read it. Step.cid is the original; Step.body is the content.
        if s.mode == "inline":
            return (
                f"<step n={n!r} type='resolve_cid' cid={_esc(s.cid)!r}"
                f" mode='inline' resolved_from={_esc(s.resolved_from or s.cid)!r}>\n"
                f"  <summary>{_esc(s.summary or f'{len(s.body)} chars inlined; full body in your context.')}</summary>\n"
                f"  <body>{_esc(s.body)}</body>\n"
                f"</step>"
            )
        # offload (default)
        return (
            f"<step n={n!r} type='resolve_cid' cid={_esc(s.cid)!r}"
            f" mode='offload' resolved_from={_esc(s.resolved_from or s.cid)!r}"
            f" chars={s.chars!r}>\n"
            f"  <summary>{_esc(s.summary or 're-offloaded as a new cid; body is NOT in your context.')}</summary>\n"
            f"  <cid_hint>use this cid as {{\"$cid\": \"{s.cid}\"}} in the next tool_call arg — the kernel resolves it; do NOT re-emit the body</cid_hint>\n"
            f"</step>"
        )

    if kind == "reflection":
        # Reflections are NOT <step> blocks — they are their own tag (ADR-0052).
        return (
            f"<reflection n={n!r}>\n"
            f"  {_esc(s.body or s.summary)}\n"
            f"</reflection>"
        )

    if kind == "note":
        return f"<note>{_esc(s.body or s.summary)}</note>"

    if kind == "vetoed":
        return f"<note>VETOED: {_esc(s.body or s.summary)}</note>"

    if kind == "truncated":
        return f"<note>TRUNCATED: {_esc(s.body or s.summary)}</note>"

    # Fallback — round-trip the body.
    return f"<step n={n!r} type={_esc(kind)!r}>{_esc(s.body or s.summary)}</step>"


def _render_call_xml(tool: str, args: dict) -> str:
    """Render a tool_call's args as a ``<call>`` block with named children.

    Each arg becomes ``<argname>value</argname>`` so the model can see the
    call site as named parameters, not a raw JSON dump.
    """
    if not args:
        return ""
    lines = []
    for k, v in args.items():
        if isinstance(v, (dict, list)):
            val = json.dumps(v, default=str)
        elif isinstance(v, bool):
            val = "true" if v else "false"
        elif v is None:
            val = "null"
        else:
            val = str(v)
        lines.append(f"    <{_esc(k)}>{_esc(val)}</{_esc(k)}>")
    return "\n".join(lines)


# ── ADR-0048 Amendment A1: recall provenance + freshness (source monitoring) ──
# A recalled fact reaches the prompt with WHO wrote it, WHEN, and HOW FRESH it is, so
# the agent can discount its own machine echoes (D9) and re-verify stale facts (D10).
# The kernel stamps the raw facts (source_agent/session_id at write time, ADR-0048 D1;
# _created_at/_last_accessed_at/_activation_strength folded in at recall time in
# querymemory.go). The SDK only RENDERS them — no value-routing is decided here, so
# the threshold-based freshness label is Zero-Hardcode-clean (it states a fact about a
# memory; it never gates which agent or task runs — ADR-0048 A1 D10).

# Freshness label buckets. SDK v1 constants: the kernel owns λ for query-time ranking;
# the rendered label is a coarse re-verify hint for the agent, not a ranking term.
_FRESHNESS_STALE_ACTIVATION = 0.05  # stored activation below this → "stale" (any age)
_FRESHNESS_AGING_DAYS = 30          # older than this and not stale → "aging"


def _memory_meta(r) -> dict:
    """Parse a recall result's metadata into a dict. It arrives from the kernel as a
    JSON string (``MemoryResult.metadata``); accept a pre-parsed dict too. ``{}`` when
    absent or unparseable."""
    meta = r.get("metadata", {}) if isinstance(r, dict) else getattr(r, "metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _age_days(meta: dict) -> Optional[int]:
    """Whole days since the fact was last touched (``_last_accessed_at``, falling back
    to ``_created_at``). ``None`` when neither timestamp is present/parseable."""
    raw = meta.get("_last_accessed_at") or meta.get("_created_at")
    if not raw:
        return None
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - t).total_seconds() // 86400))


def _freshness_label(meta: dict) -> str:
    """Coarse re-verify hint (D10): ``stale`` | ``aging`` | ``fresh``, or ``''`` when the
    kernel sent no temporal/activation fields. Low stored activation ⇒ stale; otherwise
    old ⇒ aging; else fresh."""
    act = meta.get("_activation_strength")
    age = _age_days(meta)
    if act is None and age is None:
        return ""
    try:
        act_f = float(act) if act is not None else None
    except (ValueError, TypeError):
        act_f = None
    if act_f is not None and act_f < _FRESHNESS_STALE_ACTIVATION:
        return "stale"
    if age is not None and age > _FRESHNESS_AGING_DAYS:
        return "aging"
    return "fresh"


def memory_provenance_attrs(r) -> str:
    """The attribute string for a recalled fact's ``<memory>`` tag (ADR-0048 A1).

    ``source='LTM'`` always leads; then, only when present: ``author`` + ``session``
    (D9 — kernel-stamped, non-forgeable; a ``System`` source renders as ``author='system'``
    so the agent can discount its own auto-recorded step echoes), ``written`` + ``age`` +
    ``freshness`` (D10), then ``content_cid`` last. Absent fields are omitted entirely —
    no empty ``author=''``."""
    meta = _memory_meta(r)
    attrs = ["source='LTM'"]
    src = str(meta.get("source_agent", "") or "")
    if src:
        author = "system" if src == "System" else src
        attrs.append(f"author={_esc(author)!r}")
    sid = str(meta.get("session_id", "") or "")
    if sid:
        attrs.append(f"session={_esc(sid)!r}")
    written = str(meta.get("_created_at", "") or "")
    if written:
        attrs.append(f"written={_esc(written[:10])!r}")  # YYYY-MM-DD
    fresh = _freshness_label(meta)
    if fresh:
        age = _age_days(meta)
        if age is not None:
            attrs.append(f"age={_esc(f'{age}d')!r}")
        attrs.append(f"freshness={_esc(fresh)!r}")
    cid = str(meta.get("content_cid", "") or "")
    if cid:
        attrs.append(f"content_cid={_esc(cid)!r}")
    return " ".join(attrs)


def _render_memory_results(results) -> str:
    """Render a memory_query's results as ``<memory>`` children with provenance +
    freshness attributes (ADR-0048 A1)."""
    if not results:
        return "  <note>no relevant memory found</note>"
    items = []
    for r in results:
        text = r.get("text", r) if isinstance(r, dict) else getattr(r, "text", r)
        items.append(f"  <memory {memory_provenance_attrs(r)}>{_esc(text)}</memory>")
    return "\n".join(items)


def action_text(tool: str, args: Any) -> str:
    """Canonical string form of an action (tool + sorted args) — the unit the
    recurrence gate (D4) hashes and embeds to detect re-issued failing calls."""
    try:
        a = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        a = str(args)
    return f"{tool} {a}"


_TOOL_SPEC_RE = re.compile(r"<tool_spec name=['\"]?([^'\">\s]+)")


def _tool_spec_name(content: str) -> Optional[str]:
    """Extract the tool name from a ``<tool_spec name='X'>`` block (R4 detection),
    matching the form ``react.describe_tool`` injects. Returns ``None`` otherwise."""
    m = _TOOL_SPEC_RE.search(content)
    return m.group(1) if m else None


def _normalize_text(s: str) -> str:
    """Whitespace-collapsed, lowercased form for near-duplicate comparison."""
    return " ".join(s.split()).lower()


def _near_duplicate(a: str, b: str, *, min_tokens: int = 30, overlap: float = 0.85) -> bool:
    """True when two normalized text blocks are substantially the same content.

    Conservative by design: only LARGE blocks (≥ ``min_tokens``) that are either
    fully contained or share ≥ ``overlap`` of the shorter block's tokens are treated
    as duplicates — so small distinct notes are never collapsed, but the
    prior-step-result that arrives reordered via seed + recall + a later memory_query
    (the triplication observed in <LTMContext>) is caught.
    """
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short:
        return False
    if short in long:
        return True
    st = set(short.split())
    if len(st) < min_tokens:
        return False
    lt = set(long.split())
    return len(st & lt) / len(st) >= overlap


def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Cosine similarity; -1.0 when either vector is missing/zero (ranks last)."""
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


class WorkingMemory:
    """Ordered, typed working-memory buffers assembled into the prompt context.

    For 0041-01 this is an ordered list of entries whose ``assemble()`` joins
    their rendered forms with newlines — behavior-equivalent to the old
    ``"\\n".join(scratch)`` — except tool results are :class:`ToolCard`s carrying
    provenance. Later slices add bounding/relevance ranking (0041-03) over the
    same entries.
    """

    def __init__(self, embed_fn: Optional[Callable[[str], Optional[List[float]]]] = None,
                 cap: int = DEFAULT_CAP,
                 offload_fn: Optional[Callable[[str], Optional[str]]] = None,
                 text_inline_cap: int = DEFAULT_TEXT_INLINE_CAP) -> None:
        self._entries: List[Any] = []
        self._embed_fn = embed_fn
        self._cap = cap
        # R7: text -> cid offloader. When wired (needs a ContentStore WRITE path —
        # only GetContextNode exists today), a large text block is replaced in the
        # prompt by {gist + cid}, full text drill-downable. SAFE default: when None,
        # large blocks render verbatim (no collapse without a retrieval path, so a
        # later step that needs the full text — e.g. write_file — is never starved).
        self._offload_fn = offload_fn
        self._text_inline_cap = text_inline_cap
        # v2: monotonic step counter; assigned by add_step.
        self._next_n = 1

    def _embed(self, text: str) -> Optional[List[float]]:
        if self._embed_fn is None:
            return None
        try:
            v = self._embed_fn(text)
            return v or None
        except Exception:  # noqa: BLE001 — degrade to recency ranking, never crash
            return None

    def add_text(self, content: str) -> None:
        """Append a working-memory text block (workspace seed, memory result, note).

        R7: a block over ``text_inline_cap`` is offloaded when an ``offload_fn`` is
        wired — the prompt then carries {gist + cid} and the full text is drill-down
        by cid. Without an offloader (no ContentStore write path yet) it renders
        verbatim, so the full text is never dropped from a step that may need it.
        Relevance is always embedded on the FULL content (richer ranking signal).
        """
        vec = self._embed(content)
        if self._offload_fn is not None and len(content) > self._text_inline_cap:
            cid = None
            try:
                cid = self._offload_fn(content)
            except Exception:  # noqa: BLE001 — degrade to verbatim, never crash the loop
                cid = None
            if cid:
                self._entries.append(TextEntry(
                    content, vec=vec, cid=cid, summary=_summarize(content, max_chars=240)))
                return
        self._entries.append(TextEntry(content, vec=vec))

    def add_reflection(self, text: str) -> None:
        """Append a verbal reflection (ADR-0052) as a PINNED working-memory entry.

        A Reflexion-style lesson the loop extracts after the recurrence gate's
        hard-veto (``react.run_think``): *why* the repeated action failed and *what*
        to do differently. It is ``pinned`` so :meth:`assemble` keeps it every
        subsequent turn regardless of bounding — the structural gate (which counts
        duplicates) gains a *verbal* memory the LLM reads before its next attempt.
        Embedded for relevance like any block; never offloaded (reflections are short).
        """
        clean = (text or "").strip()
        if not clean:
            return
        # v2: a reflection is a Step(kind="reflection", pinned=True). The renderer
        # produces ``<reflection n="N">...</reflection>``.
        self.add_step(kind="reflection", body=clean, pinned=True)

    def add_tool_card(self, card: ToolCard) -> None:
        """Append a tool invocation card to the episodic buffer (embed once here)."""
        if card.vec is None:
            card.vec = self._embed(card.summary or card.tool)  # relevance vector (D2)
        # Action vector (D4) — only failed cards are compared by the recurrence gate,
        # so only they need it; saves an embed per successful call.
        if card.status in ("error", "denied") and card.action_vec is None:
            card.action_vec = self._embed(action_text(card.tool, card.args))
        self._entries.append(card)

    def add_step(self, kind: str, pinned: bool = False, **fields) -> None:
        """Append a typed Step entry to the working memory (v2).

        The kind field maps to the agent action. Other fields are passed to
        the Step dataclass. The rich XML renderer in assemble() consumes these.
        The step is auto-numbered (``n`` attribute on the rendered <step>) — the
        caller-supplied ``n`` (if any) is ignored in favor of the monotonic counter
        so the renderer's ``n`` is always sequential and unique within a run.
        """
        # Drop caller-supplied ``n``; the auto-counter is authoritative.
        fields.pop("n", None)
        step = Step(kind=kind, n=self._next_n, **fields)
        # Mark reflections as pinned via a side-channel: a TextEntry with pinned=True
        # would be needed for the assemble() pin logic, so we use a thin wrapper.
        # The simplest path: if pinned, also stash a TextEntry mirror that the
        # assemble() pin logic will find. This keeps backward compat with the
        # text-entry pin mechanism while letting new code use the typed API.
        self._next_n += 1
        if pinned:
            # The reflection block goes into the buffer twice: once as a Step (for
            # the new rich renderer) and once as a TextEntry with pinned=True (for
            # the assemble() pin logic). Cheap (~200 bytes) and keeps the
            # pin logic unified.
            block = f"<reflection>{fields.get('body', '').strip()}</reflection>"
            self._entries.append(TextEntry(
                block, vec=self._embed(block), pinned=True,
            ))
        self._entries.append(step)

    @property
    def cards(self) -> List[ToolCard]:
        """The episodic buffer — the ordered tool invocation cards."""
        return [e for e in self._entries if isinstance(e, ToolCard)]

    def assemble(self, intent_vec: Optional[List[float]] = None, numbered: bool = False) -> str:
        """Render a BOUNDED, relevance-ranked + pinned slice into the prompt (D2).

        Up to ``cap`` entries reach the prompt — not the whole history — turning
        per-turn prompt growth from O(N²) into ~O(k). Mandatory **pins** are always
        kept regardless of relevance: any **open-failure** card (so the recurrence
        gate keeps its evidence) and the **most recent** entry. Remaining slots are
        filled by embedding-cosine similarity to the current intent; with no
        ``intent_vec`` / embedder it degrades to recency. Selected entries render in
        chronological order. (The originating task is always present in <Task>, so it
        is not an entry here.)

        ``numbered`` (legacy): prefixes each kept entry with its ordinal
        ("1. ... 2. ...") — the v1 markdown-numbered form. Default (``numbered=False``)
        uses the v2 rich XML with ``n`` attributes on each ``<step>`` block.

        v2 renderer: each entry is rendered through :func:`render_entry_xml`, which
        dispatches to :func:`render_step_xml` for :class:`Step` entries and to the
        existing legacy renderers for :class:`ToolCard` / :class:`TextEntry` (for
        backward compat with code that still uses the old API).
        """
        n = len(self._entries)
        if n <= self._cap:
            idxs = self._dedup(list(range(n)))
            return self._render(idxs, self._collapse_superseded(idxs), numbered)

        pinned = {n - 1}  # the most recent entry
        for i, e in enumerate(self._entries):
            if isinstance(e, ToolCard) and e.status in ("error", "denied"):
                pinned.add(i)  # open-failure cards
            elif isinstance(e, TextEntry) and e.pinned:
                pinned.add(i)  # ADR-0052: verbal reflections — never bounded out

        rankable = [i for i in range(n) if i not in pinned]
        if intent_vec:
            rankable.sort(key=lambda i: _cosine(getattr(self._entries[i], "vec", None), intent_vec),
                          reverse=True)
        else:
            rankable.sort(reverse=True)  # recency fallback: most recent first

        keep = set(pinned)
        for i in rankable:
            if len(keep) >= self._cap:
                break
            keep.add(i)
        idxs = self._dedup(sorted(keep))
        return self._render(idxs, self._collapse_superseded(idxs), numbered)

    def _dedup(self, idxs: List[int]) -> List[int]:
        """Drop later TEXT entries whose content is largely already present in an
        earlier kept entry (ADR-0041 D3). The same prior-step result reaches the
        prompt via the workspace seed, the mandatory recall, AND later memory_query
        results — 2–3 near-identical copies that bloat <LTMContext>. Keeps the first
        occurrence (chronological), preserving order. Tool cards are never deduped —
        each invocation is distinct provenance the recurrence gate may need.

        v2: dispatch on entry type — Step entries render through the v2 rich
        renderer; ToolCard / TextEntry use their own .render() (legacy path).
        """
        kept_norms: List[str] = []
        out: List[int] = []
        for i in idxs:
            e = self._entries[i]
            if isinstance(e, ToolCard):
                out.append(i)
                continue
            # v2: Step entries don't have a .render() — go through the dispatcher.
            if isinstance(e, Step):
                # Skip dedup for typed Steps — each is a discrete action round.
                # The dedup target is the legacy ``<memory>`` triplication (seed +
                # recall + later memory_query) which the v2 typed path already
                # deduplicates by design (each round is a discrete Step).
                out.append(i)
                continue
            norm = _normalize_text(e.render())
            if norm and any(_near_duplicate(norm, prev) for prev in kept_norms):
                continue
            kept_norms.append(norm)
            out.append(i)
        return out

    def _collapse_superseded(self, idxs: List[int]) -> dict:
        """Replace entries made moot by a LATER event with one-line markers (R7/D7).

        Returns ``{idx: marker}`` applied at render time only — the buffer (and the
        recurrence gate's evidence) is untouched. Two cases:

        - **R4 (consumed tool spec):** a ``<tool_spec name='X'>`` block whose tool X
          was subsequently called collapses **immediately** (the spec did its job).
        - **R5 (superseded failures):** failed cards for an action that a later card
          *succeeded* collapse, with **one turn of hysteresis** — kept while the
          success is the most-recent entry (just happened), collapsed once older.
        """
        overrides: dict = {}
        cards = [(i, self._entries[i]) for i in idxs if isinstance(self._entries[i], ToolCard)]
        last = idxs[-1] if idxs else -1

        # R4 — a fetched spec whose tool was later called.
        for i in idxs:
            e = self._entries[i]
            if isinstance(e, TextEntry):
                name = _tool_spec_name(e.content)
                if name and any(c.tool == name and ci > i for ci, c in cards):
                    overrides[i] = f"<note>spec for {name!r} consumed</note>"

        # R5 — failed cards superseded by a later success of the same action.
        succeeded: dict = {}  # action signature -> earliest success index
        for ci, c in cards:
            if c.status == "ok":
                succeeded.setdefault(action_text(c.tool, c.args), ci)
        for ci, c in cards:
            if c.status in ("error", "denied"):
                succ_idx = succeeded.get(action_text(c.tool, c.args))
                # collapse only once the success is no longer the most-recent entry
                # (one-turn hysteresis — let the agent reason about the failure first).
                if succ_idx is not None and succ_idx > ci and succ_idx != last:
                    overrides[ci] = f"<note>{c.tool!r} failed, then succeeded</note>"
        return overrides

    def _render(self, idxs, overrides: Optional[dict] = None, numbered: bool = False) -> str:
        """Render the kept slice through the v2 rich XML renderer.

        For :class:`Step` entries: ``render_step_xml(s)`` (the new format).
        For :class:`ToolCard` entries: ``render_entry_xml(card)`` converts the
        card to a Step and renders (one call per card, cheap).
        For :class:`TextEntry` entries: ``render_entry_xml(entry)`` — renders
        inline for plain notes, ``<offloaded>`` for offloaded payloads.
        For superseded entries (R4/R5): the override marker is used as-is.
        """
        overrides = overrides or {}
        items = []
        for i in idxs:
            if i in overrides:
                items.append(overrides[i])
                continue
            entry = self._entries[i]
            items.append(render_entry_xml(entry, numbered=numbered))
        if numbered:
            return "\n".join(f"{n}. {s}" for n, s in enumerate(items, 1))
        return "\n".join(items)

    def __len__(self) -> int:
        return len(self._entries)


def render_entry_xml(entry: Any, numbered: bool = False) -> str:
    """Render any working-memory entry as a v2 rich XML block.

    Dispatches:
      - :class:`Step` → :func:`render_step_xml`
      - :class:`ToolCard` → convert to Step and render (preserves the v1
        tool-call provenance in the new format)
      - :class:`TextEntry` → render inline for plain notes, ``<offloaded>`` for
        offloaded payloads, ``<reflection>`` for pinned reflections
    """
    if isinstance(entry, Step):
        return render_step_xml(entry)
    if isinstance(entry, ToolCard):
        s = Step(
            kind="tool_call",
            n=0,
            tool=entry.tool,
            args=entry.args,
            status=entry.status,
            summary=entry.summary,
            body=entry.summary,
            cid=entry.cid or "",
            chars=0,
        )
        return render_step_xml(s)
    if isinstance(entry, TextEntry):
        if entry.pinned:
            # Pinned TextEntry is a reflection — render with the reflection tag.
            clean = (entry.content or "").strip()
            if clean.startswith("<reflection>") and clean.endswith("</reflection>"):
                lesson = clean[len("<reflection>"):-len("</reflection>")].strip()
                return f"<reflection>\n  {_esc(lesson)}\n</reflection>"
            return f"<reflection>\n  {_esc(clean)}\n</reflection>"
        if entry.summary is not None:
            # Offloaded TextEntry: a compact pointer the agent can pass as
            # ``{"$cid": "<cid>"}`` to a tool_call (ADR-0048 #1) — the kernel
            # resolves the cid to the full body. The hint must be IN the block
            # (not in a separate instruction) so the model can't miss it.
            # The hint also mentions the ``resolve_cid`` escape hatch for the
            # rare case where the agent needs the body in its own context.
            ref = f" cid={entry.cid!r}" if entry.cid else ""
            cid_hint = (
                f"\n  <cid_hint>use this cid as {{\"$cid\": \"{entry.cid}\"}} in a tool_call "
                f"arg — the kernel resolves it; do NOT re-emit the body. "
                f"If you need the body in YOUR context, use {{\"action\": "
                f"\"resolve_cid\", \"cid\": \"{entry.cid}\", \"as\": \"offload\"|\"inline\"}}</cid_hint>"
                if entry.cid else ""
            )
            return (
                f"<offloaded chars={len(entry.content)}{ref}>\n"
                f"{entry.summary}\n"
                f"</offloaded>{cid_hint}"
            )
        return entry.content
    # Fallback.
    return str(entry)


def resolve_content(entry: Any, substrate: Any) -> Optional[str]:
    """Drill down: fetch the full content an offloaded entry summarized, by CID.

    Works for any entry carrying a ``cid`` — a :class:`ToolCard` whose heavy result
    the kernel offloaded (ADR-0041 D3) OR an offloaded :class:`TextEntry` (R7). The
    prompt carries only ``{summary + cid}``; when the agent actually needs the full
    text it resolves the ``cid`` through the existing ``substrate.get_context_node``
    (ADR-0022). Returns ``None`` for an inline entry (no ``cid``) or on any failure —
    never raises into the loop.
    """
    cid = getattr(entry, "cid", None)
    if not cid or substrate is None or not hasattr(substrate, "get_context_node"):
        return None
    try:
        node = substrate.get_context_node(cid)
    except Exception:  # noqa: BLE001 — degrade, never crash
        return None
    data = getattr(node, "data", None) if node is not None else None
    if data is None:
        return None
    return data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
