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

Since 2026-11-15 (owner decision, T-20261105-remove-the-legacy-workingmemory-engine-and)
this module holds only the entry types and renderers the ADR-0139 D3 context
engine (``context_engine.py``) builds on: the entry-count, cosine-ranked
``WorkingMemory`` buffer, its bounded assembly and the R4/R5 de-duplication were
deleted with the legacy engine.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from .helpers import _esc  # v2: XML-escape for the rich trajectory renderer
from .payload_fence import (
    PROVENANCE_RECALLED_MEMORY,
    PROVENANCE_TOOL_OUTPUT,
    fence_untrusted,
)

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
        if cid and result.get("preview"):
            # The kernel described the offloaded result (owner decision A2,
            # 2026-11-12): size, type, shape and a preview reach the text path too.
            bits = [str(b) for b in (
                f"{result.get('size_bytes')} bytes" if result.get("size_bytes") is not None else "",
                result.get("content_type") or "", result.get("shape") or "") if b]
            summary = (f"(large result offloaded: {', '.join(bits)}; preview: "
                       f"{result.get('preview')} … {result.get('note') or ''})")
        elif cid:
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
    # ADR-0139 D3 / ADR-0046 D3: a pinned Step is an INSTRUCTION, not evidence. It is
    # kept by every assemble() for the rest of the run, never offloaded and never
    # evicted, and it does not consume the trajectory's entry cap. Used for a loaded
    # skill's body and its grant note (WorkingMemory.add_skill).
    pinned: bool = False


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
            f' query="{_esc(short_query(s.query))}"'
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
                f"    <summary>{fence_tool_output(s.summary, s.tool) if s.summary else '(no summary — re-call or describe_tool)'}</summary>\n"
                + cid_hint
                + "  </result>"
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
                f"    <summary>{fence_tool_output(s.summary or preview, s.tool)}</summary>\n"
                f"    <body_preview>{fence_tool_output(preview, s.tool)}</body_preview>\n"
                + offload_hint
                + "  </result>"
            )
        else:
            # Inline the full body. This is the default — and the path the
            # field's re-read loop came from missing. Small results (the
            # common case) go here: the model sees everything and does NOT
            # need to call again.
            result_xml = (
                f"  <result status={_esc(s.status)!r}>\n"
                f"    <body>{fence_tool_output(body, s.tool)}</body>\n"
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
        # A loaded skill (WorkingMemory.add_skill). The FULL body, always: a skill is
        # a procedure the model must follow step by step, and a head cut of it is a
        # different, wrong procedure (ADR-0139 Phase 0).
        return (
            f"<step n={n!r} type='use_skill' skill={_esc(s.skill)!r}>\n"
            f"  <skill_instructions>\n{_esc(s.body)}\n  </skill_instructions>\n"
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
    ``freshness`` (D10), ``document`` + ``section`` + ``ledger`` (the source document,
    its section path and the recall's ledger entry, when the kernel sent them), then
    ``content_cid`` last. Absent fields are omitted entirely — no empty ``author=''``."""
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
    # Where the fact comes from and the ledger entry to cite (the kernel records
    # every recall in the run ledger: T-20261109-chat-agent-answers-a-published-rule).
    # An organisation's rule or figure is stated with its document and section.
    doc = str(meta.get("_source_document", "") or "")
    if doc:
        attrs.append(f"document={_esc(doc)!r}")
    sec = str(meta.get("_section_path", "") or "")
    if sec:
        attrs.append(f"section={_esc(sec)!r}")
    ledger = str(meta.get("_ledger_ref", "") or "")
    if ledger:
        attrs.append(f"ledger={_esc(ledger)!r}")
    cid = str(meta.get("content_cid", "") or "")
    if cid:
        attrs.append(f"content_cid={_esc(cid)!r}")
    return " ".join(attrs)


def memory_origin(r) -> str:
    """Who wrote a recalled memory, for its fence's ``origin``: the kernel-stamped
    writing principal (``source:<connector>:<id>`` for an ingested document, webhook
    or poller body; an agent id; ``system``), or ``''`` when the kernel sent none."""
    src = str(_memory_meta(r).get("source_agent", "") or "")
    return "system" if src == "System" else src


def render_recalled_memory(r, indent: str = "") -> str:
    """One recalled memory as it reaches a prompt (ENG-94): the ``<memory>`` element
    with its provenance + freshness attributes (ADR-0048 A1), its TEXT inside the
    kernel's nonce payload fence (ADR-0063) marked ``provenance="recalled_memory"``
    with the writer as origin. A recalled document that says "ignore previous
    instructions" is one JSON-encoded data line the model is told is not
    instructions, and a fake closing tag in it cannot end the region. Every recall
    surface (memory_query steps and tool turns, the seed recall, the chat agent's
    seed recall) renders through here."""
    text = r.get("text", r) if isinstance(r, dict) else getattr(r, "text", r)
    fenced = fence_untrusted(text, PROVENANCE_RECALLED_MEMORY, memory_origin(r))
    return f"{indent}<memory {memory_provenance_attrs(r)}>\n{fenced}\n{indent}</memory>"


def fence_tool_output(body: Any, tool: str) -> str:
    """A tool's output as it reaches a prompt (ENG-94): inside the kernel's payload
    fence, ``provenance="tool_output"``, ``origin="tool:<name>"``. A preview or a
    summary of a tool's output is its output too and goes through here."""
    return fence_untrusted(body, PROVENANCE_TOOL_OUTPUT, f"tool:{tool}" if tool else "tool")


def _render_memory_results(results) -> str:
    """Render a memory_query's results as ``<memory>`` children with provenance +
    freshness attributes (ADR-0048 A1), each text fenced (ENG-94)."""
    if not results:
        return "  <note>no relevant memory found</note>"
    return "\n".join(render_recalled_memory(r, indent="  ") for r in results)


# The longest query text a memory_query rendering carries in an attribute. The query
# is OUR text (the model's, or the task text for the seed recall), not retrieved
# data: echoing a whole task objective into a data-marked region made it read as
# instructions planted in evidence, and the model refused its own task
# (T-20261106-agent-refuses-its-whole-legitimate-task-when).
QUERY_ATTR_MAX_CHARS = 80


def short_query(query: Any) -> str:
    """The query as one line, at most :data:`QUERY_ATTR_MAX_CHARS` chars (an
    ellipsis marks a cut). Not escaped: callers escape for their attribute."""
    q = " ".join(str(query or "").split())
    if len(q) > QUERY_ATTR_MAX_CHARS:
        q = q[: QUERY_ATTR_MAX_CHARS - 1].rstrip() + "…"
    return q


EMPTY_RECALL_NOTE = "<note>no relevant memory found</note>"


def render_memory_query_reply(query: Any, results) -> str:
    """The reply to an explicit memory_query (the native tool turn): the hits as
    tagged ``<documents>`` (data) with a capped query attribute, or, for no hits, a
    one-line note and no documents block."""
    if not results:
        return EMPTY_RECALL_NOTE
    return (f"<documents source='memory' query={_esc(short_query(query))!r}>\n"
            f"{_render_memory_results(results)}\n</documents>")


def action_text(tool: str, args: Any) -> str:
    """Canonical string form of an action (tool + sorted args) — the unit the
    recurrence gate (D4) hashes and embeds to detect re-issued failing calls."""
    try:
        a = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        a = str(args)
    return f"{tool} {a}"




def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Cosine similarity; -1.0 when either vector is missing/zero (ranks last)."""
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


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
