"""Bounded reads of long content: one page, or a search, never the whole body.

Owner decision A3 (2026-11-12, T-20261112-worker-resolve-cid-inline-has-no-size-cap-a):
re-opening content (``resolve_cid``) returns ONE window of it, with a note saying how
much there is, which part is shown, and the exact call for the next page or a search.
A ``led_`` id is paged by the kernel's ``ledger_get`` (the one read pathway for ledger
entries); this module pages everything else the loop can re-open itself (a context-
engine compacted output, a content-store cid) the same way, with the same note shape.

Documented practice it follows: Claude Code's Read returns a partial view with a note
saying how to read more (offset/limit); Anthropic's "Writing tools for agents"
recommends pagination/range selection plus "helpful instructions" when truncating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Search results kept per call; a broader query is told to narrow down.
MAX_FIND_MATCHES = 20
#: Characters of context shown around each match.
FIND_SNIPPET_CHARS = 240


def page_count(total: int, size: int) -> int:
    size = max(int(size), 1)
    return max((total + size - 1) // size, 1) if total else 1


def window(body: str, page: Optional[int], size: int) -> Tuple[str, int, int, int, int]:
    """Page ``page`` (1-based, clamped) of ``body`` in pages of ``size`` characters.

    Returns ``(text, page, pages, start, end)`` with ``start``/``end`` as 0-based
    character offsets (end exclusive). Python ``str`` slicing is per code point, so a
    cut never splits a character."""
    size = max(int(size), 1)
    pages = page_count(len(body), size)
    k = 1 if not page or page < 1 else min(int(page), pages)
    start = (k - 1) * size
    end = min(start + size, len(body))
    return body[start:end], k, pages, start, end


def find(body: str, query: str, size: int, *, max_matches: int = MAX_FIND_MATCHES,
         snippet: int = FIND_SNIPPET_CHARS) -> Tuple[List[Dict[str, Any]], int]:
    """Case-insensitive substring search inside ONE body.

    Returns ``(matches, total)``: up to ``max_matches`` ``{"page", "offset",
    "snippet"}`` (page numbers in pages of ``size``) and the total match count."""
    q = (query or "").strip().lower()
    if not q:
        return [], 0
    hay = body.lower()
    size = max(int(size), 1)
    out: List[Dict[str, Any]] = []
    total, pos = 0, hay.find(q)
    half = max(snippet // 2, 20)
    while pos >= 0:
        total += 1
        if len(out) < max_matches:
            a, b = max(pos - half, 0), min(pos + len(q) + half, len(body))
            out.append({"page": pos // size + 1, "offset": pos,
                        "snippet": ("…" if a else "") + body[a:b] + ("…" if b < len(body) else "")})
        pos = hay.find(q, pos + max(len(q), 1))
    return out, total


def page_note(cid: str, total: int, start: int, end: int, page: int, pages: int,
              verb: str = "resolve_cid", key: str = "cid") -> str:
    """``N chars total, showing a–b (page k of P); next: …; search: …`` — the same
    shape as the kernel's ledger_get note."""
    parts = [f"{total} chars total, showing {start + 1}–{end} (page {page} of {pages})"]
    if page < pages:
        parts.append(f"next: {verb} {key}='{cid}' page={page + 1}")
    else:
        parts.append("this is the last page")
    parts.append(f"search: {verb} {key}='{cid}' find='<text>'")
    return "; ".join(parts)


def find_note(cid: str, query: str, total_matches: int, shown: int, pages: int,
              verb: str = "resolve_cid", key: str = "cid") -> str:
    if total_matches == 0:
        return (f"no match for {query!r} in {pages} page(s); try another word, or read "
                f"page by page: {verb} {key}='{cid}' page=1")
    more = (f"; {total_matches - shown} more not listed, narrow the search"
            if total_matches > shown else "")
    return (f"{total_matches} match(es) for {query!r}{more}; read a page with "
            f"{verb} {key}='{cid}' page=<n>")


def read_window(cid: str, body: str, size: int, page: Optional[int] = None,
                query: str = "") -> Dict[str, Any]:
    """One bounded answer for a re-opened body: a page, or the matches of ``query``."""
    total = len(body)
    pages = page_count(total, size)
    if query and query.strip():
        matches, n = find(body, query, size)
        return {"cid": cid, "find": query, "match_count": n, "matches": matches,
                "pages": pages, "total_chars": total,
                "note": find_note(cid, query, n, len(matches), pages)}
    text, k, pages, start, end = window(body, page, size)
    return {"cid": cid, "text": text, "page": k, "pages": pages, "total_chars": total,
            "note": page_note(cid, total, start, end, k, pages)}
