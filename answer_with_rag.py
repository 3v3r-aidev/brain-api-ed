from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import re
import os

from semantic_search import (
    search,
    search_meetings,
    search_in_date_window,
)

# Optional OpenAI client for answer synthesis
_USE_OAI = False
try:
    from openai import OpenAI
    _client = OpenAI()
    _USE_OAI = True
except Exception:
    try:
        import openai  # legacy
        _USE_OAI = True
    except Exception:
        _USE_OAI = False

# ─────────────────────────────────────────────────────────────
# Date-window resolution from user query (extended)
# ─────────────────────────────────────────────────────────────

_MONTHS = "(january|february|march|april|may|june|july|august|september|october|november|december)"
_Q_PAT = re.compile(r"\bq([1-4])\s*(?:[-/ ]?\s*)?(20\d{2})\b", re.I)  # Q1 2025 / Q3-2025 / Q4/2026

def _today():
    return datetime.now()

def _week_bounds(dt: datetime):
    # Monday=0 .. Sunday=6
    start = dt - timedelta(days=dt.weekday())
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end

def _month_bounds(dt: datetime):
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year+1, month=1, day=1) - timedelta(seconds=1)
    else:
        end = start.replace(month=start.month+1, day=1) - timedelta(seconds=1)
    return start, end

def _quarter_bounds(q: int, year: int):
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    sm, sd = starts[q]
    start = datetime(year, sm, sd)
    if q < 4:
        em, ed = starts[q + 1]
        end = datetime(year, em, ed) - timedelta(seconds=1)
    else:
        end = datetime(year, 12, 31, 23, 59, 59)
    return start, end

def resolve_date_window_from_query(q: str) -> Optional[Tuple[datetime, datetime]]:
    qlow = q.lower()

    # Today / yesterday
    if "today" in qlow:
        dt = _today()
        s = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        e = dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return s, e
    if "yesterday" in qlow:
        dt = _today() - timedelta(days=1)
        s = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        e = dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return s, e

    # This / last week
    if "this week" in qlow:
        return _week_bounds(_today())
    if "last week" in qlow:
        last = _today() - timedelta(days=7)
        return _week_bounds(last)

    # This / last month
    if "this month" in qlow:
        return _month_bounds(_today())
    if "last month" in qlow:
        last_month = (_today().replace(day=1) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return _month_bounds(last_month)

    # Explicit month name + year (e.g., "September 2025")
    m = re.search(rf"\b{_MONTHS}\s+20\d{{2}}\b", qlow)
    if m:
        parts = m.group(0).split()
        month_name, year = parts[0], int(parts[1])
        month_num = ["january","february","march","april","may","june",
                     "july","august","september","october","november","december"].index(month_name) + 1
        start = datetime(year, month_num, 1)
        if month_num == 12:
            end = datetime(year+1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, month_num+1, 1) - timedelta(seconds=1)
        return start, end

    # Quarter reference (Q1 2025)
    qm = _Q_PAT.search(qlow)
    if qm:
        qn = int(qm.group(1))
        yr = int(qm.group(2))
        return _quarter_bounds(qn, yr)

    # Date range "from ... to ..."
    mr = re.search(r"\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b", qlow)
    if mr:
        s = datetime.fromisoformat(mr.group(1))
        e = datetime.fromisoformat(mr.group(2)).replace(hour=23, minute=59, second=59)
        return s, e

    return None

# ─────────────────────────────────────────────────────────────
# Answer builder
# ─────────────────────────────────────────────────────────────
def _synth_answer(question: str, hits: List[Tuple[int, float, Dict]]) -> str:
    """
    If OpenAI available, synthesize a short answer grounded in the retrieved chunks,
    otherwise return a compact extract with sources.
    """
    if _USE_OAI and os.getenv("OPENAI_API_KEY"):
        # Build context
        contexts = []
        for _, _, m in hits[:6]:
            snippet = (m.get("text_preview") or m.get("text") or "")[:800]
            source = m.get("filename") or m.get("title") or "source"
            contexts.append(f"[{source}]: {snippet}")
        sys_prompt = (
            "You are an analyst assistant. Answer concisely using ONLY the provided context. "
            "If the answer is in a Reminder, respect its validity window. "
            "Cite the filename inline like [source]."
        )
        user_msg = f"Question: {question}\n\nContext:\n" + "\n\n".join(contexts)

        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("ANSWER_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback: simple extract with sources
    lines = ["Top matches:"]
    for i, (_, _, m) in enumerate(hits[:5], 1):
        title = m.get("title") or m.get("filename") or m.get("id") or "source"
        folder = m.get("folder", "")
        vf = m.get("valid_from"); vt = m.get("valid_to")
        md = m.get("meeting_date")
        meta_line = []
        if folder:
            meta_line.append(folder)
        if md:
            meta_line.append(f"meeting_date={md}")
        if vf or vt:
            meta_line.append(f"valid={vf}..{vt or '∞'}")
        preview = (m.get("text_preview") or m.get("text") or "").strip().replace("\n"," ")
        lines.append(f"{i}. {title} ({', '.join(meta_line)}) — {preview[:180]}…")
    return "\n".join(lines)

def answer(
    question: str,
    k: int = 7,
    chat_history: Optional[List[Dict]] = None,
    restrict_to_meetings: bool = False,
    use_rag: bool = True
) -> str:
    """
    Main entry used by Streamlit app.
    - If a date window is detected, use date-aware search so Reminders qualify via validity window.
    - If restrict_to_meetings=True, prefer meetings in rerank.
    """
    window = resolve_date_window_from_query(question)
    hits: List[Tuple[int, float, Dict]]

    if window:
        s, e = window
        hits = search_in_date_window(question, s, e, k=k)
    elif restrict_to_meetings:
        hits = search_meetings(question, k=k)
    else:
        hits = search(question, k=k)

    if not hits:
        return "No relevant documents or reminders found in the specified scope."

    if use_rag:
        return _synth_answer(question, hits)

    # Non-RAG: return compact source list
    return _synth_answer(question, hits)

