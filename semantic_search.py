import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import os
import re

import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

# Embedding for query (OpenAI new SDK preferred, fallback to legacy)
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    try:
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except Exception:
        pass

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

INDEX_PATH = Path("embeddings/faiss.index")
META_PATH = Path("embeddings/metadata.pkl")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def _embed_query_client(text: str) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)

def _embed_query_legacy(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)  # type: ignore
    return np.asarray(resp["data"][0]["embedding"], dtype=np.float32)

def embed_query(text: str) -> np.ndarray:
    arr = _embed_query_client(text) if _use_client else _embed_query_legacy(text)
    if arr.shape != (EMBED_DIM,):
        # allow legacy clients to return lists
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"Unexpected embedding shape {arr.shape}")
    return arr

def load_resources():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing FAISS index or metadata. Run embed_and_store.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    index, metadata = load_resources()
    qvec = embed_query(query).reshape(1, -1)
    # L2 index by default
    D, I = index.search(qvec, max(k, 200))
    out: List[Tuple[int, float, Dict]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        out.append((int(idx), float(dist), metadata.get(int(idx), {})))
    return out

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip().replace("Z", "")
    # try full ISO first
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    # fallback YYYY-MM-DD
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None

def _query_tags(query: str) -> List[str]:
    """Broader tag extraction: any token length >=3 is considered a tag candidate."""
    return [t for t in _TOKEN_RE.findall(query.lower()) if len(t) >= 3]

def rerank(results: List[Tuple[int, float, Dict]], query: str, prefer_meetings: bool = False, prefer_recent: bool = False) -> List[Tuple[int,float,Dict]]:
    qtags = set(_query_tags(query))
    now = datetime.now()

    rescored: List[Tuple[float, Tuple[int, float, Dict]]] = []
    for rid, dist, meta in results:
        # Base score: smaller L2 distance -> larger score
        base = -float(dist)

        folder = str(meta.get("folder", "")).lower()
        tags = {t.strip().lower() for t in (meta.get("tags") or []) if t}
        tag_overlap = len(qtags.intersection(tags))
        # Scale tag bonus so it materially affects score
        tag_bonus = tag_overlap * 1000.0

        # Meetings recency bonus (keep it significant but bounded)
        meet_bonus = 0.0
        mdate = _parse_iso(meta.get("meeting_date"))
        if prefer_recent and mdate:
            # ~7.3e6 today with *10; keep smaller to avoid drowning others
            meet_bonus = mdate.toordinal() * 5.0

        # Optional folder preference for meetings
        meeting_folder_priority = 0.0
        if prefer_meetings and folder == "meetings":
            meeting_folder_priority = 1_000_000_000.0  # strong preference when explicitly requesting meetings

        # Reminders: modest folder bonus + strong validity priority
        reminder_bonus = 0.0
        if folder == "reminders":
            vfrom = _parse_iso(meta.get("valid_from"))
            vto = _parse_iso(meta.get("valid_to"))
            valid_now = True
            if vfrom and now < vfrom:
                valid_now = False
            if vto and now > vto:
                valid_now = False
            # Give a strong positive priority to valid reminders so they don't lose to meetings recency
            reminder_bonus += (9_000_000.0 if valid_now else -1_000_000.0)

        score = base + tag_bonus + meet_bonus + meeting_folder_priority + reminder_bonus
        rescored.append((score, (rid, dist, meta)))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in rescored]

def search_meetings(query: str, k: int = 5, prefer_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    raw = search(query, k=max(k, 100))
    re_ranked = rerank(raw, query=query, prefer_meetings=True, prefer_recent=prefer_recent)
    return re_ranked[:k]

def filter_by_date_range(results: List[Tuple[int, float, Dict]], start: datetime, end: datetime) -> List[Tuple[int, float, Dict]]:
    """Keep results where:
       - meeting_date ∈ [start, end], OR
       - reminder validity window [valid_from, valid_to] overlaps [start, end]
    """
    def _overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
        return a_start <= b_end and b_start <= a_end

    kept: List[Tuple[int, float, Dict]] = []
    for rid, dist, meta in results:
        md = _parse_iso(meta.get("meeting_date"))
        if md and (start <= md <= end):
            kept.append((rid, dist, meta))
            continue

        vf = _parse_iso(meta.get("valid_from"))
        vt = _parse_iso(meta.get("valid_to")) or datetime.max
        if vf and _overlap(vf, vt, start, end):
            kept.append((rid, dist, meta))

    return kept

def rerank_for_recency(results: List[Tuple[int, float, Dict]], query: str, favor_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    return rerank(results, query=query, prefer_meetings=False, prefer_recent=favor_recent)

def search_in_date_window(query: str, start: datetime, end: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    pool = search(query, k=max(k, 200))
    windowed = filter_by_date_range(pool, start, end)
    if not windowed:
        return []
    return rerank_for_recency(windowed, query=query)[:k]

if __name__ == "__main__":
    hits = search_meetings("hr hiring policy last month", k=5)
    for i, (vid, dist, meta) in enumerate(hits, 1):
        print(f"{i}. dist={dist:.4f} file={meta.get('filename')} valid_from={meta.get('valid_from')} valid_to={meta.get('valid_to')}")
        print((meta.get("text_preview", "") or "")[:160], "\n---")

