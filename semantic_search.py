import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import os
import re

import numpy as np

# Optional FAISS for similarity search
import faiss  # type: ignore

# Embedding backends:
# - OpenAI (remote) or
# - sentence-transformers (local)
_BACKEND_CLIENT = None
_ST_MODEL = None

EMBEDDINGS_DIR = Path("embeddings")
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
META_PATH = EMBEDDINGS_DIR / "metadata.pkl"
CONF_PATH = EMBEDDINGS_DIR / "config.json"

# ------------------------------
# Config helpers
# ------------------------------
DEFAULT_CONF = {
    "backend": "openai",            # "openai" | "local"
    "model": "text-embedding-3-small",
    "dim": 1536,
    "metric": "ip"                  # "ip" | "l2"
}

def _load_conf() -> Dict:
    if CONF_PATH.exists():
        try:
            return json.loads(CONF_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_CONF.copy()

_CONF = _load_conf()

# ------------------------------
# Embedding functions
# ------------------------------
def _init_openai():
    global _BACKEND_CLIENT
    if _BACKEND_CLIENT is None:
        try:
            from openai import OpenAI  # type: ignore
            _BACKEND_CLIENT = OpenAI()
        except Exception as e:
            raise RuntimeError(f"OpenAI client not available: {e}")

def _init_st():
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _ST_MODEL = SentenceTransformer(_CONF.get("model", "all-MiniLM-L6-v2"))
        except Exception as e:
            raise RuntimeError(f"SentenceTransformer not available: {e}")

def embed_query(text: str) -> np.ndarray:
    """
    Embed a query string using the same backend & model as used for the index.
    """
    backend = _CONF.get("backend", "openai")
    if backend == "local":
        _init_st()
        vec = _ST_MODEL.encode([text], normalize_embeddings=(_CONF.get("metric") == "ip"))
        return np.asarray(vec[0], dtype=np.float32)
    else:
        _init_openai()
        try:
            resp = _BACKEND_CLIENT.embeddings.create(model=_CONF.get("model"), input=text)
            vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
            # Normalize for cosine/IP if index metric is IP
            if _CONF.get("metric") == "ip":
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            return vec
        except Exception as e:
            # legacy fallback
            import openai  # type: ignore
            openai.api_key = os.getenv("OPENAI_API_KEY")
            resp = openai.Embedding.create(model=_CONF.get("model"), input=text)
            vec = np.asarray(resp["data"][0]["embedding"], dtype=np.float32)
            if _CONF.get("metric") == "ip":
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            return vec

# ------------------------------
# Load FAISS + metadata
# ------------------------------
def load_resources():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Missing embeddings. Run the Refresh Data step first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata: Dict[int, Dict] = pickle.load(f)
    return index, metadata

# ------------------------------
# Utilities
# ------------------------------
def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Try full ISO first
        return datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        pass
    # Try common fallback YYYY-MM-DD
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None

_TAG_RE = re.compile(r"[A-Za-z0-9_]+")

def _query_tags(query: str) -> List[str]:
    """
    Broader tag extraction from the user query.
    Any alphanumeric token length >=3 qualifies as a candidate tag.
    """
    return [t for t in _TAG_RE.findall(query.lower()) if len(t) >= 3]

# ------------------------------
# Core search
# ------------------------------
def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    index, metadata = load_resources()
    qvec = embed_query(query).reshape(1, -1)
    # search in FAISS
    D, I = index.search(qvec, max(k, 100))
    out: List[Tuple[int, float, Dict]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        out.append((int(idx), float(dist), metadata.get(int(idx), {})))
    return out[:k]

# ------------------------------
# Date-aware filtering
# ------------------------------
def filter_by_date_range(
    results: List[Tuple[int, float, Dict]],
    start: datetime,
    end: datetime
) -> List[Tuple[int, float, Dict]]:
    """
    Keep hits where either:
      - meeting_date ∈ [start, end], OR
      - reminder validity window [valid_from, valid_to] overlaps [start, end]
    """
    def _overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
        return a_start <= b_end and b_start <= a_end

    kept: List[Tuple[int, float, Dict]] = []
    for rid, dist, meta in results:
        # Meetings path
        md = _parse_iso(meta.get("meeting_date"))
        if md and (start <= md <= end):
            kept.append((rid, dist, meta))
            continue

        # Reminders path: use validity window
        vf = _parse_iso(meta.get("valid_from"))
        vt = _parse_iso(meta.get("valid_to")) or datetime.max
        if vf and _overlap(vf, vt, start, end):
            kept.append((rid, dist, meta))
    return kept

# ------------------------------
# Reranking
# ------------------------------
def rerank(
    results: List[Tuple[int, float, Dict]],
    query: str,
    prefer_meetings: bool = False,
    prefer_recent: bool = False
) -> List[Tuple[int, float, Dict]]:
    """
    Combine FAISS distance with light heuristics:
      - Tag overlap bonus
      - Prefer meetings toggle
      - Modest reminders bonus
      - Validity window bonus for reminders
      - Optional recency boost for meetings
    """
    qtags = set(_query_tags(query))
    now = datetime.now()
    metric = _CONF.get("metric", "ip")

    rescored: List[Tuple[float, Tuple[int, float, Dict]]] = []
    for rid, dist, meta in results:
        # Convert FAISS distance to a base score where higher is better
        # IP gives similarity (higher is better). L2 gives distance (smaller is better).
        if metric == "l2":
            base = 1.0 / (1.0 + dist)
        else:
            base = dist

        folder = str(meta.get("folder", "")).lower()
        tags = {t.strip().lower() for t in (meta.get("tags") or []) if t}
        # Tag overlap (from query tokens and metadata tags)
        tag_overlap = len(qtags.intersection(tags))
        tag_bonus = 0.05 * tag_overlap

        # Meetings recency
        meet_bonus = 0.0
        mdate = _parse_iso(meta.get("meeting_date"))
        if prefer_recent and mdate:
            # Use ordinal scaled to a small positive value
            meet_bonus += (mdate.toordinal() / 365000.0)  # tiny boost

        # Folder preference
        folder_bonus = 0.0
        if prefer_meetings and folder == "meetings":
            folder_bonus += 0.2
        if folder == "reminders":
            folder_bonus += 0.15  # modest lift so reminders aren’t buried

        # Reminder current validity bonus
        validity_bonus = 0.0
        if folder == "reminders":
            vfrom = _parse_iso(meta.get("valid_from"))
            vto = _parse_iso(meta.get("valid_to"))
            valid_now = True
            if vfrom and now < vfrom:
                valid_now = False
            if vto and now > vto:
                valid_now = False
            validity_bonus = (0.2 if valid_now else -0.4)

        score = base + tag_bonus + folder_bonus + meet_bonus + validity_bonus
        rescored.append((score, (rid, dist, meta)))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in rescored]

def rerank_for_recency(results: List[Tuple[int, float, Dict]], query: str) -> List[Tuple[int, float, Dict]]:
    return rerank(results, query=query, prefer_meetings=False, prefer_recent=True)

# ------------------------------
# Convenience entry points
# ------------------------------
def search_meetings(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    pool = search(query, k=max(k, 50))
    return rerank(pool, query=query, prefer_meetings=True, prefer_recent=True)[:k]

def search_in_date_window(query: str, start: datetime, end: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    pool = search(query, k=max(k, 200))
    windowed = filter_by_date_range(pool, start, end)
    if not windowed:
        return []
    return rerank_for_recency(windowed, query=query)[:k]

