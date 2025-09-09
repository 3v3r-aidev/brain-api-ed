# =========================
# 🧠 AI CEO Assistant + 🗂️ Chat Hub (Emoji + Diagnostics, Single File)
# =========================
# 👉 Save as: chat_ceo.py
# ✅ Core deps: streamlit, pandas
# 🧩 Works even if file_parser.py, embed_and_store.py, or answer_with_rag.py are missing.
# 🛠️ Features:
#   - 🗂️ Multi-conversation sidebar (create/search/select/rename/delete)
#   - 💬 Chat UI (stores to SQLite)
#   - 📜 DB-backed history + CSV export
#   - 🚀 One-click Parse ➜ Embed pipeline
#   - 🧪 Diagnostics panel (env/DB/status/errors)
#
# If you later add:
#   - file_parser.main()
#   - embed_and_store.main()
#   - answer_with_rag.answer()
# this UI will automatically use them.

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 🧩 Optional imports (safe if missing)
# ─────────────────────────────────────────────────────────────
_last_exception = None
def _try_import(name, attr=None):
    """Import helper that never crashes the app; records the last exception."""
    global _last_exception
    try:
        mod = __import__(name)
        return getattr(mod, attr) if attr else mod
    except Exception as e:
        _last_exception = f"Import error for {name}: {e}"
        return None

file_parser = _try_import("file_parser")                         # expects file_parser.main()
embed_and_store = _try_import("embed_and_store")                 # expects embed_and_store.main()
rag_answer = _try_import("answer_with_rag", "answer")            # expects answer_with_rag.answer()

# ─────────────────────────────────────────────────────────────
# ⚙️ App config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧠 AI CEO Assistant", page_icon="🧠", layout="wide")

# ─────────────────────────────────────────────────────────────
# 🗄️ SQLite (embedded, hardened for Streamlit threads)
# ─────────────────────────────────────────────────────────────
DB_PATH = Path("chats.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts TEXT NOT NULL,
    FOREIGN KEY(conv_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
"""

def _connect():
    # check_same_thread=False avoids issues under Streamlit's threaded runtime
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@contextmanager
def _conn():
    conn = _connect()
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def db_init():
    with _conn() as conn:
        for stmt in SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)

def db_new_conversation(title: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations(title, created_at, updated_at) VALUES(?,?,?)",
            (title.strip() or "Untitled", now, now),
        )
        return cur.lastrowid

def db_rename_conversation(conv_id: int, title: str):
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as conn:
        conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                     (title.strip() or "Untitled", now, conv_id))

def db_delete_conversation(conv_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

def db_list_conversations(search: str = ""):
    q = "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    params = ()
    if search.strip():
        q = "SELECT id, title, created_at, updated_at FROM conversations WHERE title LIKE ? ORDER BY updated_at DESC"
        params = (f"%{search.strip()}%",)
    with _conn() as conn:
        return conn.execute(q, params).fetchall()

def db_add_message(conv_id: int, role: str, content: str, ts: str | None = None) -> int:
    ts = ts or datetime.now().isoformat(timespec="seconds")
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages(conv_id, role, content, ts) VALUES(?,?,?,?)",
            (conv_id, role, content, ts),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conv_id))
        return cur.lastrowid

def db_get_messages(conv_id: int):
    with _conn() as conn:
        return conn.execute(
            "SELECT id, role, content, ts FROM messages WHERE conv_id=? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()

# ─────────────────────────────────────────────────────────────
# 🗃️ Legacy JSON (kept for compatibility; optional)
# ─────────────────────────────────────────────────────────────
HIST_PATH = Path("chat_history.json")
REFRESH_PATH = Path("last_refresh.txt")

def load_history():
    try:
        if HIST_PATH.exists():
            return json.loads(HIST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"⚠️ History load error: {e}")
    return []

def save_history(history):
    try:
        HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"⚠️ History save error: {e}")

def save_refresh_time():
    try:
        REFRESH_PATH.write_text(datetime.now().strftime("%b-%d-%Y %I:%M %p"))
    except Exception as e:
        st.warning(f"⚠️ Refresh timestamp save error: {e}")

def load_refresh_time():
    try:
        if REFRESH_PATH.exists():
            return REFRESH_PATH.read_text(encoding="utf-8")
    except Exception as e:
        st.warning(f"⚠️ Refresh timestamp read error: {e}")
    return "N/A"

# ─────────────────────────────────────────────────────────────
# 🔐 Minimal login (replace with your own if needed)
# ─────────────────────────────────────────────────────────────
USERNAME = "admin123"
PASSWORD = "BestOrg123@#"

def login():
    st.title("🔐 Login to AI CEO Assistant")
    with st.form("login_form"):
        u = st.text_input("👤 Username")
        p = st.text_input("🔑 Password", type="password")
        if st.form_submit_button("➡️ Login"):
            if u == USERNAME and p == PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# ─────────────────────────────────────────────────────────────
# 🏁 App init
# ─────────────────────────────────────────────────────────────
db_init()

# Seed an active conversation once
if "active_conv" not in st.session_state:
    try:
        st.session_state["active_conv"] = db_new_conversation("🧾 Default Conversation")
    except Exception as e:
        _last_exception = f"DB seed error: {e}"

# ─────────────────────────────────────────────────────────────
# 🧪 Diagnostics (sidebar)
# ─────────────────────────────────────────────────────────────
with st.sidebar.expander("🧪 Diagnostics", expanded=True):
    st.code({
        "cwd": os.getcwd(),
        "db_file_exists": DB_PATH.exists(),
        "db_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0,
        "has_file_parser": bool(file_parser),
        "has_embed_and_store": bool(embed_and_store),
        "has_rag_answer": bool(rag_answer),
        "last_exception": _last_exception,
    }, language="json")

# Auth gate
if not st.session_state["authenticated"]:
    login()
    st.stop()

# ─────────────────────────────────────────────────────────────
# 🗂️ Sidebar — Conversations (Chat Hub)
# ─────────────────────────────────────────────────────────────
st.sidebar.subheader("🗂️ Conversations")

with st.sidebar.expander("➕ New Conversation"):
    new_title = st.text_input("📝 Title", placeholder="e.g., Buyer agreement review")
    if st.button("✅ Create"):
        try:
            cid = db_new_conversation(new_title or "Untitled")
            st.session_state["active_conv"] = cid
            st.rerun()
        except Exception as e:
            st.error(f"❌ Create conversation failed: {e}")

search = st.sidebar.text_input("🔎 Search")
try:
    convs = db_list_conversations(search)
except Exception as e:
    convs = []
    st.sidebar.error(f"❌ List conversations failed: {e}")

if not convs:
    st.sidebar.caption("ℹ️ No conversations yet.")
else:
    for cid, title, _, _ in convs:
        if st.sidebar.button(title or f"Conversation {cid}", key=f"conv_sel_{cid}"):
            st.session_state["active_conv"] = cid
            st.rerun()

acid = st.session_state.get("active_conv")
if acid:
    st.sidebar.markdown("---")
    current_title = next((t for i, t, _, _ in convs if i == acid), "Untitled") if convs else "Untitled"
    new_name = st.sidebar.text_input("✏️ Rename", value=current_title)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Save Name", key="conv_rename"):
            try:
                db_rename_conversation(acid, new_name or "Untitled")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Rename failed: {e}")
    with c2:
        if st.button("🗑️ Delete", key="conv_delete"):
            try:
                db_delete_conversation(acid)
                remain = db_list_conversations("")
                st.session_state["active_conv"] = remain[0][0] if remain else db_new_conversation("🧾 Default Conversation")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Delete failed: {e}")

# ─────────────────────────────────────────────────────────────
# 🧭 Sidebar — App Navigation
# ─────────────────────────────────────────────────────────────
mode = st.sidebar.radio("🧭 Navigation", ["💬 New Chat", "📜 View History", "🔁 Refresh Data"], index=0)
st.sidebar.markdown("---")
st.sidebar.checkbox("🗂️ Limit to meeting docs only", value=False, key="limit_meetings")
st.sidebar.caption("💡 Tip: start a message with **REMINDER:** to save a reminder text file in `./reminders`.")

# ─────────────────────────────────────────────────────────────
# 🤖 Safe answer() wrapper (uses stub if missing)
# ─────────────────────────────────────────────────────────────
def _safe_answer(query: str, k: int = 7, chat_history=None, restrict_to_meetings: bool = False) -> str:
    """Call your RAG answer() if available; otherwise return a safe stub."""
    global _last_exception
    if rag_answer is None:
        return f"(stub) You asked: {query}"
    try:
        return rag_answer(query, k=k, chat_history=chat_history, restrict_to_meetings=restrict_to_meetings)
    except TypeError:
        # Handle different signatures gracefully
        try:
            return rag_answer(query)
        except Exception as e:
            _last_exception = f"rag_answer call failed: {e}"
            return f"(stub) You asked: {query}"
    except Exception as e:
        _last_exception = f"rag_answer error: {e}"
        return f"(stub) You asked: {query}"

# ─────────────────────────────────────────────────────────────
# 💬 Page: New Chat
# ─────────────────────────────────────────────────────────────
if mode == "💬 New Chat":
    st.title("💬 New Chat")
    # Render legacy JSON turns (optional)
    legacy = load_history()
    for turn in legacy:
        role = turn.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(f"🗨️ [{turn.get('timestamp', 'N/A')}]  \n{turn.get('content', '')}")

    user_msg = st.chat_input("✍️ Type your question or add a REMINDER…")
    if user_msg:
        # 1) 📝 REMINDER shortcut
        try:
            if user_msg.strip().lower().startswith("reminder:"):
                body = re.sub(r"^reminder:\s*", "", user_msg.strip(), flags=re.I)
                title_hint = (body.split("\n", 1)[0] or "Reminder")[:60]
                folder = Path("reminders"); folder.mkdir(exist_ok=True)
                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{re.sub(r'[^a-zA-Z0-9._ -]+','_',title_hint)}.txt"
                path = folder / fname
                path.write_text(body, encoding="utf-8")
                st.success(f"💾 Reminder saved: `{path}`")
        except Exception as e:
            st.warning(f"⚠️ Reminder save failed: {e}")

        # 2) 🧾 Append to legacy JSON
        now = datetime.now().strftime("%b-%d-%Y %I:%M%p")
        legacy.append({"role": "user", "content": user_msg, "timestamp": now})
        save_history(legacy)

        # 3) 🧱 Mirror into SQLite
        try:
            db_add_message(st.session_state["active_conv"], "user", user_msg, ts=now)
        except Exception as e:
            st.warning(f"⚠️ DB add (user) failed: {e}")

        # 4) 🤖 Get assistant reply
        with st.chat_message("assistant"):
            st.markdown("🧠 Thinking…")
            try:
                reply = _safe_answer(
                    user_msg,
                    k=7,
                    chat_history=legacy,
                    restrict_to_meetings=st.session_state["limit_meetings"],
                )
            except Exception as e:
                reply = f"❌ Error: {e}"
            ts = datetime.now().strftime("%b-%d-%Y %I:%M%p")
            st.markdown(f"🧾 [{ts}]  \n{reply}")

        # 5) 💽 Save assistant turn (JSON + DB)
        legacy.append({"role": "assistant", "content": reply, "timestamp": ts})
        save_history(legacy)
        try:
            db_add_message(st.session_state["active_conv"], "assistant", reply, ts=ts)
        except Exception as e:
            st.warning(f"⚠️ DB add (assistant) failed: {e}")

        st.rerun()

# ─────────────────────────────────────────────────────────────
# 📜 Page: View History (DB-backed)
# ─────────────────────────────────────────────────────────────
elif mode == "📜 View History":
    st.title("📜 Conversation History")
    cid = st.session_state.get("active_conv")
    try:
        msgs = db_get_messages(cid) if cid else []
    except Exception as e:
        msgs = []
        st.error(f"❌ Load messages failed: {e}")

    if not msgs:
        st.info("ℹ️ No messages in this conversation yet.")
    else:
        for _, role, content, ts in msgs:
            who = "👤 You" if role == "user" else "🤖 Assistant" if role == "assistant" else "⚙️ System"
            st.markdown(f"**{who} | [{ts}]**  \n{content}")

        df = pd.DataFrame(msgs, columns=["id", "role", "content", "timestamp"])
        st.download_button(
            label="⬇️ Download Conversation as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"conversation_{cid}.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────
# 🔁 Page: Refresh Data (single button: Parse ➜ Embed & Store)
# ─────────────────────────────────────────────────────────────
elif mode == "🔁 Refresh Data":
    st.title("🔁 Refresh Data")
    st.write("📥 Parse & 🧩 Embed your documents so the assistant answers with the latest context.")

    # One-click pipeline: Parse -> Embed & Store
    if st.button("🚀 Parse & Embed (One Click)"):
        ok = True

        # Step 1: Parse
        with st.spinner("📥 Parsing documents..."):
            if file_parser and hasattr(file_parser, "main"):
                try:
                    file_parser.main()
                    st.success("✅ Parsing complete.")
                except Exception as e:
                    st.error(f"❌ Parsing failed: {e}")
                    ok = False
            else:
                st.warning("⚠️ file_parser.main() not found.")
                ok = False

        # Step 2: Embed & Store (only if parse succeeded)
        if ok:
            with st.spinner("🧩 Creating embeddings & storing..."):
                if embed_and_store and hasattr(embed_and_store, "main"):
                    try:
                        embed_and_store.main()
                        save_refresh_time()
                        st.success("✅ Embeddings stored.")
                    except Exception as e:
                        st.error(f"❌ Embedding failed: {e}")
                        ok = False
                else:
                    st.warning("⚠️ embed_and_store.main() not found.")
                    ok = False

        if ok:
            st.balloons()

    st.caption(f"🕒 Last refresh: {load_refresh_time()}")

