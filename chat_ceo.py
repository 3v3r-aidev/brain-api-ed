# =========================
# 🧠 AI CEO Assistant + 🗂️ Embedded Chat Hub (Single Streamlit App)
# =========================
# 👉 Copy-paste this entire file as: chat_ceo.py
# ✅ Core deps: streamlit, pandas
# 🔌 Optional (for Google Drive persistence on Streamlit Cloud):
#     google-api-python-client, google-auth, google-auth-httplib2, google-auth-oauthlib
# 🧩 Your existing modules are used if present:
#     file_parser.py, embed_and_store.py, answer_with_rag.py (function: answer)

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 🧩 Try to import your existing modules (safe fallbacks if missing)
# ─────────────────────────────────────────────────────────────
try:
    import file_parser
except Exception:
    file_parser = None

try:
    import embed_and_store
except Exception:
    embed_and_store = None

try:
    from answer_with_rag import answer as rag_answer
except Exception:
    rag_answer = None


# ─────────────────────────────────────────────────────────────
# ☁️ Optional Google Drive Sync (best-effort; no-op if libs/secrets missing)
# ─────────────────────────────────────────────────────────────
def _init_drive_sync():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except Exception:
        return {"enabled": False, "reason": "google client libs not installed"}

    if "gdrive" not in st.secrets:
        return {"enabled": False, "reason": "no gdrive secrets"}

    try:
        creds = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gdrive"]),
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        service = build("drive", "v3", credentials=creds)
        return {"enabled": True, "service": service}
    except Exception as e:
        return {"enabled": False, "reason": f"init error: {e}"}


def _drive_find_file(service, name, folder_id=None):
    q = f"name = '{name}' and trashed = false"
    if folder_id:
        q += f" and '{folder_id}' in parents"
    r = service.files().list(q=q, fields="files(id,name)").execute()
    files = r.get("files", [])
    return files[0]["id"] if files else None


def drive_download_db(ctx, db_name="chats.db"):
    if not ctx.get("enabled"):
        return False
    service = ctx["service"]
    file_id = _drive_find_file(service, db_name, st.secrets["gdrive"].get("folder_id"))
    if not file_id:
        return False
    from googleapiclient.http import MediaIoBaseDownload
    import io

    req = service.files().get_media(fileId=file_id)
    with io.FileIO(db_name, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


def drive_upload_db(ctx, db_name="chats.db"):
    if not ctx.get("enabled"):
        return False
    service = ctx["service"]
    from googleapiclient.http import MediaIoBaseUpload
    import os

    if not Path(db_name).exists():
        return False
    file_id = _drive_find_file(service, db_name, st.secrets["gdrive"].get("folder_id"))
    media = MediaIoBaseUpload(open(db_name, "rb"), mimetype="application/octet-stream", resumable=False)
    meta = {"name": db_name}
    folder_id = st.secrets["gdrive"].get("folder_id")
    if folder_id:
        meta["parents"] = [folder_id]
    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        service.files().create(body=meta, media_body=media, fields="id").execute()
    return True


DRIVE_CTX = _init_drive_sync()
if DRIVE_CTX.get("enabled"):
    try:
        drive_download_db(DRIVE_CTX)  # best-effort restore DB on startup
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 🗂️ Embedded SQLite DB (self-contained; no external db.py needed)
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

@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    yield conn
    conn.commit()
    conn.close()

def db_init():
    with _conn() as conn:
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip():
                conn.execute(stmt)

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
        conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title.strip() or "Untitled", now, conv_id),
        )

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
# 🗃️ Legacy JSON history (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────
HIST_PATH = Path("chat_history.json")
REFRESH_PATH = Path("last_refresh.txt")

def load_history():
    if HIST_PATH.exists():
        return json.loads(HIST_PATH.read_text(encoding="utf-8"))
    return []

def save_history(history):
    HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def reset_chat():
    if HIST_PATH.exists():
        HIST_PATH.unlink()

def save_refresh_time():
    REFRESH_PATH.write_text(datetime.now().strftime("%b-%d-%Y %I:%M %p"))

def load_refresh_time():
    if REFRESH_PATH.exists():
        return REFRESH_PATH.read_text(encoding="utf-8")
    return "N/A"


# ─────────────────────────────────────────────────────────────
# 🔐 Minimal Login (replace with your auth if needed)
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
# ⚙️ App Config + DB init
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧠 AI CEO Assistant", page_icon="🧠", layout="wide")
db_init()  # ensure schema

if not st.session_state["authenticated"]:
    login()
    st.stop()


# ─────────────────────────────────────────────────────────────
# 🗂️ Sidebar — Conversations (Embedded Chat Hub)
# ─────────────────────────────────────────────────────────────
st.sidebar.subheader("🗂️ Conversations")

# Ensure one active conversation exists
if "active_conv" not in st.session_state:
    st.session_state["active_conv"] = db_new_conversation("🧾 Default Conversation")

# ➕ New conversation
with st.sidebar.expander("➕ New Conversation"):
    _new_title = st.text_input("📝 Title", placeholder="e.g., Buyer agreement review")
    if st.button("✅ Create"):
        _cid = db_new_conversation(_new_title or "Untitled")
        st.session_state["active_conv"] = _cid
        try:
            drive_upload_db(DRIVE_CTX)
        except Exception:
            pass
        st.rerun()

# 🔎 Search & list
_search = st.sidebar.text_input("🔎 Search")
_convs = db_list_conversations(_search)
if not _convs:
    st.sidebar.caption("ℹ️ No conversations yet.")
else:
    for _cid, _title, _, _ in _convs:
        if st.sidebar.button(_title or f"Conversation {_cid}", key=f"conv_sel_{_cid}"):
            st.session_state["active_conv"] = _cid
            st.rerun()

# ✏️ Rename / 🗑️ Delete
_acid = st.session_state.get("active_conv")
if _acid:
    st.sidebar.markdown("---")
    _current_title = next((t for i, t, _, _ in _convs if i == _acid), "Untitled") if _convs else "Untitled"
    _new_name = st.sidebar.text_input("✏️ Rename", value=_current_title)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Save Name", key="conv_rename"):
            db_rename_conversation(_acid, _new_name or "Untitled")
            try:
                drive_upload_db(DRIVE_CTX)
            except Exception:
                pass
            st.rerun()
    with c2:
        if st.button("🗑️ Delete", key="conv_delete"):
            db_delete_conversation(_acid)
            try:
                drive_upload_db(DRIVE_CTX)
            except Exception:
                pass
            remain = db_list_conversations("")
            st.session_state["active_conv"] = remain[0][0] if remain else db_new_conversation("🧾 Default Conversation")
            st.rerun()


# ─────────────────────────────────────────────────────────────
# 🧭 Sidebar — App Navigation
# ─────────────────────────────────────────────────────────────
mode = st.sidebar.radio("🧭 Navigation", ["💬 New Chat", "📜 View History", "🔁 Refresh Data"], index=0)
st.sidebar.markdown("---")
st.sidebar.checkbox("🗂️ Limit to meeting docs only", value=False, key="limit_meetings")
st.sidebar.caption("💡 Tip: start a message with **REMINDER:** to save a reminder text file.")


# ─────────────────────────────────────────────────────────────
# 💬 Page: NEW CHAT
# ─────────────────────────────────────────────────────────────
if mode == "💬 New Chat":
    st.title("💬 New Chat")
    history = load_history()  # legacy JSON kept so your old chat flow still renders above

    # Show prior turns (legacy JSON)
    for turn in history:
        role = turn.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(f"🗨️ [{turn.get('timestamp', 'N/A')}]  \n{turn.get('content', '')}")

    user_msg = st.chat_input("✍️ Type your question or add a REMINDER…")
    if user_msg:
        # 1) 📝 REMINDER shortcut
        if user_msg.strip().lower().startswith("reminder:"):
            body = re.sub(r"^reminder:\s*", "", user_msg.strip(), flags=re.I)
            title_hint = (body.split("\n", 1)[0] or "Reminder")[:60]
            folder = Path("reminders"); folder.mkdir(exist_ok=True)
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{re.sub(r'[^a-zA-Z0-9._ -]+','_',title_hint)}.txt"
            path = folder / fname
            path.write_text(body, encoding="utf-8")
            st.success(f"💾 Reminder saved: `{path}`")

        # 2) 🧾 Append to legacy JSON
        now = datetime.now().strftime("%b-%d-%Y %I:%M%p")
        history.append({"role": "user", "content": user_msg, "timestamp": now})
        save_history(history)

        # 3) 🧱 Mirror to SQLite (active conversation)
        try:
            db_add_message(st.session_state["active_conv"], "user", user_msg, ts=now)
        except Exception:
            pass

        # 4) 🤖 Get assistant reply via your RAG pipeline
        with st.chat_message("assistant"):
            st.markdown("🧠 Thinking…")
            try:
                if rag_answer is not None:
                    reply = rag_answer(
                        user_msg,
                        k=7,
                        chat_history=history,
                        restrict_to_meetings=st.session_state["limit_meetings"],
                    )
                else:
                    reply = "⚠️ answer_with_rag.answer() not available. Please ensure the module is present."
            except TypeError:
                # Fallback if your answer() signature differs
                reply = rag_answer(user_msg) if rag_answer else "⚠️ answer() missing."
            except Exception as e:
                reply = f"❌ Error: {e}"

            ts = datetime.now().strftime("%b-%d-%Y %I:%M%p")
            st.markdown(f"🧾 [{ts}]  \n{reply}")

        # 5) 💽 Save assistant turn (JSON + DB + best-effort Drive upload)
        history.append({"role": "assistant", "content": reply, "timestamp": ts})
        save_history(history)
        try:
            db_add_message(st.session_state["active_conv"], "assistant", reply, ts=ts)
            drive_upload_db(DRIVE_CTX)
        except Exception:
            pass

        st.rerun()


# ─────────────────────────────────────────────────────────────
# 📜 Page: VIEW HISTORY (DB-backed)
# ─────────────────────────────────────────────────────────────
elif mode == "📜 View History":
    st.title("📜 Conversation History")
    cid = st.session_state.get("active_conv")
    msgs = db_get_messages(cid) if cid else []

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
# 🔁 Page: REFRESH DATA (Parsing + Embeddings)
# ─────────────────────────────────────────────────────────────
elif mode == "🔁 Refresh Data":
    st.title("🔁 Refresh Data")
    st.write("📥 Parse & 🧩 Embed your documents so the assistant answers with the latest context.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 Parse Documents"):
            if file_parser and hasattr(file_parser, "main"):
                try:
                    file_parser.main()
                    st.success("✅ Parsing complete.")
                except Exception as e:
                    st.error(f"❌ Parsing failed: {e}")
            else:
                st.warning("⚠️ file_parser.main() not found.")

    with c2:
        if st.button("🧩 Embed & Store"):
            if embed_and_store and hasattr(embed_and_store, "main"):
                try:
                    embed_and_store.main()
                    save_refresh_time()
                    st.success("✅ Embeddings stored.")
                except Exception as e:
                    st.error(f"❌ Embedding failed: {e}")
            else:
                st.warning("⚠️ embed_and_store.main() not found.")

    st.caption(f"🕒 Last refresh: {load_refresh_time()}")
