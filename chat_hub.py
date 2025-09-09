import json, os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

import db

APP_TITLE = "Chat Hub"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ───────────────────── Drive Sync (optional) ─────────────────────
drive_enabled = False
drive_error = None
try:
    import drive_sync
    # Secrets might be missing or malformed; handle gracefully
    if "gdrive" in st.secrets:
        # Pull DB from Drive on first load
        if "db_synced" not in st.session_state:
            try:
                ok = drive_sync.download_db()
                st.session_state["db_synced"] = True
            except Exception as e:
                drive_error = f"download_db() failed: {e}"
        drive_enabled = True
except Exception as e:
    drive_error = f"drive_sync import/init failed: {e}"

# ───────────────────── DB init ─────────────────────
db.init_db()

# Seed a sample chat if DB is empty so the UI never looks blank
def _maybe_seed_sample():
    rows = db.list_conversations("")
    if not rows:
        cid = db.new_conversation("Sample conversation")
        db.add_message(cid, "user", "Hello, this is a sample message.")
        db.add_message(cid, "assistant", "Hi! This is a sample assistant reply.")
        return cid
    return rows[0][0]

if "active_conv" not in st.session_state:
    st.session_state["active_conv"] = _maybe_seed_sample()

# ───────────────────── Sidebar ─────────────────────
st.sidebar.title(APP_TITLE)

with st.sidebar.expander("Diagnostics", expanded=True):
    st.code({
        "working_dir": os.getcwd(),
        "db_file_exists": Path("chats.db").exists(),
        "db_size_bytes": Path("chats.db").stat().st_size if Path("chats.db").exists() else 0,
        "drive_enabled": drive_enabled,
        "drive_error": drive_error,
        "has_secrets_gdrive": "gdrive" in st.secrets,
        "has_folder_id": ("gdrive" in st.secrets) and ("folder_id" in st.secrets["gdrive"]),
    }, language="json")

with st.sidebar.expander("New Conversation"):
    new_title = st.text_input("Title", placeholder="e.g., Buyer agreement review")
    if st.button("Create"):
        cid = db.new_conversation(new_title or "Untitled")
        st.session_state["active_conv"] = cid
        # best-effort upload
        try:
            if drive_enabled:
                drive_sync.upload_db()
        except:
            pass
        st.experimental_rerun()

search = st.sidebar.text_input("Search chats")
rows = db.list_conversations(search)
if not rows:
    st.sidebar.info("No conversations yet. Use 'New Conversation' above.")
else:
    for cid, title, _, _ in rows:
        if st.sidebar.button(title or f"Conversation {cid}", key=f"conv_{cid}"):
            st.session_state["active_conv"] = cid

st.sidebar.markdown("---")
st.sidebar.subheader("Utilities")
uploaded = st.sidebar.file_uploader("Import chat_history.json", type=["json"], help="Import from your AI app")
if uploaded:
    try:
        data = json.loads(uploaded.read().decode("utf-8"))
        guess = "Imported Chat"
        for item in data:
            if item.get("role") == "user" and item.get("content"):
                guess = item["content"][:40]
                break
        cid = db.new_conversation(guess)
        for item in data:
            db.add_message(cid, item.get("role","user"), item.get("content",""), ts=item.get("timestamp"))
        st.success(f"Imported {len(data)} messages into '{guess}'")
        st.session_state["active_conv"] = cid
        try:
            if drive_enabled:
                drive_sync.upload_db()
        except:
            pass
    except Exception as e:
        st.error(f"Import failed: {e}")

if "active_conv" in st.session_state:
    if st.sidebar.button("Export current ↧"):
        cid = st.session_state["active_conv"]
        msgs = db.get_messages(cid)
        df = pd.DataFrame(msgs, columns=["id", "role", "content", "timestamp"])
        st.sidebar.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"conversation_{cid}.csv",
            mime="text/csv",
        )

# ───────────────────── Main Area ─────────────────────
st.title("Chats")
active = st.session_state.get("active_conv")
if not active:
    st.info("Select or create a conversation from the left.")
    st.stop()

conv_rows = {r[0]: r for r in rows}
title = conv_rows.get(active, (active, "Untitled", None, None))[1] if rows else "Untitled"

c1, c2, c3 = st.columns([6,1,1])
with c1:
    new_name = st.text_input("Title", value=title, label_visibility="collapsed")
with c2:
    if st.button("Rename"):
        db.rename_conversation(active, new_name)
        try:
            if drive_enabled:
                drive_sync.upload_db()
        except:
            pass
        st.experimental_rerun()
with c3:
    if st.button("Delete", type="primary"):
        db.delete_conversation(active)
        try:
            if drive_enabled:
                drive_sync.upload_db()
        except:
            pass
        st.session_state.pop("active_conv", None)
        st.experimental_rerun()

msgs = db.get_messages(active)
if not msgs:
    st.caption("No messages yet. Add one below.")
for _, role, content, ts in msgs:
    who = "👤" if role == "user" else "🤖" if role == "assistant" else "⚙️"
    st.markdown(f"**{who} {role.capitalize()} — [{ts}]**")
    st.markdown(content if content else "_(empty message)_")
    st.markdown("---")

with st.form("new_msg"):
    role = st.selectbox("Role", ["user", "assistant", "system"])
    content = st.text_area("Message", height=120)
    if st.form_submit_button("Add") and content.strip():
        db.add_message(active, role, content.strip())
        try:
            if drive_enabled:
                drive_sync.upload_db()
        except:
            pass
        st.experimental_rerun()
