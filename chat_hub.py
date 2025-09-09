import json
from datetime import datetime
import pandas as pd
import streamlit as st

import db

# Optional: Google Drive sync
try:
    import drive_sync
    if "db_synced" not in st.session_state:
        drive_sync.download_db()
        st.session_state["db_synced"] = True
except Exception as e:
    st.warning(f"Drive sync disabled: {e}")

APP_TITLE = "Chat Hub"
st.set_page_config(page_title=APP_TITLE, layout="wide")

db.init_db()

# ───────────────────── Sidebar ─────────────────────
st.sidebar.title(APP_TITLE)
with st.sidebar.expander("New Conversation"):
    new_title = st.text_input("Title", placeholder="e.g., Buyer agreement review")
    if st.button("Create"):
        cid = db.new_conversation(new_title or "Untitled")
        st.session_state["active_conv"] = cid
        try: drive_sync.upload_db()
        except: pass
        st.experimental_rerun()

search = st.sidebar.text_input("Search chats")
rows = db.list_conversations(search)
if not rows:
    st.sidebar.info("No conversations yet.")
else:
    for cid, title, _, _ in rows:
        if st.sidebar.button(title, key=f"conv_{cid}"):
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
        try: drive_sync.upload_db()
        except: pass
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
        try: drive_sync.upload_db()
        except: pass
        st.experimental_rerun()
with c3:
    if st.button("Delete", type="primary"):
        db.delete_conversation(active)
        try: drive_sync.upload_db()
        except: pass
        st.session_state.pop("active_conv", None)
        st.experimental_rerun()

msgs = db.get_messages(active)
for _, role, content, ts in msgs:
    who = "👤" if role == "user" else "🤖" if role == "assistant" else "⚙️"
    st.markdown(f"**{who} {role.capitalize()} — [{ts}]**")
    st.markdown(content)
    st.markdown("---")

with st.form("new_msg"):
    role = st.selectbox("Role", ["user", "assistant", "system"])
    content = st.text_area("Message", height=120)
    if st.form_submit_button("Add") and content.strip():
        db.add_message(active, role, content.strip())
        try: drive_sync.upload_db()
        except: pass
        st.experimental_rerun()
