import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

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
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    yield conn
    conn.commit()
    conn.close()

def init_db():
    with get_conn() as conn:
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip():
                conn.execute(stmt)

def new_conversation(title: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations(title, created_at, updated_at) VALUES(?,?,?)",
            (title.strip() or "Untitled", now, now),
        )
        return cur.lastrowid

def rename_conversation(conv_id: int, title: str):
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                     (title.strip() or "Untitled", now, conv_id))

def delete_conversation(conv_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

def list_conversations(search: str = ""):
    q = "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    params = ()
    if search.strip():
        q = "SELECT id, title, created_at, updated_at FROM conversations WHERE title LIKE ? ORDER BY updated_at DESC"
        params = (f"%{search.strip()}%",)
    with get_conn() as conn:
        return conn.execute(q, params).fetchall()

def add_message(conv_id: int, role: str, content: str, ts: str | None = None) -> int:
    ts = ts or datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages(conv_id, role, content, ts) VALUES(?,?,?,?)",
            (conv_id, role, content, ts),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conv_id))
        return cur.lastrowid

def get_messages(conv_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, role, content, ts FROM messages WHERE conv_id=? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
