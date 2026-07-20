"""
chat_history.py — Persistent Chat Session Storage
====================================================
Manages a SQLite database (chat_history.db) that stores all chat sessions
and their messages independently from the knowledge base (knowledge_base.db).

Provides full CRUD:
  - Create : create_session()
  - Read   : get_all_sessions(), get_messages()
  - Update : rename_session(), add_message()
  - Delete : delete_session()
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_DB = Path(__file__).parent / "chat_history.db"


# ------------------------------------------------------------------
# Schema Initialization
# ------------------------------------------------------------------
def init_history_db() -> None:
    """
    Creates the chat_sessions and chat_messages tables if they do not
    already exist. Safe to call on every app startup.
    """
    conn = sqlite3.connect(HISTORY_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL DEFAULT 'New Chat',
            model      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            sources    TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# CREATE
# ------------------------------------------------------------------
def create_session(model_name: str) -> str:
    """
    Creates a new empty chat session and returns its UUID.

    Parameters:
        model_name : The active chat model alias at the time of creation.

    Returns:
        The new session's unique ID string.
    """
    session_id = str(uuid.uuid4())
    now        = datetime.now().isoformat()

    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("""
        INSERT INTO chat_sessions (id, name, model, created_at, updated_at)
        VALUES (?, 'New Chat', ?, ?, ?)
    """, (session_id, model_name, now, now))
    conn.commit()
    conn.close()

    return session_id


# ------------------------------------------------------------------
# READ
# ------------------------------------------------------------------
def get_all_sessions() -> list[dict]:
    """
    Returns all chat sessions ordered by most recently updated first.

    Returns:
        List of dicts: [{"id", "name", "model", "created_at", "updated_at"}, ...]
    """
    conn    = sqlite3.connect(HISTORY_DB)
    cursor  = conn.cursor()
    cursor.execute("""
        SELECT id, name, model, created_at, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [
        {"id": r[0], "name": r[1], "model": r[2],
         "created_at": r[3], "updated_at": r[4]}
        for r in rows
    ]


def get_messages(session_id: str) -> list[dict]:
    """
    Returns all messages for a specific session in chronological order.

    Returns:
        List of dicts: [{"role", "content", "sources"}, ...]
        Sources is deserialized from JSON if present.
    """
    conn   = sqlite3.connect(HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, sources
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()

    messages = []
    for role, content, sources_json in rows:
        msg = {"role": role, "content": content}
        if sources_json:
            msg["sources"] = json.loads(sources_json)
        messages.append(msg)

    return messages


def session_exists(session_id: str) -> bool:
    """Returns True if the session ID exists in the database."""
    conn   = sqlite3.connect(HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


# ------------------------------------------------------------------
# UPDATE
# ------------------------------------------------------------------
def rename_session(session_id: str, new_name: str) -> None:
    """
    Renames a chat session.

    Parameters:
        session_id : Target session UUID.
        new_name   : The new display name. Stripped of leading/trailing whitespace.
    """
    now = datetime.now().isoformat()
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("""
        UPDATE chat_sessions SET name = ?, updated_at = ? WHERE id = ?
    """, (new_name.strip(), now, session_id))
    conn.commit()
    conn.close()


def add_message(session_id: str,
                role: str,
                content: str,
                sources: list | None = None) -> None:
    """
    Appends a message to a chat session and updates the session's updated_at
    timestamp. If the session name is still 'New Chat', auto-names it using
    the first user message (truncated to 45 characters).

    Parameters:
        session_id : Target session UUID.
        role       : 'user' or 'assistant'.
        content    : The message text.
        sources    : Optional list of source dicts (for assistant messages).
    """
    now          = datetime.now().isoformat()
    sources_json = json.dumps(sources) if sources else None

    conn = sqlite3.connect(HISTORY_DB)

    # Auto-name the session from the first user message
    if role == "user":
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM chat_sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if row and row[0] == "New Chat":
            auto_name = content[:45] + ("..." if len(content) > 45 else "")
            conn.execute(
                "UPDATE chat_sessions SET name = ?, updated_at = ? WHERE id = ?",
                (auto_name, now, session_id)
            )

    conn.execute("""
        INSERT INTO chat_messages (session_id, role, content, sources, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, role, content, sources_json, now))

    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
        (now, session_id)
    )

    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# DELETE
# ------------------------------------------------------------------
def delete_session(session_id: str) -> None:
    """
    Permanently deletes a session and all its messages.
    The ON DELETE CASCADE constraint handles the child rows automatically.

    Parameters:
        session_id : Target session UUID.
    """
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
