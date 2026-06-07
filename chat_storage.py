import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://postgres:postgres@localhost:5433/rag_chatbot",
)
SQLITE_PATH = os.getenv("CHAT_DB_PATH", "./data/chat_history.db")

_engine = None
_backend: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_backend() -> str:
    global _backend
    if _backend:
        return _backend
    try:
        engine = create_engine(POSTGRES_CONNECTION_STRING)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _backend = "postgres"
    except Exception:
        Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
        _backend = "sqlite"
    return _backend


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    if _get_backend() == "postgres":
        _engine = create_engine(POSTGRES_CONNECTION_STRING)
    else:
        _engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    return _engine


def init_chat_db() -> None:
    engine = _get_engine()
    backend = _get_backend()

    with engine.begin() as conn:
        if backend == "postgres":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id VARCHAR(36) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        mode VARCHAR(20) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(36) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        role VARCHAR(20) NOT NULL,
                        content TEXT NOT NULL,
                        sources JSONB,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        sources TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                    )
                    """
                )
            )


def get_storage_label() -> str:
    return "PostgreSQL" if _get_backend() == "postgres" else "SQLite"


def create_session(mode: str, title: str = "New Chat") -> str:
    init_chat_db()
    session_id = str(uuid.uuid4())
    now = _now()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO chat_sessions (id, title, mode, created_at, updated_at)
                VALUES (:id, :title, :mode, :created_at, :updated_at)
                """
            ),
            {
                "id": session_id,
                "title": title[:255],
                "mode": mode,
                "created_at": now,
                "updated_at": now,
            },
        )
    return session_id


def list_sessions(mode: str | None = None) -> list[dict[str, Any]]:
    init_chat_db()
    engine = _get_engine()
    query = """
        SELECT id, title, mode, created_at, updated_at
        FROM chat_sessions
    """
    params: dict[str, Any] = {}
    if mode:
        query += " WHERE mode = :mode"
        params["mode"] = mode
    query += " ORDER BY updated_at DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [dict(row) for row in rows]


def get_messages(session_id: str) -> list[dict[str, Any]]:
    init_chat_db()
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT role, content, sources
                FROM chat_messages
                WHERE session_id = :session_id
                ORDER BY id ASC
                """
            ),
            {"session_id": session_id},
        ).mappings().all()

    messages = []
    for row in rows:
        sources = row["sources"]
        if sources is None:
            sources = []
        elif isinstance(sources, str):
            sources = json.loads(sources) if sources else []
        elif not isinstance(sources, list):
            sources = list(sources) if sources else []
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "sources": sources or [],
            }
        )
    return messages


def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    init_chat_db()
    now = _now()
    engine = _get_engine()
    sources_json = json.dumps(sources or [])

    with engine.begin() as conn:
        if _get_backend() == "postgres":
            conn.execute(
                text(
                    """
                    INSERT INTO chat_messages (session_id, role, content, sources, created_at)
                    VALUES (:session_id, :role, :content, CAST(:sources AS JSONB), :created_at)
                    """
                ),
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "sources": sources_json,
                    "created_at": now,
                },
            )
        else:
            conn.execute(
                text(
                    """
                    INSERT INTO chat_messages (session_id, role, content, sources, created_at)
                    VALUES (:session_id, :role, :content, :sources, :created_at)
                    """
                ),
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "sources": sources_json,
                    "created_at": now,
                },
            )
        conn.execute(
            text("UPDATE chat_sessions SET updated_at = :updated_at WHERE id = :id"),
            {"updated_at": now, "id": session_id},
        )


def update_session_title(session_id: str, title: str) -> None:
    init_chat_db()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE chat_sessions
                SET title = :title, updated_at = :updated_at
                WHERE id = :id
                """
            ),
            {"title": title[:255], "updated_at": _now(), "id": session_id},
        )


def delete_session(session_id: str) -> None:
    init_chat_db()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM chat_messages WHERE session_id = :id"),
            {"id": session_id},
        )
        conn.execute(
            text("DELETE FROM chat_sessions WHERE id = :id"),
            {"id": session_id},
        )


def auto_title(first_message: str) -> str:
    cleaned = " ".join(first_message.strip().split())
    if len(cleaned) <= 48:
        return cleaned or "New Chat"
    return cleaned[:45] + "…"
