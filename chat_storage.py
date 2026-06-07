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


def _migrate_schema(conn) -> None:
    migrations = [
        "ALTER TABLE chat_messages ADD COLUMN feedback VARCHAR(10)",
        "ALTER TABLE chat_messages ADD COLUMN response_time_ms INTEGER",
    ]
    for sql in migrations:
        try:
            conn.execute(text(sql))
        except Exception:
            pass


def ensure_document_chunks_table() -> None:
    """Create document_chunks if missing (handles DBs created before this table existed)."""
    engine = _get_engine()
    backend = _get_backend()
    with engine.begin() as conn:
        if backend == "postgres":
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        collection VARCHAR(100) NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER,
                        content TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER,
                        content TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
            )


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
                        feedback VARCHAR(10),
                        response_time_ms INTEGER,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        collection VARCHAR(100) NOT NULL,
                        pages INTEGER DEFAULT 0,
                        chunks INTEGER DEFAULT 0,
                        file_type VARCHAR(20) NOT NULL,
                        uploaded_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        collection VARCHAR(100) NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER,
                        content TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
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
                        feedback TEXT,
                        response_time_ms INTEGER,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        pages INTEGER DEFAULT 0,
                        chunks INTEGER DEFAULT 0,
                        file_type TEXT NOT NULL,
                        uploaded_at TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER,
                        content TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
            )
        _migrate_schema(conn)
    ensure_document_chunks_table()


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
    query = "SELECT id, title, mode, created_at, updated_at FROM chat_sessions"
    params: dict[str, Any] = {}
    if mode:
        query += " WHERE mode = :mode"
        params["mode"] = mode
    query += " ORDER BY updated_at DESC"
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [dict(row) for row in rows]


def _parse_sources(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw) if raw else []
    if isinstance(raw, list):
        return raw
    return list(raw) if raw else []


def get_messages(session_id: str) -> list[dict[str, Any]]:
    init_chat_db()
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, role, content, sources, feedback, response_time_ms, created_at
                FROM chat_messages
                WHERE session_id = :session_id
                ORDER BY id ASC
                """
            ),
            {"session_id": session_id},
        ).mappings().all()

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "sources": _parse_sources(row["sources"]),
            "feedback": row.get("feedback"),
            "response_time_ms": row.get("response_time_ms"),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    response_time_ms: int | None = None,
) -> int | None:
    init_chat_db()
    now = _now()
    engine = _get_engine()
    sources_json = json.dumps(sources or [])

    with engine.begin() as conn:
        if _get_backend() == "postgres":
            result = conn.execute(
                text(
                    """
                    INSERT INTO chat_messages
                    (session_id, role, content, sources, response_time_ms, created_at)
                    VALUES (:session_id, :role, :content, CAST(:sources AS JSONB),
                            :response_time_ms, :created_at)
                    RETURNING id
                    """
                ),
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "sources": sources_json,
                    "response_time_ms": response_time_ms,
                    "created_at": now,
                },
            )
            message_id = result.scalar()
        else:
            result = conn.execute(
                text(
                    """
                    INSERT INTO chat_messages
                    (session_id, role, content, sources, response_time_ms, created_at)
                    VALUES (:session_id, :role, :content, :sources,
                            :response_time_ms, :created_at)
                    """
                ),
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "sources": sources_json,
                    "response_time_ms": response_time_ms,
                    "created_at": now,
                },
            )
            message_id = result.lastrowid

        conn.execute(
            text("UPDATE chat_sessions SET updated_at = :updated_at WHERE id = :id"),
            {"updated_at": now, "id": session_id},
        )
    return message_id


def set_message_feedback(message_id: int, feedback: str) -> None:
    init_chat_db()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE chat_messages SET feedback = :feedback WHERE id = :id"),
            {"feedback": feedback, "id": message_id},
        )


def update_session_title(session_id: str, title: str) -> None:
    init_chat_db()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE chat_sessions SET title = :title, updated_at = :updated_at WHERE id = :id"
            ),
            {"title": title[:255], "updated_at": _now(), "id": session_id},
        )


def delete_session(session_id: str) -> None:
    init_chat_db()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM chat_messages WHERE session_id = :id"), {"id": session_id})
        conn.execute(text("DELETE FROM chat_sessions WHERE id = :id"), {"id": session_id})


def auto_title(first_message: str) -> str:
    cleaned = " ".join(first_message.strip().split())
    if len(cleaned) <= 48:
        return cleaned or "New Chat"
    return cleaned[:45] + "…"


def register_document(
    file_name: str,
    collection: str,
    pages: int,
    chunks: int,
    file_type: str,
) -> None:
    init_chat_db()
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO documents (file_name, collection, pages, chunks, file_type, uploaded_at)
                VALUES (:file_name, :collection, :pages, :chunks, :file_type, :uploaded_at)
                """
            ),
            {
                "file_name": file_name,
                "collection": collection,
                "pages": pages,
                "chunks": chunks,
                "file_type": file_type,
                "uploaded_at": _now(),
            },
        )


def list_documents(collection: str | None = None) -> list[dict[str, Any]]:
    init_chat_db()
    engine = _get_engine()
    query = "SELECT * FROM documents"
    params: dict[str, Any] = {}
    if collection:
        query += " WHERE collection = :collection"
        params["collection"] = collection
    query += " ORDER BY uploaded_at DESC"
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [dict(row) for row in rows]


def save_document_chunks(
    file_name: str,
    collection: str,
    chunks: list[dict],
) -> None:
    ensure_document_chunks_table()
    init_chat_db()
    engine = _get_engine()
    now = _now()
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM document_chunks WHERE file_name = :file_name AND collection = :collection"
            ),
            {"file_name": file_name, "collection": collection},
        )
        for chunk in chunks:
            conn.execute(
                text(
                    """
                    INSERT INTO document_chunks
                    (file_name, collection, chunk_index, page, content, char_count, created_at)
                    VALUES (:file_name, :collection, :chunk_index, :page, :content, :char_count, :created_at)
                    """
                ),
                {
                    "file_name": file_name,
                    "collection": collection,
                    "chunk_index": chunk["chunk_index"],
                    "page": chunk.get("page"),
                    "content": chunk["content"],
                    "char_count": chunk["char_count"],
                    "created_at": now,
                },
            )


def get_document_chunks(file_name: str, collection: str) -> list[dict[str, Any]]:
    ensure_document_chunks_table()
    init_chat_db()
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT chunk_index, page, content, char_count
                FROM document_chunks
                WHERE file_name = :file_name AND collection = :collection
                ORDER BY chunk_index ASC
                """
            ),
            {"file_name": file_name, "collection": collection},
        ).mappings().all()
    return [dict(row) for row in rows]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def get_analytics() -> dict[str, Any]:
    return get_dashboard_analytics()


def get_dashboard_analytics() -> dict[str, Any]:
    init_chat_db()
    engine = _get_engine()
    with engine.connect() as conn:
        total_chats = conn.execute(text("SELECT COUNT(*) FROM chat_sessions")).scalar() or 0
        total_messages = conn.execute(text("SELECT COUNT(*) FROM chat_messages")).scalar() or 0
        total_docs = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
        total_chunks = conn.execute(
            text("SELECT COALESCE(SUM(chunks), 0) FROM documents")
        ).scalar() or 0
        avg_response = conn.execute(
            text(
                "SELECT AVG(response_time_ms) FROM chat_messages "
                "WHERE role = 'assistant' AND response_time_ms IS NOT NULL"
            )
        ).scalar()

        general_questions = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE m.role = 'user' AND s.mode = 'general'
                """
            )
        ).scalar() or 0

        doc_questions = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE m.role = 'user' AND s.mode = 'rag'
                """
            )
        ).scalar() or 0

        general_rows = conn.execute(
            text(
                """
                SELECT m.content FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE s.mode = 'general'
                """
            )
        ).fetchall()
        doc_rows = conn.execute(
            text(
                """
                SELECT m.content FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE s.mode = 'rag'
                """
            )
        ).fetchall()

        response_rows = conn.execute(
            text(
                """
                SELECT response_time_ms FROM chat_messages
                WHERE role = 'assistant' AND response_time_ms IS NOT NULL
                ORDER BY id DESC LIMIT 30
                """
            )
        ).fetchall()

        recent_rows = conn.execute(
            text(
                """
                SELECT s.mode, m.role, m.content, m.response_time_ms, m.created_at
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                ORDER BY m.id DESC LIMIT 15
                """
            )
        ).mappings().all()

        chunk_rows = conn.execute(
            text("SELECT file_name, char_count FROM document_chunks ORDER BY file_name, chunk_index")
        ).fetchall()

    general_tokens = sum(_estimate_tokens(r[0] or "") for r in general_rows)
    doc_tokens = sum(_estimate_tokens(r[0] or "") for r in doc_rows)
    chunk_sizes = [r[1] for r in chunk_rows if r[1]]
    total_chunk_chars = sum(chunk_sizes)

    doc_storage_map: dict[str, int] = {}
    for file_name, char_count in chunk_rows:
        doc_storage_map[file_name] = doc_storage_map.get(file_name, 0) + (char_count or 0)
    doc_storage = [{"file_name": k, "chars": v} for k, v in doc_storage_map.items()]

    recent_activity = []
    for row in recent_rows:
        section = "General Chat" if row["mode"] == "general" else "Documents"
        preview = (row["content"] or "")[:60].replace("\n", " ")
        recent_activity.append(
            {
                "Section": section,
                "Role": row["role"],
                "Preview": preview + ("…" if len(row["content"] or "") > 60 else ""),
                "Response (ms)": row["response_time_ms"] or "—",
                "Time": str(row["created_at"])[:19],
            }
        )

    return {
        "total_chats": total_chats,
        "total_messages": total_messages,
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "avg_response_ms": round(avg_response or 0, 0),
        "general_questions": general_questions,
        "doc_questions": doc_questions,
        "general_tokens": general_tokens,
        "doc_tokens": doc_tokens,
        "total_tokens": general_tokens + doc_tokens,
        "chunk_sizes": chunk_sizes,
        "total_chunk_chars": total_chunk_chars,
        "doc_storage": doc_storage,
        "response_times": [r[0] for r in reversed(response_rows)],
        "recent_activity": recent_activity,
        "updated_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    }


def export_session_txt(session_id: str) -> str:
    messages = get_messages(session_id)
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"[{role}] {msg['content']}\n")
    return "\n".join(lines)
