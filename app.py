import streamlit as st
from langchain_core.messages import HumanMessage

from agentic_chatbot_backend import chatbot, format_sources, rag_chatbot
from chat_storage import (
    add_message,
    auto_title,
    create_session,
    delete_session,
    get_messages,
    get_storage_label,
    init_chat_db,
    list_sessions,
    update_session_title,
)
from rag_backend import check_db_connection, get_active_backend, ingest_pdf_file

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_chat_db()

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a1f3c 0%, #0b1120 45%, #0b1120 100%);
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 6rem;
    max-width: 920px;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1321 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4,
[data-testid="stSidebar"] .stMarkdown h5,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #e2e8f0 !important;
}

/* File uploader — fix invisible text on white box */
[data-testid="stSidebar"] [data-testid="stFileUploader"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] > section,
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
    background: #151d2e !important;
    border: 1px dashed rgba(99,102,241,0.45) !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #64748b !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background: #4f46e5 !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] span {
    color: #e2e8f0 !important;
    background: rgba(99,102,241,0.12) !important;
}

[data-testid="stSidebar"] label[data-baseweb="radio"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.35rem;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Chat history items */
.chat-history-item {
    padding: 0.55rem 0.65rem;
    border-radius: 10px;
    margin-bottom: 0.25rem;
    border: 1px solid transparent;
    font-size: 0.84rem;
    color: #cbd5e1 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.chat-history-item.active {
    background: rgba(99,102,241,0.15);
    border-color: rgba(99,102,241,0.35);
    color: #f1f5f9 !important;
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
}
.hero-wrap h1 {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc !important;
    margin: 0 0 0.3rem 0;
}
.hero-wrap p { color: #94a3b8 !important; margin: 0 0 0.85rem 0; font-size: 0.92rem; }
.badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    display: inline-flex; padding: 0.3rem 0.75rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600;
}
.badge-mode { background: rgba(99,102,241,0.2); color: #a5b4fc !important; border: 1px solid rgba(99,102,241,0.35); }
.badge-rag { background: rgba(16,185,129,0.15); color: #6ee7b7 !important; border: 1px solid rgba(16,185,129,0.35); }
.badge-stack { background: rgba(255,255,255,0.05); color: #94a3b8 !important; border: 1px solid rgba(148,163,184,0.15); }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 16px !important;
    padding: 0.85rem 1.1rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #f1f5f9 !important;
    line-height: 1.65 !important;
    font-size: 0.95rem !important;
}

/* ── Chat input ── */
[data-testid="stBottomBlockContainer"] {
    background: rgba(11,17,32,0.95) !important;
    border-top: 1px solid rgba(99,102,241,0.15) !important;
}
div[data-testid="stChatInput"] textarea {
    background: #151d2e !important;
    color: #f8fafc !important;
    caret-color: #f8fafc !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; opacity: 1 !important; }

.source-card {
    background: #151d2e; border: 1px solid rgba(99,102,241,0.2);
    border-left: 3px solid #6366f1; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
}
.source-card p { color: #cbd5e1 !important; margin: 0; font-size: 0.85rem; }
.source-card strong { color: #a5b4fc !important; }

.status-card {
    background: rgba(255,255,255,0.04); border-radius: 12px;
    padding: 0.75rem 1rem; border: 1px solid rgba(148,163,184,0.12);
}
.status-ok { color: #34d399 !important; font-weight: 600; font-size: 0.85rem; }
.status-warn { color: #fbbf24 !important; font-weight: 600; font-size: 0.85rem; }
.status-err { color: #f87171 !important; font-weight: 600; font-size: 0.85rem; }

.empty-state { text-align: center; padding: 3rem 1rem; }
.empty-state h3 { color: #94a3b8 !important; font-weight: 600; }
.empty-state p { color: #64748b !important; font-size: 0.9rem; }

.streamlit-expanderHeader { color: #a5b4fc !important; font-weight: 600 !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Retrieved Sources ({len(sources)})"):
        for source in sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <p><strong>Source {source['index']}</strong> · Page {source['page']}</p>
                    <p>{source['preview']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_state(is_rag: bool) -> None:
    if is_rag:
        st.markdown(
            """
            <div class="empty-state">
                <div style="font-size:2.5rem;">📄</div>
                <h3>Document Q&A Ready</h3>
                <p>Upload a PDF in the sidebar, click <strong>Ingest Documents</strong>, then ask questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div style="font-size:2.5rem;">💬</div>
                <h3>Start a conversation</h3>
                <p>Type a message below — powered by Groq &amp; LangGraph.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def current_mode_key(is_rag: bool) -> str:
    return "rag" if is_rag else "general"


def ensure_active_session(mode_key: str) -> str:
    sessions = list_sessions(mode_key)
    if st.session_state.get("active_session_id") and any(
        s["id"] == st.session_state.active_session_id for s in sessions
    ):
        return st.session_state.active_session_id

    if sessions:
        st.session_state.active_session_id = sessions[0]["id"]
        return sessions[0]["id"]

    new_id = create_session(mode_key)
    st.session_state.active_session_id = new_id
    return new_id


# ── Session state ──
if "upload_log" not in st.session_state:
    st.session_state.upload_log = []
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "menu_open_id" not in st.session_state:
    st.session_state.menu_open_id = None

# ── Sidebar ──
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        mode_key = st.session_state.get("mode_key", "general")
        st.session_state.active_session_id = create_session(mode_key)
        st.rerun()

    st.markdown("##### 💬 Chat History")
    st.caption(f"Saved in {get_storage_label()}")

    chat_mode = st.radio(
        "Chat Mode",
        ["💬 General Chat", "📄 Document Q&A (RAG)"],
        label_visibility="collapsed",
    )
    is_rag_mode = chat_mode.startswith("📄")
    mode_key = current_mode_key(is_rag_mode)
    st.session_state.mode_key = mode_key

    sessions = list_sessions(mode_key)
    active_id = ensure_active_session(mode_key)

    if not sessions:
        st.caption("No chats yet. Start a new one!")

    for session in sessions:
        sid = session["id"]
        is_active = sid == active_id
        title = session["title"] or "New Chat"

        col_title, col_menu = st.columns([5, 1])
        with col_title:
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                title,
                key=f"load_{sid}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state.active_session_id = sid
                st.rerun()

        with col_menu:
            with st.popover("⋮", use_container_width=True):
                st.markdown(f"**{title}**")
                new_title = st.text_input(
                    "Rename",
                    value=title,
                    key=f"rename_input_{sid}",
                    label_visibility="collapsed",
                )
                if st.button("Save name", key=f"rename_save_{sid}", use_container_width=True):
                    if new_title.strip():
                        update_session_title(sid, new_title.strip())
                    st.rerun()
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{sid}",
                    use_container_width=True,
                    type="secondary",
                ):
                    delete_session(sid)
                    if st.session_state.active_session_id == sid:
                        st.session_state.active_session_id = None
                    st.rerun()

    st.divider()

    st.markdown("##### 🗄️ Vector Store")
    db_ok, db_message = check_db_connection()
    backend = get_active_backend()

    if backend == "pgvector":
        st.markdown(
            '<div class="status-card"><span class="status-ok">● PostgreSQL + pgVector</span></div>',
            unsafe_allow_html=True,
        )
    elif db_ok:
        st.markdown(
            '<div class="status-card"><span class="status-warn">● Chroma (local fallback)</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-card"><span class="status-err">● Offline</span></div>',
            unsafe_allow_html=True,
        )

    if is_rag_mode:
        st.divider()
        st.markdown("##### 📄 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button(
            "⬆️ Ingest Documents",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        ):
            if not db_ok:
                st.error("Vector store unavailable.")
            else:
                with st.spinner("Processing PDFs…"):
                    for uploaded in uploaded_files:
                        try:
                            count, name = ingest_pdf_file(uploaded)
                            st.session_state.upload_log.append(f"✅ {name} — {count} chunks")
                        except Exception as exc:
                            st.session_state.upload_log.append(f"❌ {uploaded.name} — {exc}")
                st.success("Documents indexed!")
                st.rerun()

        if st.session_state.upload_log:
            st.markdown("**Indexed**")
            for entry in st.session_state.upload_log[-4:]:
                st.caption(entry)

# ── Header ──
mode_badge = "badge-rag" if is_rag_mode else "badge-mode"
mode_label = "RAG Document Q&A" if is_rag_mode else "General Chat"

st.markdown(
    f"""
    <div class="hero-wrap">
        <h1>🤖 Agentic AI Chatbot</h1>
        <p>LangGraph · Groq LLM · LangChain RAG · pgVector · Chat history in {get_storage_label()}</p>
        <div class="badge-row">
            <span class="badge {mode_badge}">{mode_label}</span>
            <span class="badge badge-stack">LangGraph</span>
            <span class="badge badge-stack">Groq</span>
            <span class="badge badge-stack">{backend.upper() if db_ok else "OFFLINE"}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load messages from database ──
session_id = st.session_state.active_session_id
message_history = get_messages(session_id) if session_id else []

config = {"configurable": {"thread_id": session_id or "default-thread"}}

if not message_history:
    render_empty_state(is_rag_mode)

for message in message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

# ── Chat input ──
placeholder = (
    "Ask a question about your uploaded documents…"
    if is_rag_mode
    else "Type your message here…"
)
user_input = st.chat_input(placeholder)

if user_input and session_id:
    add_message(session_id, "user", user_input)

    if len(message_history) == 0:
        update_session_title(session_id, auto_title(user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_rag_mode and not db_ok:
            ai_message = (
                "Vector store is unavailable. Install dependencies or run "
                "`docker compose up -d`."
            )
            st.markdown(ai_message)
            sources = []
        else:
            graph = rag_chatbot if is_rag_mode else chatbot
            stream = graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
            ai_message = st.write_stream(
                chunk.content
                for chunk, _ in stream
                if getattr(chunk, "content", None)
            )
            sources = []
            if is_rag_mode:
                state = graph.get_state(config)
                retrieved = state.values.get("sources", [])
                sources = format_sources(retrieved)
                if sources:
                    render_sources(sources)

    add_message(session_id, "assistant", ai_message or "", sources)
    st.rerun()
