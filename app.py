import streamlit as st
from langchain_core.messages import HumanMessage

from agentic_chatbot_backend import chatbot, format_sources, rag_chatbot
from rag_backend import check_db_connection, ingest_pdf_file

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0f172a 100%);
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 980px;
}

.hero-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(8px);
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
}

.hero-subtitle {
    color: #94a3b8;
    margin-top: 0.35rem;
    font-size: 0.98rem;
}

.mode-pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.mode-general {
    background: rgba(59, 130, 246, 0.18);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.mode-rag {
    background: rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f8fafc;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 0.75rem 1rem;
}

div[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    background: rgba(15, 23, 42, 0.85) !important;
}

.source-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.55rem;
}

.source-card p {
    margin: 0;
    color: #cbd5e1;
    font-size: 0.86rem;
    line-height: 1.45;
}

.status-ok { color: #34d399; font-weight: 600; }
.status-error { color: #f87171; font-weight: 600; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "general_history" not in st.session_state:
    st.session_state.general_history = []
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "upload_log" not in st.session_state:
    st.session_state.upload_log = []

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    chat_mode = st.radio(
        "Select mode",
        ["General Chat", "Document Q&A (RAG)"],
        help="General chat uses Groq directly. RAG mode answers from uploaded PDFs.",
    )
    is_rag_mode = chat_mode == "Document Q&A (RAG)"

    st.markdown("---")
    st.markdown("#### 🗄️ Vector Database")
    db_ok, db_message = check_db_connection()
    using_chroma = "Chroma" in db_message
    if db_ok and not using_chroma:
        st.markdown(f'<p class="status-ok">● {db_message}</p>', unsafe_allow_html=True)
    elif db_ok and using_chroma:
        st.markdown(f'<p class="status-ok">● {db_message}</p>', unsafe_allow_html=True)
        with st.expander("Enable pgVector (optional)"):
            st.markdown(
                "1. Install & start **Docker Desktop**\n"
                "2. Run: `docker compose up -d`\n"
                "3. Refresh this page — it will auto-switch to pgVector"
            )
    else:
        st.markdown(
            f'<p class="status-error">● Database offline</p><p style="color:#94a3b8;font-size:0.82rem;">{db_message}</p>',
            unsafe_allow_html=True,
        )

    if is_rag_mode:
        st.markdown("---")
        st.markdown("#### 📄 Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Ingest Documents", use_container_width=True, disabled=not uploaded_files):
            if not db_ok:
                st.error("Database is not connected. Start PostgreSQL first.")
            else:
                with st.spinner("Loading, chunking, and embedding documents..."):
                    for uploaded in uploaded_files:
                        try:
                            chunk_count, file_name = ingest_pdf_file(uploaded)
                            st.session_state.upload_log.append(
                                f"{file_name}: {chunk_count} chunks indexed"
                            )
                        except Exception as exc:
                            st.session_state.upload_log.append(
                                f"{uploaded.name}: failed ({exc})"
                            )
                st.success("Document ingestion completed.")

        if st.session_state.upload_log:
            st.markdown("**Indexed files**")
            for entry in st.session_state.upload_log[-5:]:
                st.caption(f"• {entry}")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        if is_rag_mode:
            st.session_state.rag_history = []
        else:
            st.session_state.general_history = []
        st.rerun()

mode_class = "mode-rag" if is_rag_mode else "mode-general"
mode_label = "RAG Document Q&A" if is_rag_mode else "General Agentic Chat"

st.markdown(
    f"""
    <div class="hero-card">
        <p class="hero-title">Agentic AI Chatbot</p>
        <p class="hero-subtitle">
            LangGraph orchestration • Groq LLM • LangChain RAG • pgVector retrieval
        </p>
        <span class="mode-pill {mode_class}">{mode_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

message_history = (
    st.session_state.rag_history if is_rag_mode else st.session_state.general_history
)
config = {
    "configurable": {
        "thread_id": "rag-thread-1" if is_rag_mode else "general-thread-1",
    }
}

for message in message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Retrieved Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <p><strong>Source {source['index']}</strong> • Page {source['page']}</p>
                            <p>{source['preview']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

chat_placeholder = (
    "Ask a question about your uploaded PDFs..."
    if is_rag_mode
    else "Ask anything..."
)
user_input = st.chat_input(chat_placeholder)

if user_input:
    message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_rag_mode and not db_ok:
            ai_message = (
                "RAG mode requires PostgreSQL with pgVector. "
                "Run `docker compose up -d` and upload PDFs from the sidebar."
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
                for chunk, _metadata in stream
                if getattr(chunk, "content", None)
            )

            sources = []
            if is_rag_mode:
                final_state = graph.get_state(config)
                retrieved = final_state.values.get("sources", [])
                sources = format_sources(retrieved)
                if sources:
                    with st.expander("📚 Retrieved Sources"):
                        for source in sources:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <p><strong>Source {source['index']}</strong> • Page {source['page']}</p>
                                    <p>{source['preview']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    message_history.append(
        {"role": "assistant", "content": ai_message, "sources": sources}
    )
