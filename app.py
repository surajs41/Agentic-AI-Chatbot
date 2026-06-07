import io
import time

import streamlit as st
from langchain_core.messages import HumanMessage

from agentic_chatbot_backend import chatbot, format_sources, rag_chatbot
from chat_storage import (
    add_message,
    auto_title,
    create_session,
    delete_session,
    export_session_txt,
    get_document_chunks,
    get_messages,
    get_storage_label,
    init_chat_db,
    list_documents,
    list_sessions,
    register_document,
    save_document_chunks,
    update_session_title,
)
from rag_backend import (
    check_db_connection,
    get_active_backend,
    ingest_file,
)
from resume_service import analyze_jd_and_resume, build_resume_docx, extract_resume_text
from ui.analytics_dashboard import render_analytics_dashboard
from ui.components import (
    MODEL_NAME,
    render_assistant_actions,
    render_chunk_analytics_box,
    render_doc_landing,
    render_agent_steps,
    render_landing_hero,
    render_message_header,
    render_model_info_panel,
    render_sources_panel,
)
from ui.themes import THEMES, build_css

MODE_GENERAL = "general"
MODE_RAG = "rag"
COLLECTION = "General"

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_chat_db()

for key, val in {
    "theme": "dark",
    "active_general_session": None,
    "active_doc_session": None,
    "selected_doc": None,
    "agent_steps": [],
    "last_sources": [],
    "last_runtime": {},
    "doc_last_sources": [],
    "doc_last_runtime": {},
    "resume_file_name": None,
    "resume_file_bytes": None,
    "last_resume_docx": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.theme not in THEMES:
    st.session_state.theme = "dark"

st.markdown(build_css(st.session_state.theme), unsafe_allow_html=True)


def ensure_session(mode: str) -> str:
    key = "active_general_session" if mode == MODE_GENERAL else "active_doc_session"
    sessions = list_sessions(mode)
    active = st.session_state.get(key)
    if active and any(s["id"] == active for s in sessions):
        return active
    if sessions:
        st.session_state[key] = sessions[0]["id"]
        return sessions[0]["id"]
    sid = create_session(mode)
    st.session_state[key] = sid
    return sid


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def run_general_query(session_id: str, user_input: str) -> tuple[str, int, int]:
    config = {"configurable": {"thread_id": f"gen-{session_id}"}}
    start = time.time()
    stream = chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    )
    ai_message = st.write_stream(
        chunk.content for chunk, _ in stream if getattr(chunk, "content", None)
    )
    elapsed = int((time.time() - start) * 1000)
    tokens = estimate_tokens((ai_message or "") + user_input)
    st.session_state.last_runtime = {
        "Response Time": f"{elapsed / 1000:.1f} sec",
        "Tokens Used": str(tokens),
    }
    return ai_message or "", elapsed, tokens


def run_rag_query(session_id: str, user_input: str, backend: str) -> tuple[str, list, int, int]:
    config = {"configurable": {"thread_id": f"rag-{session_id}"}}
    start = time.time()

    st.session_state.agent_steps = [
        {"label": "Query received", "done": True},
        {"label": "Vector search (pgVector)", "done": False},
        {"label": "Context ranking", "done": False},
        {"label": "Groq LLM response", "done": False},
    ]

    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "collection": COLLECTION,
        "context": "",
        "sources": [],
    }

    stream = rag_chatbot.stream(inputs, config=config, stream_mode="messages")
    ai_message = st.write_stream(
        chunk.content for chunk, _ in stream if getattr(chunk, "content", None)
    )

    st.session_state.agent_steps[1]["done"] = True
    st.session_state.agent_steps[2]["done"] = True
    state = rag_chatbot.get_state(config)
    sources = format_sources(state.values.get("sources", []))
    st.session_state.agent_steps[3]["done"] = True

    elapsed = int((time.time() - start) * 1000)
    tokens = estimate_tokens((ai_message or "") + user_input)
    st.session_state.doc_last_sources = sources
    st.session_state.doc_last_runtime = {
        "Response Time": f"{elapsed / 1000:.1f} sec",
        "Tokens Used": str(tokens),
    }
    return ai_message or "", sources, elapsed, tokens


def render_sidebar_sessions(mode: str, session_key: str) -> None:
    label = "General chats" if mode == MODE_GENERAL else "Document Q&A chats"
    st.markdown(f"##### 💬 {label}")
    active = st.session_state.get(session_key)
    sessions = list_sessions(mode)
    if not sessions:
        st.caption("No chats yet.")
        return
    for session in sessions:
        sid = session["id"]
        title = (session["title"] or "New Chat")[:30]
        c1, c2 = st.columns([5, 1])
        with c1:
            kind = "primary" if sid == active else "secondary"
            if st.button(title, key=f"{mode}_{sid}", use_container_width=True, type=kind):
                st.session_state[session_key] = sid
                st.rerun()
        with c2:
            with st.popover("⋮", use_container_width=True):
                if st.button("Delete", key=f"del_{mode}_{sid}", use_container_width=True):
                    delete_session(sid)
                    if st.session_state.get(session_key) == sid:
                        st.session_state[session_key] = None
                    st.rerun()


def is_likely_jd(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) > 180:
        return True
    lower = cleaned.lower()
    keywords = (
        "job description",
        "requirements",
        "responsibilities",
        "qualifications",
        "we are looking",
        "about the role",
        "must have",
    )
    return any(k in lower for k in keywords)


def run_resume_analysis(jd_text: str, resume_bytes: bytes, resume_name: str) -> tuple[str, bytes, int]:
    start = time.time()
    resume_text = extract_resume_text(io.BytesIO(resume_bytes), resume_name)
    tailored = analyze_jd_and_resume(jd_text, resume_text)
    docx_bytes = build_resume_docx(tailored)
    elapsed = int((time.time() - start) * 1000)
    st.session_state.last_runtime = {
        "Response Time": f"{elapsed / 1000:.1f} sec",
        "Tokens Used": str(estimate_tokens(tailored + jd_text + resume_text)),
    }
    return tailored, docx_bytes, elapsed


def render_general_chat_page() -> None:
    session_id = ensure_session(MODE_GENERAL)
    messages = get_messages(session_id)

    chat_col, panel_col = st.columns([2.2, 1], gap="large")

    with chat_col:
        if not messages:
            render_landing_hero()

        for msg in messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                render_message_header(msg["role"], msg.get("created_at"))
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_assistant_actions(msg.get("id"), msg["content"], msg.get("feedback"))

        up_col, info_col = st.columns([1.2, 3])
        with up_col:
            resume_upload = st.file_uploader(
                "Resume",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
                key="chat_resume_upload",
            )
            if resume_upload:
                st.session_state.resume_file_name = resume_upload.name
                st.session_state.resume_file_bytes = resume_upload.getvalue()
        with info_col:
            if st.session_state.get("resume_file_name"):
                st.caption(f"Resume attached: **{st.session_state.resume_file_name}** — paste JD below and send")
            else:
                st.caption("Attach resume (PDF/DOCX) for JD alignment, or chat normally")

        user_input = st.chat_input("Ask anything — paste JD here for resume tailoring…")
        if user_input:
            if not messages:
                update_session_title(session_id, auto_title(user_input))
            add_message(session_id, "user", user_input)

            with st.chat_message("user", avatar="👤"):
                render_message_header("user")
                st.markdown(user_input)
            with st.chat_message("assistant", avatar="🤖"):
                render_message_header("assistant")
                has_resume = st.session_state.get("resume_file_bytes") and st.session_state.get("resume_file_name")
                if has_resume and is_likely_jd(user_input):
                    with st.spinner("Analyzing JD and tailoring resume…"):
                        ai_message, docx_bytes, elapsed = run_resume_analysis(
                            user_input,
                            st.session_state.resume_file_bytes,
                            st.session_state.resume_file_name,
                        )
                    st.session_state.last_resume_docx = docx_bytes
                    st.markdown(ai_message)
                    st.download_button(
                        "Download tailored resume (.docx)",
                        data=docx_bytes,
                        file_name="tailored_resume.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="dl_resume_new",
                    )
                elif has_resume:
                    ai_message = (
                        "Paste the full **Job Description** in the chat (requirements, responsibilities, "
                        "qualifications). I will generate a tailored one-page Word resume aligned to that JD."
                    )
                    elapsed = 0
                    st.markdown(ai_message)
                else:
                    ai_message, elapsed, tokens = run_general_query(session_id, user_input)
                add_message(session_id, "assistant", ai_message, response_time_ms=elapsed)
                render_assistant_actions(None, ai_message, None)
            st.rerun()

    with panel_col:
        render_model_info_panel(
            get_active_backend().upper(),
            "N/A — General Chat",
            st.session_state.last_runtime,
        )
        st.download_button(
            "📥 Export Chat",
            data=export_session_txt(session_id),
            file_name="chat.txt",
            use_container_width=True,
        )


def render_documents_page(db_ok: bool, backend: str) -> None:
    st.markdown("### 📄 Documents — Upload, Analytics & Q&A")
    st.caption("Upload a PDF here. View how it is chunked and embedded, then ask questions below.")

    up_col, info_col = st.columns([1, 1])
    with up_col:
        uploaded = st.file_uploader(
            "⬇️ Drop PDF · TXT · CSV · MD",
            type=["pdf", "txt", "csv", "md"],
            accept_multiple_files=False,
        )
        if st.button("Index Document", type="primary", disabled=not uploaded or not db_ok):
            with st.spinner("Chunking & embedding…"):
                count, pages, name, ext, records = ingest_file(uploaded, COLLECTION)
                register_document(name, COLLECTION, pages, count, ext)
                save_document_chunks(name, COLLECTION, records)
                st.session_state.selected_doc = name
            st.success(f"Indexed **{name}** — {count} chunks embedded.")
            st.rerun()

    docs = list_documents(COLLECTION)
    doc_names = [d["file_name"] for d in docs]

    if not doc_names:
        st.info("No documents yet. Upload a file above to get started.")
        return

    with info_col:
        selected = st.selectbox(
            "Select document",
            doc_names,
            index=doc_names.index(st.session_state.selected_doc)
            if st.session_state.selected_doc in doc_names
            else 0,
        )
        st.session_state.selected_doc = selected
        doc_meta = next(d for d in docs if d["file_name"] == selected)
        st.markdown(
            f"**{selected}**  \n"
            f"📃 {doc_meta.get('pages', 0)} pages · "
            f"🧩 {doc_meta.get('chunks', 0)} chunks · "
            f"✅ embedded in {backend}"
        )

    chunks = get_document_chunks(selected, COLLECTION)

    st.markdown("---")
    analytics_col, chunks_col = st.columns([1, 1])

    with analytics_col:
        if chunks:
            render_chunk_analytics_box(chunks, backend)
        else:
            st.warning("Re-index this file to view chunk analytics.")

    with chunks_col:
        st.markdown("#### 🧩 Chunk Breakdown")
        if chunks:
            for chunk in chunks:
                with st.expander(
                    f"Chunk {chunk['chunk_index']} · Page {chunk.get('page', '?')} · "
                    f"{chunk.get('char_count', 0)} chars"
                ):
                    st.text(chunk["content"])
        else:
            st.caption("No chunk data stored.")

    st.markdown("---")
    st.markdown("#### 💬 Ask questions about this document")

    session_id = ensure_session(MODE_RAG)
    doc_messages = get_messages(session_id)

    for msg in doc_messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            render_message_header(msg["role"], msg.get("created_at"))
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources used"):
                    for s in msg["sources"]:
                        st.caption(f"📄 {s.get('source')} · Page {s.get('page')}")

    doc_input = st.chat_input(f"Ask about **{selected}**…")
    if doc_input:
        if not doc_messages:
            update_session_title(session_id, auto_title(doc_input))
        add_message(session_id, "user", doc_input)

        with st.chat_message("user", avatar="👤"):
            st.markdown(doc_input)
        with st.chat_message("assistant", avatar="🤖"):
            if not db_ok:
                reply = "Vector store offline. Start Docker: `docker compose up -d`"
                sources, elapsed = [], 0
                st.markdown(reply)
            else:
                reply, sources, elapsed, _ = run_rag_query(session_id, doc_input, backend)
            add_message(session_id, "assistant", reply, sources, response_time_ms=elapsed)
        st.rerun()


def render_analytics_page(db_ok: bool, backend: str) -> None:
    st.markdown("### 📊 Analytics Dashboard")
    render_analytics_dashboard(db_ok, backend, MODEL_NAME)


def render_settings_page() -> None:
    st.markdown("### ⚙ Settings")
    theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        format_func=lambda k: THEMES[k]["label"],
        index=list(THEMES.keys()).index(st.session_state.theme),
    )
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()
    st.info(f"Model: **{MODEL_NAME}** · Provider: **Groq**")


# ── Navbar ──
n1, n2, n3 = st.columns([5, 2, 1])
with n1:
    st.markdown(
        '<div class="top-nav"><strong>🤖 Agentic AI Assistant</strong>'
        '<span> · Groq + LangGraph + pgVector</span></div>',
        unsafe_allow_html=True,
    )
with n2:
    t = st.selectbox(
        "theme",
        list(THEMES.keys()),
        format_func=lambda k: THEMES[k]["label"],
        index=list(THEMES.keys()).index(st.session_state.theme),
        label_visibility="collapsed",
    )
    if t != st.session_state.theme:
        st.session_state.theme = t
        st.rerun()
with n3:
    st.markdown("👤 Suraj")

db_ok, _ = check_db_connection()
backend = get_active_backend()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🤖 Agentic AI")
    st.caption("**Chats** = general AI  ·  **Documents** = PDF Q&A")

    page = st.radio(
        "Menu",
        ["💬 Chats", "📄 Documents", "📊 Analytics", "⚙ Settings"],
        label_visibility="collapsed",
    )

    if st.button("➕ New Chat", type="primary", use_container_width=True):
        mode = MODE_GENERAL if page.startswith("💬") else MODE_RAG
        key = "active_general_session" if mode == MODE_GENERAL else "active_doc_session"
        st.session_state[key] = create_session(mode)
        st.rerun()

    st.caption(f"{'🟢' if db_ok else '🔴'} {backend} · {get_storage_label()}")

    if page.startswith("💬"):
        render_sidebar_sessions(MODE_GENERAL, "active_general_session")
    elif page.startswith("📄"):
        render_sidebar_sessions(MODE_RAG, "active_doc_session")

# ── Route ──
if page.startswith("📄"):
    render_documents_page(db_ok, backend)
elif page.startswith("📊"):
    render_analytics_page(db_ok, backend)
elif page.startswith("⚙"):
    render_settings_page()
else:
    render_general_chat_page()
