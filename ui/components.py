import os
from datetime import datetime, timezone

import streamlit as st

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def format_timestamp(value) -> str:
    if not value:
        return "just now"
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60} hr ago"


def render_message_header(role: str, created_at=None) -> None:
    label = "You" if role == "user" else "Assistant"
    icon = "👤" if role == "user" else "🤖"
    st.markdown(
        f'<div class="msg-meta">{icon} <strong>{label}</strong> · {format_timestamp(created_at)}</div>',
        unsafe_allow_html=True,
    )


def render_assistant_actions(message_id: int | None, content: str, feedback: str | None) -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.download_button(
            "📋 Copy",
            data=content,
            file_name="response.txt",
            mime="text/plain",
            key=f"copy_{message_id or hash(content)}",
            use_container_width=True,
        )
    with c2:
        if st.button("👍", key=f"up_{message_id}", use_container_width=True) and message_id:
            from chat_storage import set_message_feedback

            set_message_feedback(message_id, "up")
            st.rerun()
    with c3:
        if st.button("👎", key=f"down_{message_id}", use_container_width=True) and message_id:
            from chat_storage import set_message_feedback

            set_message_feedback(message_id, "down")
            st.rerun()


def render_model_info_panel(
    backend: str,
    collection: str,
    extra: dict | None = None,
) -> None:
    extra = extra or {}
    rows = [
        ("Model", MODEL_NAME),
        ("Provider", "Groq"),
        ("Temperature", "0.7"),
        ("Max Tokens", "1024"),
        ("Embedding", "BGE-small-en-v1.5"),
        ("Vector DB", backend),
        ("Collection", collection),
        ("Response Time", extra.get("Response Time", "—")),
        ("Tokens Used", extra.get("Tokens Used", "—")),
    ]
    html_rows = "".join(
        f'<div class="info-row"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in rows
    )
    st.markdown(
        f'<div class="info-panel"><h4>⚙ Model & Runtime</h4>{html_rows}</div>',
        unsafe_allow_html=True,
    )


def render_sources_panel(sources: list[dict]) -> None:
    if not sources:
        st.caption("Ask a question after uploading documents to see sources here.")
        return
    for source in sources:
        st.markdown(
            f"""
            <div class="source-card">
                <p>📄 <strong>{source.get('source', 'Document')}</strong></p>
                <p>Page {source.get('page', 'N/A')} · {source.get('confidence', 90)}% match</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_agent_steps(steps: list[dict]) -> None:
    for step in steps:
        css = "step-done" if step.get("done") else "step-pending"
        icon = "✅" if step.get("done") else "○"
        st.markdown(f'<p class="{css}">{icon} {step["label"]}</p>', unsafe_allow_html=True)


def render_landing_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <h1>💬 General AI Chat</h1>
            <p>Ask anything — powered by Groq LLM and LangGraph. No document upload required.</p>
            <p><strong>For PDF Q&A:</strong> go to <strong>📄 Documents</strong> in the sidebar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_doc_landing() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <h1>📄 Document Q&A</h1>
            <p>Upload a PDF → view chunk analytics → ask questions grounded in your file.</p>
            <p><strong>Stack:</strong> LangChain chunking · FastEmbed · pgVector · Groq</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chunk_analytics_box(chunks: list[dict], backend: str) -> None:
    if not chunks:
        return
    counts = [c.get("char_count", len(c.get("content", ""))) for c in chunks]
    pages = sorted({c.get("page") for c in chunks if c.get("page")})
    total_chars = sum(counts)
    avg_size = total_chars // len(counts)

    st.markdown("#### 📊 Chunk & Embedding Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chunks", len(chunks))
    c2.metric("Avg Chunk Size", f"{avg_size} chars")
    c3.metric("Total Characters", total_chars)
    c4.metric("Pages Indexed", len(pages))

    page_list = ", ".join(str(p) for p in pages) if pages else "N/A"
    st.caption(f"Pages covered: {page_list} · Stored in {backend.upper()}")


def render_action_upload_step(title: str, hint: str) -> None:
    st.markdown(
        f"""
        <div class="step-box">
            <h4>Step 1 — Upload document for: {title}</h4>
            <p>{hint}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
