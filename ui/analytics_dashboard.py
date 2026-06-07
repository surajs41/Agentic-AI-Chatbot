import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from chat_storage import get_dashboard_analytics
from rag_backend import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def _pie(fig, **kwargs):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        **kwargs,
    )
    return fig


@st.fragment(run_every=5)
def render_analytics_dashboard(db_ok: bool, backend: str, model_name: str) -> None:
    data = get_dashboard_analytics()
    st.caption(f"Auto-refreshes every 5 seconds · Last update: {data['updated_at']}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("General questions", data["general_questions"])
    m2.metric("Document questions", data["doc_questions"])
    m3.metric("Est. tokens used", f"{data['total_tokens']:,}")
    m4.metric("Avg response", f"{data['avg_response_ms']:.0f} ms")
    m5.metric("Temperature", "0.7")
    m6.metric("Chunk storage", f"{data['total_chunk_chars']:,} chars")

    row1_a, row1_b = st.columns(2)
    with row1_a:
        st.markdown("#### Questions by section")
        q_df = pd.DataFrame(
            {
                "Section": ["General Chat", "Documents Q&A"],
                "Questions": [data["general_questions"], data["doc_questions"]],
            }
        )
        fig_q = _pie(
            px.pie(
                q_df,
                names="Section",
                values="Questions",
                color="Section",
                color_discrete_sequence=["#6366f1", "#22c55e"],
                hole=0.45,
            ),
            title="Questions Asked",
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with row1_b:
        st.markdown("#### Token usage by section")
        t_df = pd.DataFrame(
            {
                "Section": ["General Chat", "Documents Q&A"],
                "Tokens": [data["general_tokens"], data["doc_tokens"]],
            }
        )
        fig_t = _pie(
            px.pie(
                t_df,
                names="Section",
                values="Tokens",
                color="Section",
                color_discrete_sequence=["#f59e0b", "#3b82f6"],
                hole=0.45,
            ),
            title="Estimated Tokens",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    row2_a, row2_b = st.columns(2)
    with row2_a:
        st.markdown("#### Chunk size distribution")
        if data["chunk_sizes"]:
            cs_df = pd.DataFrame({"chars": data["chunk_sizes"]})
            fig_c = px.histogram(
                cs_df,
                x="chars",
                nbins=min(20, max(5, len(data["chunk_sizes"]) // 2)),
                color_discrete_sequence=["#8b5cf6"],
            )
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Characters per chunk",
                yaxis_title="Count",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("No chunk data yet. Index a document first.")

    with row2_b:
        st.markdown("#### Storage by document")
        if data["doc_storage"]:
            ds_df = pd.DataFrame(data["doc_storage"])
            fig_d = _pie(
                px.pie(
                    ds_df,
                    names="file_name",
                    values="chars",
                    hole=0.4,
                ),
                title="Chunk space (characters)",
            )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("No documents indexed yet.")

    st.markdown("#### Runtime configuration")
    cfg_df = pd.DataFrame(
        [
            {"Setting": "LLM Model", "Value": model_name},
            {"Setting": "Temperature", "Value": "0.7"},
            {"Setting": "Chunk size", "Value": str(CHUNK_SIZE)},
            {"Setting": "Chunk overlap", "Value": str(CHUNK_OVERLAP)},
            {"Setting": "Embedding model", "Value": EMBEDDING_MODEL},
            {"Setting": "Vector DB", "Value": backend.upper()},
            {"Setting": "Status", "Value": "Online" if db_ok else "Offline"},
            {"Setting": "Total chats", "Value": str(data["total_chats"])},
            {"Setting": "Total messages", "Value": str(data["total_messages"])},
            {"Setting": "Documents", "Value": str(data["total_documents"])},
            {"Setting": "Total chunks", "Value": str(data["total_chunks"])},
        ]
    )
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)

    if data["recent_activity"]:
        st.markdown("#### Recent activity")
        act_df = pd.DataFrame(data["recent_activity"])
        st.dataframe(act_df, use_container_width=True, hide_index=True)

    st.markdown("#### Response time trend")
    if data["response_times"]:
        rt_df = pd.DataFrame({"Message #": range(1, len(data["response_times"]) + 1), "ms": data["response_times"]})
        fig_rt = go.Figure()
        fig_rt.add_trace(
            go.Scatter(
                x=rt_df["Message #"],
                y=rt_df["ms"],
                mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=6),
            )
        )
        fig_rt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Recent assistant messages",
            yaxis_title="Response time (ms)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_rt, use_container_width=True)
