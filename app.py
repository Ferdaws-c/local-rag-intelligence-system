"""
app.py — SmartHome Hub Assistant (Streamlit Web Interface)
===========================================================
Launches the web UI for the local RAG assistant.

Run with:
    streamlit run app.py

Features:
  - Sidebar model selector (3 speed/quality options)
  - Real-time download progress popup on first model load
  - Live token-by-token streaming of model responses
  - Source document citations with similarity scores
  - Per-model chat history and retrieval log
  - Light / dark mode compatible (toggle via ⋮ → Settings)
"""

import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from rag_core import answer_query, init_models

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="SmartHome Hub Assistant",
    page_icon="🏠",
    layout="centered",
)

# Prevent transparency bleed-through when switching to light mode
st.markdown(
    """
    <style>
    .stApp { background-color: var(--background-color); }
    [data-testid="stSidebar"] { background-color: var(--secondary-background-color); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar — Model Selector & Knowledge Base Info
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    MODEL_OPTIONS = {
        "⚡ Fast — qwen2.5-0.5b (~1–2s)":          "qwen2.5-0.5b",
        "⚖️ Balanced — phi-3.5-mini (~5–10s)":     "phi-3.5-mini",
        "🎯 Best Quality — phi-4-mini (~8–15s)":    "phi-4-mini",
    }

    selected_label = st.selectbox(
        "Chat Model",
        options=list(MODEL_OPTIONS.keys()),
        index=1,
        help="Smaller = faster responses. Larger = higher quality answers.",
    )
    selected_model = MODEL_OPTIONS[selected_label]
    st.caption(f"Active: `{selected_model}`")
    st.divider()

    st.header("ℹ️ Knowledge Base")
    st.markdown("""
    **Documents loaded:**
    - FAQ
    - Setup Guide
    - Technical Specifications
    - Warranty & Returns
    - Troubleshooting Guide

    **Embedding model:** `qwen3-embedding-0.6b`
    """)

    st.divider()
    st.header("🔍 Retrieval Log")
    retrieval_log_placeholder = st.empty()


# ------------------------------------------------------------------
# Cached Model Loading with Real-Time Download Progress
# ------------------------------------------------------------------
@st.cache_resource
def load_models(chat_model_name: str) -> dict:
    """
    Loads both the embedding model and the selected chat model.

    Decorated with @st.cache_resource so each model is loaded only once
    per Streamlit session. Switching models loads the new model once,
    then serves it instantly on all subsequent queries.

    Shows a live download-progress popup only on the first load of each
    model. On cache hits the function body is skipped entirely.
    """
    with st.status("🔄 Loading AI Models...", expanded=True) as status:

        # Capture the Streamlit session context so the SDK's background
        # download thread can update the progress bars safely.
        ctx = get_script_run_ctx()

        # Embedding model progress bar
        st.markdown("**Embedding model** — `qwen3-embedding-0.6b`")
        embed_bar  = st.progress(0, text="Checking cache...")
        embed_done = st.empty()

        def embed_cb(pct: float):
            if get_script_run_ctx() is None:
                add_script_run_ctx(threading.current_thread(), ctx)
            p = min(int(pct), 100)
            embed_bar.progress(p, text=f"Downloading: {p}%")

        # Chat model progress bar
        st.markdown(f"**Chat model** — `{chat_model_name}`")
        chat_bar  = st.progress(0, text="Checking cache...")
        chat_done = st.empty()

        def chat_cb(pct: float):
            if get_script_run_ctx() is None:
                add_script_run_ctx(threading.current_thread(), ctx)
            p = min(int(pct), 100)
            chat_bar.progress(p, text=f"Downloading: {p}%")

        result = init_models(
            chat_model_name,
            embed_progress_cb=embed_cb,
            chat_progress_cb=chat_cb,
        )

        embed_done.success("Embedding model ready ✅")
        chat_done.success(f"`{chat_model_name}` ready ✅")
        status.update(label="✅ All models loaded!", state="complete", expanded=False)

    return result


# ------------------------------------------------------------------
# App Header
# ------------------------------------------------------------------
st.title("🏠 SmartHome Hub v2.0 Assistant")
st.caption(
    "Ask any question about your SmartHome Hub. "
    "All answers come from the local knowledge base — no internet required."
)
st.divider()

# ------------------------------------------------------------------
# Session State — keyed per model so switching models starts fresh
# ------------------------------------------------------------------
chat_key = f"messages_{selected_model}"
log_key  = f"log_{selected_model}"

if chat_key not in st.session_state:
    st.session_state[chat_key] = []
if log_key not in st.session_state:
    st.session_state[log_key] = []

# ------------------------------------------------------------------
# Display Existing Chat History
# ------------------------------------------------------------------
for msg in st.session_state[chat_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['filename']}** — `{src['score'] * 100:.1f}%` match"
                    )
                    st.caption(src["content"][:200] + "...")

# ------------------------------------------------------------------
# Chat Input & Response Generation
# ------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your SmartHome Hub..."):

    # Load (or retrieve cached) models
    try:
        models = load_models(selected_model)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state[chat_key].append({"role": "user", "content": prompt})

    # Stream the assistant's response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        def stream_update(text: str):
            """Updates the UI with accumulated text as tokens arrive."""
            answer_placeholder.markdown(text + "▌")

        with st.spinner("Searching knowledge base..."):
            result = answer_query(
                question=prompt,
                embedding_client=models["embedding_client"],
                chat_client=models["chat_client"],
                top_k=5,
                stream_callback=stream_update,
            )

        answer  = result["answer"]
        sources = result["sources"]

        # Replace the streaming placeholder with the final clean text
        answer_placeholder.markdown(answer)

        with st.expander("📄 Sources retrieved", expanded=True):
            for src in sources:
                st.markdown(
                    f"**{src['filename']}** — `{src['score'] * 100:.1f}%` match"
                )
                st.caption(src["content"][:200] + "...")

    # Persist to session state
    st.session_state[chat_key].append({
        "role": "assistant", "content": answer, "sources": sources,
    })
    st.session_state[log_key].insert(0, {"query": prompt, "sources": sources})

# ------------------------------------------------------------------
# Retrieval Log in Sidebar (last 3 queries)
# ------------------------------------------------------------------
with retrieval_log_placeholder.container():
    entries = st.session_state.get(log_key, [])
    if entries:
        for entry in entries[:3]:
            st.caption(f"Q: {entry['query'][:40]}...")
            for src in entry["sources"][:2]:
                st.caption(f"  → {src['filename']} ({src['score']:.3f})")
    else:
        st.caption("No queries yet.")
