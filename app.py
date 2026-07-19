"""
app.py — Local RAG Intelligence System (Streamlit Web Interface)
=================================================================
Run with:
    streamlit run app.py

Features:
  - Persistent chat history with full CRUD (create, rename, delete sessions)
  - Sidebar model selector (3 speed/quality options)
  - Real-time model loading progress
  - Live token-by-token streaming of model responses
  - Source document citations with similarity scores
  - Light / dark mode compatible
"""

import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from chat_history import (
    add_message,
    create_session,
    delete_session,
    get_all_sessions,
    get_messages,
    init_history_db,
    rename_session,
    session_exists,
)
from rag_core import answer_query, init_models

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Local RAG Intelligence System",
    page_icon="🧠",
    layout="centered",
)

# Disable the annoying 'c' (clear cache) and 'r' (rerun) keyboard shortcuts
import streamlit.components.v1 as components
components.html(
    """
    <script>
    const parentWin = window.parent;
    parentWin.addEventListener('keydown', function(e) {
        if ((e.key.toLowerCase() === 'c' || e.key.toLowerCase() === 'r') && 
            e.target.nodeName !== 'INPUT' && 
            e.target.nodeName !== 'TEXTAREA' &&
            !e.target.isContentEditable) {
            e.stopImmediatePropagation();
            e.stopPropagation();
            e.preventDefault();
        }
    }, true);
    </script>
    """,
    height=0,
    width=0,
)

# Solid background colors to prevent transparency bleed in light mode
st.markdown(
    """
    <style>
    .stApp { background-color: var(--background-color); }
    [data-testid="stSidebar"] { background-color: var(--secondary-background-color); }
    /* Make sidebar session buttons look clean */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border: none;
        background: transparent;
        padding: 4px 8px;
        border-radius: 6px;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(128,128,128,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Database & Session Initialization
# ------------------------------------------------------------------
init_history_db()

# Initialize session state keys
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "active_options_id" not in st.session_state:
    st.session_state.active_options_id = None

# ------------------------------------------------------------------
# Cached Model Loading
# ------------------------------------------------------------------
MODEL_OPTIONS = {
    "⚡ Fast — qwen2.5-0.5b (~1–2s)":       "qwen2.5-0.5b",
    "⚖️ Balanced — phi-3.5-mini (~5–10s)":  "phi-3.5-mini",
    "🎯 Best Quality — phi-4-mini (~8–15s)": "phi-4-mini",
}


@st.cache_resource
def load_models(chat_model_name: str) -> dict:
    """
    Loads the embedding model and the selected chat model.
    Cached per model name — only runs once per model per process lifetime.
    """
    with st.status("🔄 Loading AI Models...", expanded=True) as status:
        ctx      = get_script_run_ctx()
        embed_ui = st.empty()
        chat_ui  = st.empty()

        def embed_cb(pct: float):
            if get_script_run_ctx() is None:
                add_script_run_ctx(threading.current_thread(), ctx)
            p = min(int(pct), 100)
            embed_ui.progress(p, text=f"Loading Embedding Model: {p}%")

        def chat_cb(pct: float):
            if get_script_run_ctx() is None:
                add_script_run_ctx(threading.current_thread(), ctx)
            p = min(int(pct), 100)
            chat_ui.progress(p, text=f"Loading Chat Model (`{chat_model_name}`): {p}%")

        result = init_models(
            chat_model_name,
            embed_progress_cb=embed_cb,
            chat_progress_cb=chat_cb,
        )
        status.update(label="✅ All models loaded and ready!", state="complete", expanded=False)

    return result


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:

    # ── New Chat Button ────────────────────────────────────────────
    if st.button("✏️  New Chat", use_container_width=True, type="primary"):
        # Get the currently selected model (read from session state if set)
        current_model = st.session_state.get("selected_model", "phi-3.5-mini")
        new_id = create_session(current_model)
        st.session_state.active_session_id = new_id
        st.session_state.active_options_id = None
        st.rerun()

    st.divider()

    # ── Chat History ───────────────────────────────────────────────
    st.markdown("**💬 Chat History**")

    all_sessions = get_all_sessions()

    if not all_sessions:
        st.caption("No chats yet. Start typing below!")
    else:
        for session in all_sessions:
            sid   = session["id"]
            sname = session["name"]
            is_active = (sid == st.session_state.active_session_id)
            is_options_open = (sid == st.session_state.active_options_id)

            col_name, col_arrow = st.columns([7, 1])

            with col_name:
                label = f"{'▶ ' if is_active else ''}{sname}"
                if st.button(
                    label,
                    key=f"load_{sid}",
                    use_container_width=True,
                    help=f"Model: {session['model']}",
                ):
                    st.session_state.active_session_id = sid
                    st.session_state.active_options_id = None
                    st.rerun()

            with col_arrow:
                arrow_label = "▲" if is_options_open else "▼"
                if st.button(arrow_label, key=f"arrow_{sid}", help="Toggle options"):
                    if is_options_open:
                        st.session_state.active_options_id = None
                    else:
                        st.session_state.active_options_id = sid
                    st.rerun()

            if is_options_open:
                new_name = st.text_input(
                    "Rename Chat",
                    value=sname,
                    key=f"rename_input_{sid}",
                    label_visibility="collapsed",
                )
                
                col_save, col_del, col_cancel = st.columns(3)
                with col_save:
                    if st.button("Save", key=f"save_{sid}", use_container_width=True):
                        if new_name.strip() and new_name.strip() != sname:
                            rename_session(sid, new_name.strip())
                        st.session_state.active_options_id = None
                        st.rerun()
                with col_del:
                    if st.button("Delete", key=f"del_{sid}", use_container_width=True, type="primary"):
                        delete_session(sid)
                        if st.session_state.active_session_id == sid:
                            st.session_state.active_session_id = None
                        st.session_state.active_options_id = None
                        st.rerun()
                with col_cancel:
                    if st.button("Close", key=f"close_{sid}", use_container_width=True):
                        st.session_state.active_options_id = None
                        st.rerun()
                st.divider()

    st.divider()

    # ── Model Settings ─────────────────────────────────────────────
    st.markdown("**⚙️ Model Settings**")
    selected_label = st.selectbox(
        "Chat Model",
        options=list(MODEL_OPTIONS.keys()),
        index=1,
        help="Smaller = faster. Larger = better answers.",
    )
    selected_model = MODEL_OPTIONS[selected_label]
    st.session_state["selected_model"] = selected_model
    st.caption(f"Active: `{selected_model}`")

    st.divider()

    # ── Knowledge Base Info ────────────────────────────────────────
    st.markdown("**ℹ️ Knowledge Base**")
    st.markdown("""
    - FAQ
    - Setup Guide
    - Technical Specifications
    - Warranty & Returns
    - Troubleshooting Guide

    Embedding: `qwen3-embedding-0.6b`
    """)


# ------------------------------------------------------------------
# Ensure an active session exists
# ------------------------------------------------------------------
if st.session_state.active_session_id is None or \
   not session_exists(st.session_state.active_session_id):
    # Auto-create a session on first visit
    new_id = create_session(selected_model)
    st.session_state.active_session_id = new_id

active_session_id = st.session_state.active_session_id

# ------------------------------------------------------------------
# Main Chat Area
# ------------------------------------------------------------------
st.title("🧠 Local RAG Intelligence System")
st.caption(
    "Query your documents using fully offline, privacy-preserving AI. "
    "All answers are grounded in the local knowledge base."
)
st.divider()

# Load messages for the active session from the database
messages = get_messages(active_session_id)

# Display all messages in the active session
for msg in messages:
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
# Chat Input
# ------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents..."):

    # Load (or retrieve cached) models
    try:
        models = load_models(selected_model)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # Persist and display user message
    add_message(active_session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
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

        answer_placeholder.markdown(answer)

        with st.expander("📄 Sources retrieved", expanded=True):
            for src in sources:
                st.markdown(
                    f"**{src['filename']}** — `{src['score'] * 100:.1f}%` match"
                )
                st.caption(src["content"][:200] + "...")

    # Persist assistant response to history DB
    add_message(active_session_id, "assistant", answer, sources=sources)

    # Refresh sidebar session names (first message auto-names the session)
    st.rerun()
