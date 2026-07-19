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
if "active_session_id" not in st.session_state or st.session_state.active_session_id is None:
    st.session_state.active_session_id = "temp_new_session"
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
        st.session_state.active_session_id = "temp_new_session"
        st.session_state.active_options_id = None
        if "chat_history_selectbox" in st.session_state:
            del st.session_state["chat_history_selectbox"]
        st.rerun()

    st.divider()

    # ── Search chats ───────────────────────────────────────────────
    st.markdown("**💬 Chat History**")
    search_q = st.text_input(
        "Search Chats",
        key="chat_search_input",
        label_visibility="collapsed",
        placeholder="🔍 Search chats...",
    )

    all_sessions = get_all_sessions()

    # Filter sessions by search query if present
    if search_q.strip():
        all_sessions = [
            s for s in all_sessions if search_q.lower() in s["name"].lower()
        ]

    # Build selectbox options map
    session_options = {}
    if st.session_state.active_session_id == "temp_new_session":
        session_options["New Chat"] = "temp_new_session"
    
    for s in all_sessions:
        session_options[s["name"]] = s["id"]

    if not session_options:
        st.caption("No chats found.")
    else:
        # Determine the selectbox index based on active_session_id
        options_keys = list(session_options.keys())
        try:
            active_val = [k for k, v in session_options.items() if v == st.session_state.active_session_id][0]
            active_idx = options_keys.index(active_val)
        except (IndexError, ValueError):
            active_idx = 0

        selected_name = st.selectbox(
            "Select Chat History",
            options=options_keys,
            index=active_idx,
            label_visibility="collapsed",
            key="chat_history_selectbox",
        )

        selected_id = session_options[selected_name]

        # Handle session loading on selection change
        if selected_id != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_id
            st.session_state.active_options_id = None
            st.rerun()

        # Under the selectbox, provide Rename/Delete controls (except for temp new session)
        if selected_id != "temp_new_session":
            col_ren, col_del = st.columns(2)
            with col_ren:
                if st.button("✏️ Rename", use_container_width=True):
                    st.session_state.active_options_id = "rename"
                    st.rerun()
            with col_del:
                if st.button("🗑️ Delete", use_container_width=True):
                    st.session_state.active_options_id = "delete"
                    st.rerun()

            # Inline Rename Input
            if st.session_state.active_options_id == "rename":
                new_name = st.text_input(
                    "Enter new name:",
                    value=selected_name,
                    key=f"rename_inp_{selected_id}",
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save", key=f"save_btn_{selected_id}", use_container_width=True):
                        if new_name.strip() and new_name.strip() != selected_name:
                            rename_session(selected_id, new_name.strip())
                        st.session_state.active_options_id = None
                        if "chat_history_selectbox" in st.session_state:
                            del st.session_state["chat_history_selectbox"]
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_btn_{selected_id}", use_container_width=True):
                        st.session_state.active_options_id = None
                        st.rerun()

            # Inline Delete Confirmation
            elif st.session_state.active_options_id == "delete":
                st.warning("Delete this chat?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, Delete", key=f"del_confirm_{selected_id}", use_container_width=True, type="primary"):
                        delete_session(selected_id)
                        st.session_state.active_session_id = "temp_new_session"
                        st.session_state.active_options_id = None
                        if "chat_history_selectbox" in st.session_state:
                            del st.session_state["chat_history_selectbox"]
                        st.rerun()
                with col_no:
                    if st.button("No, Keep", key=f"del_cancel_{selected_id}", use_container_width=True):
                        st.session_state.active_options_id = None
                        st.rerun()

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
   (st.session_state.active_session_id != "temp_new_session" and not session_exists(st.session_state.active_session_id)):
    st.session_state.active_session_id = "temp_new_session"

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

# Load messages for the active session (bypass DB query for new unsaved sessions)
if active_session_id == "temp_new_session":
    messages = []
else:
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

    # Lazy-create the session in DB only when the first message is sent
    if active_session_id == "temp_new_session":
        active_session_id = create_session(selected_model)
        st.session_state.active_session_id = active_session_id

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
