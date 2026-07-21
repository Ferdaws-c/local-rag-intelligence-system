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
import shutil
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "source_documents"

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Sidebar background: exact colors per theme, covering all mobile/desktop wrappers
st.markdown("""
<style>
/* ── DARK MODE sidebar ── */
[data-theme="dark"] [data-testid="stSidebar"],
[data-theme="dark"] [data-testid="stSidebar"] > div,
[data-theme="dark"] [data-testid="stSidebar"] > div > div,
[data-theme="dark"] section[data-testid="stSidebar"],
[data-theme="dark"] section[data-testid="stSidebar"] > div,
[data-theme="dark"] [data-testid="stSidebarContent"],
[data-theme="dark"] [data-testid="stSidebarNav"],
[data-theme="dark"] div[class*="sidebar"],
[data-theme="dark"] div[class*="Sidebar"] {
    background-color: #0e1117 !important;
    background: #0e1117 !important;
}

/* ── LIGHT MODE sidebar ── */
[data-theme="light"] [data-testid="stSidebar"],
[data-theme="light"] [data-testid="stSidebar"] > div,
[data-theme="light"] [data-testid="stSidebar"] > div > div,
[data-theme="light"] section[data-testid="stSidebar"],
[data-theme="light"] section[data-testid="stSidebar"] > div,
[data-theme="light"] [data-testid="stSidebarContent"],
[data-theme="light"] [data-testid="stSidebarNav"],
[data-theme="light"] div[class*="sidebar"],
[data-theme="light"] div[class*="Sidebar"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
}

/* ── Force Sidebar Scrollability ── */
[data-testid="stSidebarContent"], 
[data-testid="stSidebarUserContent"] {
    overflow-y: auto !important;
    padding-bottom: 5rem !important; /* Extra padding so the bottom selectbox isn't cut off */
}
</style>
""", unsafe_allow_html=True)

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
from ingest import run_ingestion, SUPPORTED_EXTENSIONS, rename_document, clear_knowledge_base
from sdk_utils import MemoryMonitor

# Start the background idle timeout watcher
MemoryMonitor.start()

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
    const parentDoc = window.parent.document;
    parentDoc.addEventListener('keydown', function(e) {
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

# Additional UI polish
st.markdown(
    """
    <style>
    .stApp { background-color: var(--background-color); }
    /* Base sidebar button style */
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
    /* Document list overflow control */
    .doc-list-scroll {
        max-height: 220px;
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 2px;
        margin-top: 4px;
    }
    /* Naked icon buttons — applied via JS observer */
    .naked-icon button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 2px !important;
        min-height: unset !important;
        height: 1.4rem !important;
        width: auto !important;
        font-size: 0.85rem !important;
        color: inherit !important;
        line-height: 1 !important;
    }
    .naked-icon button:hover {
        background: transparent !important;
        opacity: 0.6 !important;
    }
    /* Doc name button — looks like plain text */
    .doc-name-btn button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 1px 0 !important;
        text-align: left !important;
        font-size: 0.82rem !important;
        color: inherit !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
        min-height: unset !important;
        height: auto !important;
        display: block !important;
    }
    .doc-name-btn button:hover {
        background: transparent !important;
        opacity: 0.75 !important;
    }
    /* Inline actions row */
    .doc-actions {
        padding: 2px 0 4px 18px;
        border-left: 1px solid rgba(128,128,128,0.25);
        margin: 0 0 4px 6px;
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
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = "idle"   # idle | running | done | error
if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = ""
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "doc_confirm_delete" not in st.session_state:
    st.session_state.doc_confirm_delete = None
if "clear_kb_confirm" not in st.session_state:
    st.session_state.clear_kb_confirm = False
if "reingest_all_confirm" not in st.session_state:
    st.session_state.reingest_all_confirm = False
if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None
if "pending_ingest" not in st.session_state:
    st.session_state.pending_ingest = None  # filename string waiting for user confirmation
if "pending_ingest_list" not in st.session_state:
    st.session_state.pending_ingest_list = []  # list of filenames queued for batch ingest
if "doc_expanded" not in st.session_state:
    st.session_state.doc_expanded = {}   # {filename: bool}
if "viewing_doc" not in st.session_state:
    st.session_state.viewing_doc = None  # filename string or None
if "kb_section_open" not in st.session_state:
    st.session_state.kb_section_open = True
if "ingest_progress" not in st.session_state:
    st.session_state.ingest_progress = 0.0
if "ingest_progress_text" not in st.session_state:
    st.session_state.ingest_progress_text = ""
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "processing_prompt" not in st.session_state:
    st.session_state.processing_prompt = None

# Define UI lock state: True if any document requires user confirmation
is_ui_locked = bool(st.session_state.get("pending_ingest")) or \
               bool(st.session_state.get("doc_confirm_delete")) or \
               bool(st.session_state.get("confirm_delete")) or \
               bool(st.session_state.get("clear_kb_confirm")) or \
               bool(st.session_state.get("reingest_all_confirm")) or \
               bool(st.session_state.get("is_generating"))

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
    if st.button("✏️  New Chat", use_container_width=True, type="primary", disabled=is_ui_locked):
        st.session_state.active_session_id = "temp_new_session"
        st.session_state.active_options_id = None
        st.session_state.chat_history_selectbox = "temp_new_session"
        st.rerun()

    st.divider()

    # ── Search chats ───────────────────────────────────────────────
    st.markdown("**💬 Chat History**")
    search_q = st.text_input(
        "Search Chats",
        key="chat_search_input",
        label_visibility="collapsed",
        placeholder="🔍 Search chats...",
        disabled=is_ui_locked,
    )

    all_sessions = get_all_sessions()

    # Filter sessions by search query if present
    if search_q.strip():
        all_sessions = [
            s for s in all_sessions if search_q.lower() in s["name"].lower()
        ]

    # Build options list of IDs
    options_list = []
    id_to_name = {}
    seen_names = {}

    if st.session_state.active_session_id == "temp_new_session":
        options_list.append("temp_new_session")
        id_to_name["temp_new_session"] = "New Chat"
        seen_names["New Chat"] = 0

    for s in all_sessions:
        options_list.append(s["id"])
        base_name = s["name"]
        if base_name in seen_names:
            seen_names[base_name] += 1
            # Streamlit bug: st.selectbox fails if format_func returns duplicate strings.
            # Fix: Inject invisible zero-width spaces (\u200b) to force string uniqueness.
            unique_name = base_name + ("\u200b" * seen_names[base_name])
        else:
            seen_names[base_name] = 0
            unique_name = base_name
            
        id_to_name[s["id"]] = unique_name

    if not options_list:
        st.caption("No chats found.")
    else:
        def on_chat_change():
            st.session_state.active_session_id = st.session_state.chat_history_selectbox
            st.session_state.active_options_id = None

        active_idx = 0
        if st.session_state.active_session_id in options_list:
            active_idx = options_list.index(st.session_state.active_session_id)

        selected_id = st.selectbox(
            "Select Chat History",
            options=options_list,
            index=active_idx,
            format_func=lambda x: id_to_name.get(x, "Unknown Chat"),
            label_visibility="collapsed",
            key="chat_history_selectbox",
            on_change=on_chat_change,
            disabled=is_ui_locked,
        )

        if selected_id != "temp_new_session":
            with st.expander("Manage Chat"):
                current_name = id_to_name.get(selected_id, "")
                new_name = st.text_input("Rename", value=current_name, key=f"ren_{selected_id}", label_visibility="collapsed", disabled=is_ui_locked)
                if st.button("Save Name", use_container_width=True, disabled=is_ui_locked):
                    if new_name.strip() and new_name.strip() != current_name:
                        rename_session(selected_id, new_name.strip())
                        st.rerun()
                
                st.divider()
                if "confirm_delete" not in st.session_state:
                    st.session_state.confirm_delete = None

                if st.session_state.confirm_delete == selected_id:
                    st.warning("Delete this chat?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes", use_container_width=True, type="primary"):
                            delete_session(selected_id)
                            st.session_state.active_session_id = "temp_new_session"
                            if "chat_history_selectbox" in st.session_state:
                                del st.session_state["chat_history_selectbox"]
                            st.session_state.confirm_delete = None
                            st.rerun()
                    with col_no:
                        if st.button("No", use_container_width=True):
                            st.session_state.confirm_delete = None
                            st.rerun()
                else:
                    if st.button("Delete Chat", type="primary", use_container_width=True, disabled=is_ui_locked):
                        st.session_state.confirm_delete = selected_id
                        st.rerun()

    st.divider()

    # ── Model Settings ─────────────────────────────────────────────
    st.markdown("**⚙️ Model Settings**")
    selected_label = st.selectbox(
        "Chat Model",
        options=list(MODEL_OPTIONS.keys()),
        index=1,
        help="Smaller = faster. Larger = better answers.",
        disabled=is_ui_locked,
    )
    selected_model = MODEL_OPTIONS[selected_label]
    st.session_state["selected_model"] = selected_model
    st.caption(f"Active: `{selected_model}`")

    # ── Auto-Free Memory Setting ──
    timeout_options = {
        "30 Seconds": 30,
        "2 Minutes": 120,
        "5 Minutes": 300,
        "30 Minutes": 1800,
        "Keep (Don't free)": 0
    }
    selected_timeout = st.selectbox(
        "🧹 Auto-Free Memory",
        options=list(timeout_options.keys()),
        index=0, # Default: 30 Seconds
        disabled=is_ui_locked,
        help="Unloads the AI from RAM if you haven't asked a question recently."
    )
    from sdk_utils import MemoryMonitor
    MemoryMonitor.set_timeout(timeout_options[selected_timeout])


    st.divider()

    # ── Knowledge Base Management ──────────────────────────────────
    # Header row: label | chevron toggle | upload
    kb_h1, kb_h2, kb_h3 = st.columns([5, 1, 1])
    with kb_h1:
        st.markdown("**ℹ️ Knowledge Base**")
    with kb_h2:
        kb_arrow = "▼" if st.session_state.kb_section_open else "▶"
        st.markdown('<div class="naked-icon">', unsafe_allow_html=True)
        if st.button(kb_arrow, key="kb_toggle_arrow", disabled=is_ui_locked):
            st.session_state.kb_section_open = not st.session_state.kb_section_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with kb_h3:
        st.markdown('<div class="naked-icon">', unsafe_allow_html=True)
        if st.button("📎", key="kb_upload_btn", help="Upload document", disabled=is_ui_locked):
            st.session_state.show_uploader = not st.session_state.show_uploader
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.kb_section_open:

        # ── Upload area ───────────────────────────────────────────
        pending_list = st.session_state.get("pending_ingest_list", [])

        if pending_list:
            # Show all queued files and ask for confirmation
            names = ", ".join(f"**{n}**" for n in pending_list)
            st.warning(f"Process {len(pending_list)} file(s): {names}?")
            ui_container = st.empty()
            yes_clicked = no_clicked = False
            with ui_container.container():
                c1, c2 = st.columns(2)
                with c1:
                    yes_clicked = st.button("✅ Yes", key="yes_ingest", use_container_width=True, type="primary")
                with c2:
                    no_clicked = st.button("❌ No", key="no_ingest", use_container_width=True)

            if yes_clicked:
                files_to_ingest = st.session_state.pending_ingest_list[:]
                st.session_state.pending_ingest_list = []
                st.session_state.pending_ingest      = None
                st.session_state.show_uploader       = False
                ui_container.empty()
                prog_ph = ui_container.empty()

                MemoryMonitor.set_busy(True)
                try:
                    models = load_models(st.session_state.get("selected_model", list(MODEL_OPTIONS.values())[1]))
                    total_files = len(files_to_ingest)
                    for idx, fname in enumerate(files_to_ingest):
                        def _progress_cb(pct: float, label: str, _idx=idx, _total=total_files):
                            overall = (_idx + pct) / _total
                            p = max(0.0, min(float(overall), 1.0))
                            prog_ph.progress(p, text=f"File {_idx+1}/{_total} — {int(pct*100)}% — {label}")
                        result = run_ingestion(models["embedding_client"], progress_callback=_progress_cb, target_file=fname)
                        if result["status"] != "ok":
                            prog_ph.error(f"❌ Error on {fname}: {result['message']}")
                            break
                    else:
                        prog_ph.success(f"✅ Done — {result['total']} total chunks in knowledge base")
                except Exception as e:
                    prog_ph.error(f"❌ Crash: {repr(e)}")
                finally:
                    MemoryMonitor.set_busy(False)
                    MemoryMonitor.force_unload_after(5)

                import time
                time.sleep(1.5)
                st.rerun()

            elif no_clicked:
                # Discard all saved files
                for fname in st.session_state.pending_ingest_list:
                    (DOCUMENTS_DIR / fname).unlink(missing_ok=True)
                st.session_state.pending_ingest_list = []
                st.session_state.pending_ingest      = None
                st.session_state.show_uploader       = False
                st.rerun()

        elif st.session_state.show_uploader:
            uploaded_files = st.file_uploader(
                "Add documents",
                type=["txt", "pdf", "docx"],
                label_visibility="collapsed",
                key="kb_uploader",
                disabled=is_ui_locked,
                accept_multiple_files=True,
            )
            if uploaded_files:
                saved = []
                for uf in uploaded_files:
                    dest = DOCUMENTS_DIR / uf.name
                    dest.write_bytes(uf.getvalue())
                    saved.append(uf.name)
                st.session_state.pending_ingest_list = saved
                st.session_state.pending_ingest      = saved[0] if len(saved) == 1 else None
                st.rerun()

        # ── Document list ─────────────────────────────────────────
        all_docs = sorted(
            f for f in DOCUMENTS_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        st.markdown('<div class="doc-list-scroll">', unsafe_allow_html=True)
        for doc in all_docs:
            max_len   = 20
            short_name = doc.name if len(doc.name) <= max_len else doc.name[:max_len - 2] + "…"
            is_expanded = st.session_state.doc_expanded.get(doc.name, False)
            is_viewing  = st.session_state.viewing_doc == doc.name

            # Row: doc name button | + toggle
            d_col, p_col = st.columns([5, 1])
            with d_col:
                # Clicking the name opens/closes the doc viewer in main area
                label = f"{'▶ ' if is_viewing else '📄 '}{short_name}"
                st.markdown('<div class="doc-name-btn">', unsafe_allow_html=True)
                if st.button(label, key=f"view_{doc.name}", use_container_width=True, help=doc.name, disabled=is_ui_locked):
                    st.session_state.viewing_doc = None if is_viewing else doc.name
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with p_col:
                # "+" icon toggles inline actions
                st.markdown('<div class="naked-icon">', unsafe_allow_html=True)
                plus_label = "−" if is_expanded else "+"
                if st.button(plus_label, key=f"plus_{doc.name}", disabled=is_ui_locked):
                    st.session_state.doc_expanded[doc.name] = not is_expanded
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            # Inline actions (shown when + is toggled on)
            if is_expanded:
                st.markdown('<div class="doc-actions">', unsafe_allow_html=True)

                # Rename
                new_doc_name = st.text_input(
                    "Rename",
                    value=doc.stem,
                    key=f"ren_doc_{doc.name}",
                    label_visibility="collapsed",
                    placeholder="New name…",
                    disabled=is_ui_locked,
                )
                if st.button("Save", key=f"save_doc_{doc.name}", use_container_width=True, disabled=is_ui_locked):
                    new_doc_name = new_doc_name.strip()
                    if new_doc_name and new_doc_name != doc.stem:
                        new_path = DOCUMENTS_DIR / (new_doc_name + doc.suffix)
                        doc.rename(new_path)
                        # If this doc was being viewed, update viewer reference
                        if st.session_state.viewing_doc == doc.name:
                            st.session_state.viewing_doc = new_path.name
                        st.session_state.doc_expanded.pop(doc.name, None)
                        # Instant DB rename (no background thread needed)
                        st.session_state.ingest_status         = "running"
                        st.session_state.ingest_message        = ""
                        st.session_state.ingest_progress       = 1.0
                        st.session_state.ingest_progress_text  = f"Renamed to {new_path.name}"

                        result = rename_document(doc.name, new_path.name)
                        st.session_state.ingest_status  = "done" if result["status"] == "ok" else "error"
                        st.session_state.ingest_message = (
                            f"✅ Rebuilt — {result['total']} chunks"
                            if result["status"] == "ok" else f"❌ {result['message']}"
                        )
                        st.rerun()

                # Delete
                if st.session_state.doc_confirm_delete == doc.name:
                    st.warning(f"Delete **{doc.name}**?")
                    ui_container = st.empty()
                    yes_clicked = False
                    no_clicked = False
                    
                    with ui_container.container():
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            yes_clicked = st.button("Yes", key=f"del_yes_{doc.name}", use_container_width=True, type="primary")
                        with dc2:
                            no_clicked = st.button("No", key=f"del_no_{doc.name}", use_container_width=True)

                    if yes_clicked:
                        if st.session_state.viewing_doc == doc.name:
                            st.session_state.viewing_doc = None
                        
                        filename_to_delete = doc.name
                        doc.unlink(missing_ok=True)
                        st.session_state.doc_confirm_delete = None
                        st.session_state.doc_expanded.pop(filename_to_delete, None)
                        
                        # Instantly clear buttons
                        ui_container.empty()
                        prog_ph = ui_container.empty()
                        
                        def _progress_cb(pct: float, label: str):
                            p = max(0.0, min(float(pct), 1.0))
                            prog_ph.progress(p, text=f"{int(p * 100)}% — {label}")

                        try:
                            models = load_models(st.session_state.get("selected_model", list(MODEL_OPTIONS.values())[1]))
                            result = run_ingestion(models["embedding_client"], progress_callback=_progress_cb, target_file=filename_to_delete, is_delete=True)
                            if result["status"] == "ok":
                                prog_ph.success(f"✅ Deleted — {result['total']} chunks remain")
                            else:
                                prog_ph.error(f"❌ {result['message']}")
                        except Exception as e:
                            prog_ph.error(f"❌ Crash: {repr(e)}")

                        import time
                        time.sleep(1.5)
                        st.rerun()
                        
                    elif no_clicked:
                        st.session_state.doc_confirm_delete = None
                        st.rerun()
                else:
                    if st.button("Delete", key=f"del_{doc.name}", use_container_width=True, type="primary", disabled=is_ui_locked):
                        st.session_state.doc_confirm_delete = doc.name
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)



    st.caption("Embedding: `qwen3-embedding-0.6b`")

    # ── Re-ingest All Documents ────────────────────────────────────
    st.divider()
    if st.session_state.reingest_all_confirm:
        st.info("🔄 **Re-ingest all documents?** This will rebuild the entire knowledge base.")
        rei_ui = st.empty()
        rei_yes = False
        rei_no  = False
        with rei_ui.container():
            ra, rb = st.columns(2)
            with ra:
                rei_yes = st.button("✅ Yes, Re-ingest", key="rei_yes", use_container_width=True, type="primary")
            with rb:
                rei_no = st.button("Cancel", key="rei_no", use_container_width=True)

        if rei_yes:
            st.session_state.reingest_all_confirm = False
            rei_ui.empty()
            prog = rei_ui.empty()

            def _rei_cb(pct: float, label: str):
                p = max(0.0, min(float(pct), 1.0))
                prog.progress(p, text=f"{int(p * 100)}% — {label}")

            try:
                models = load_models(st.session_state.get("selected_model", list(MODEL_OPTIONS.values())[1]))
                result = run_ingestion(models["embedding_client"], progress_callback=_rei_cb)
                if result["status"] == "ok":
                    prog.success(f"✅ Done — {result['total']} chunks in knowledge base")
                else:
                    prog.error(f"❌ {result['message']}")
            except Exception as e:
                prog.error(f"❌ Crash: {repr(e)}")

            import time
            time.sleep(1.5)
            st.rerun()

        elif rei_no:
            st.session_state.reingest_all_confirm = False
            st.rerun()
    else:
        # Guard: check if there are any documents to re-ingest
        has_docs = any(
            f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in DOCUMENTS_DIR.iterdir()
        ) if DOCUMENTS_DIR.exists() else False

        if st.button("🔄 Re-ingest All Documents", key="reingest_all_btn",
                     use_container_width=True, disabled=is_ui_locked):
            if not has_docs:
                st.toast("⚠️ No documents to ingest. Upload a file first.", icon="⚠️")
            else:
                st.session_state.reingest_all_confirm = True
                st.rerun()

    # ── Clear All Knowledge Base ─────────────────────────────────────
    st.divider()
    if st.session_state.clear_kb_confirm:
        st.warning("⚠️ **Delete the entire knowledge base?** This cannot be undone.")
        clr_ui = st.empty()
        clr_yes = False
        clr_no  = False
        with clr_ui.container():
            ca, cb = st.columns(2)
            with ca:
                clr_yes = st.button("🗑️ Yes, Clear All", key="clr_yes", use_container_width=True, type="primary")
            with cb:
                clr_no = st.button("Cancel", key="clr_no", use_container_width=True)

        if clr_yes:
            st.session_state.clear_kb_confirm = False
            clr_ui.empty()
            prog = clr_ui.empty()

            def _clr_cb(pct: float, label: str):
                p = max(0.0, min(float(pct), 1.0))
                prog.progress(p, text=f"{int(p * 100)}% — {label}")

            try:
                result = clear_knowledge_base(progress_callback=_clr_cb)
                if result["status"] == "ok":
                    prog.success(f"✅ Cleared — {result['deleted']} chunks permanently deleted")
                else:
                    prog.error(f"❌ {result['message']}")
            except Exception as e:
                prog.error(f"❌ Crash: {repr(e)}")
            finally:
                MemoryMonitor.force_unload_after(5)

            import time
            time.sleep(1.5)
            st.rerun()

        elif clr_no:
            st.session_state.clear_kb_confirm = False
            st.rerun()
    else:
        # Guard: check if the database has any chunks
        import sqlite3 as _sqlite3
        from rag_core import DB_FILE as _DB_FILE
        _db_has_data = False
        if _DB_FILE.exists():
            try:
                _conn = _sqlite3.connect(_DB_FILE)
                _cur  = _conn.cursor()
                _cur.execute("SELECT COUNT(*) FROM documents")
                _db_has_data = _cur.fetchone()[0] > 0
                _conn.close()
            except Exception:
                pass

        if st.button("🗑️ Clear Knowledge Base", key="clear_kb_btn",
                     use_container_width=True, disabled=is_ui_locked):
            if not _db_has_data:
                st.toast("ℹ️ The knowledge base is already empty.", icon="ℹ️")
            else:
                st.session_state.clear_kb_confirm = True
                st.rerun()

# ------------------------------------------------------------------
# Ensure an active session exists
# ------------------------------------------------------------------
if st.session_state.active_session_id is None or \
   (st.session_state.active_session_id != "temp_new_session" and not session_exists(st.session_state.active_session_id)):
    st.session_state.active_session_id = "temp_new_session"

active_session_id = st.session_state.active_session_id

# ------------------------------------------------------------------
# Main Area — Document Viewer (shown when a doc name is clicked)
# ------------------------------------------------------------------
if st.session_state.viewing_doc:
    vdoc_path = DOCUMENTS_DIR / st.session_state.viewing_doc
    if vdoc_path.exists():
        close_col, title_col = st.columns([1, 7])
        with close_col:
            if st.button("✕", key="close_viewer", help="Close viewer", disabled=is_ui_locked):
                st.session_state.viewing_doc = None
                st.rerun()
        with title_col:
            st.markdown(f"**📄 {st.session_state.viewing_doc}**")
        st.divider()

        suffix = vdoc_path.suffix.lower()
        if suffix == ".txt":
            content = vdoc_path.read_text(encoding="utf-8")
            st.text_area(
                "Document Content",
                value=content,
                height=400,
                label_visibility="collapsed",
                disabled=True,
            )
        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(vdoc_path))
                text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                st.text_area("PDF Content", value=text, height=400,
                             label_visibility="collapsed", disabled=True)
            except Exception as e:
                st.error(f"Could not read PDF: {e}")
        elif suffix == ".docx":
            try:
                import docx
                doc_obj = docx.Document(str(vdoc_path))
                text    = "\n\n".join(p.text for p in doc_obj.paragraphs if p.text)
                st.text_area("DOCX Content", value=text, height=400,
                             label_visibility="collapsed", disabled=True)
            except Exception as e:
                st.error(f"Could not read DOCX: {e}")
        st.divider()
    else:
        st.session_state.viewing_doc = None

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
for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "user":
            # Retry button — small, flat, below the message
            if st.button("↺ Retry", key=f"retry_{i}", help="Re-run this query", disabled=is_ui_locked):
                st.session_state.retry_prompt = msg["content"]
                st.rerun()
        elif msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['filename']}** — `{src['score'] * 100:.1f}%` match"
                    )
                    st.caption(src["content"][:200] + "...")

# ------------------------------------------------------------------
# Chat Input (handles both typed input and retry)
# ------------------------------------------------------------------
# Consume a pending retry, or wait for typed input
_typed = st.chat_input("Ask a question about your documents...", disabled=is_ui_locked)
new_prompt = st.session_state.retry_prompt or _typed

if new_prompt and not st.session_state.is_generating:
    st.session_state.processing_prompt = new_prompt
    if st.session_state.retry_prompt:
        st.session_state.retry_prompt = None
    st.session_state.is_generating = True
    st.rerun()

prompt = st.session_state.processing_prompt

if prompt:
    # Reset idle timer since the user is actively querying
    MemoryMonitor.ping()
    MemoryMonitor.set_busy(True)

    try:
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
                    top_k=8,
                    stream_callback=stream_update,
                    chat_history=messages,
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

        # Refresh sidebar session names and release lock
        st.session_state.processing_prompt = None
        st.session_state.is_generating = False
        st.rerun()

    finally:
        MemoryMonitor.set_busy(False)

