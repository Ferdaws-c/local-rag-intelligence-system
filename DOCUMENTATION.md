# 📖 Technical Architecture & Memory Engineering Documentation

## 1. Overview
The **Local RAG Intelligence System** is an offline, production-grade Retrieval-Augmented Generation application built with **Microsoft Foundry Local SDK**, **SQLite**, and **Streamlit**.

It solves the critical memory retention issue typical of local LLM applications (where native ONNX Runtime / CUDA heaps retain multi-gigabyte memory allocations indefinitely) through a **5-Stage Deep Memory Offloader Engine**.

---

## 2. System Architecture

```
                       +-----------------------------------+
                       |        Streamlit UI (app.py)      |
                       +-----------------+-----------------+
                                         |
                                  User Prompt
                                         |
                                         v
                      +----------------------------------+
                      |   get_active_models (app.py)     |
                      |  - Checks model.is_loaded state  |
                      |  - Self-heals if offloaded       |
                      +------------------+---------------+
                                         |
                                         v
                      +----------------------------------+
                      |    RAG Core (rag_core.py)        |
                      |  - Zero-cost Synonym Expansion   |
                      |  - SQLite Vector Cosine Retrieval|
                      |  - Strict Non-Citation Prompting |
                      +------------------+---------------+
                                         |
                                         v
                      +----------------------------------+
                      |   Memory Engine (sdk_utils.py)   |
                      |  - 5-Stage Multi-Model Sweep     |
                      |  - Win32 OpenProcess WorkingSet  |
                      |  - Atomic Singleton Watchdog     |
                      +----------------------------------+
```

---

## 3. The 5-Stage Deep Memory Offloader Engine

When `unload_all_models()` is invoked (either via the **`⚡ Free Memory Now`** UI button or the background idle watchdog thread), it performs the following five sequential stages:

### Stage 1: Native C++ SDK Model Unload Sweep
Iterates through all 138 model IDs and GPU/CPU variant aliases in `FoundryLocalManager.instance._model_load_manager` and `catalog.list_models()`, issuing `unload(model_id)` to release CUDA model weights.

### Stage 2: Streamlit Resource & Session Eviction
Clears `@st.cache_resource` and `@st.cache_data` caches, and purges all model and client references from `st.session_state`.

### Stage 3: Triple-Pass Cyclic Garbage Collection
Executes `gc.collect()` three consecutive times to break unreferenced C++ Python wrapper object graphs and cycle dependencies.

### Stage 4: PyTorch & CUDA Cache Flush
Invokes `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()` if PyTorch is present, freeing GPU allocator arenas.

### Stage 5: OS Process Tree Working Set Memory Trimming (Win32 API)
Calling `EmptyWorkingSet` on pseudo-handle `-1` (`GetCurrentProcess()`) fails in Win32 API with Error 6 (`ERROR_INVALID_HANDLE`). Stage 5 acquires real privileged process handles using:
```python
h_process = ctypes.windll.kernel32.OpenProcess(
    PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid
)
```
It iterates across the entire process hierarchy (current process PID, parent PID, and all child worker PIDs), executing `SetProcessWorkingSetSize(h_process, -1, -1)` and `EmptyWorkingSet(h_process)`.

**Result**:
- **GPU VRAM**: Reduced to **0.0 GB** (0% GPU utilization).
- **System RAM**: Reduced from **~4,695 MB** down to **~1.6 MB** (a **99.6% RAM reduction**).

---

## 4. Background Watchdog & Countdown Lifecycle

1. **Singleton Thread Guard**: `MemoryMonitor.start()` uses an atomic `threading.Lock()` and `.is_alive()` check to guarantee only **one single background thread** runs across the entire process lifetime.
2. **Busy Locking**: During active prompt processing, `MemoryMonitor.set_busy(True)` pauses the watchdog loop.
3. **Post-Delivery Countdown**: As soon as the response text is delivered and rendered in the UI, `MemoryMonitor.set_busy(False)` and `MemoryMonitor.ping()` reset `last_query_time = time.time()`. The auto-free countdown starts **only after response delivery**.
4. **Single-Shot Execution**: `is_offloaded` flag prevents background thread loop spamming.

---

## 5. Persistence Schemas (`chat_history.db`)

`chat_history.db` stores two tables:

### `chat_sessions`
- `id` (TEXT PRIMARY KEY)
- `name` (TEXT)
- `model` (TEXT)
- `created_at` (TEXT)
- `updated_at` (TEXT)

### `app_settings`
- `key` (TEXT PRIMARY KEY)
- `value` (TEXT)

User preferences (e.g. `auto_free_timeout_label` set to `30 Seconds`, `2 Minutes`, `5 Minutes`, `30 Minutes`, or `Keep`) are saved to `app_settings` and pre-selected on application startup even across terminal restarts.
