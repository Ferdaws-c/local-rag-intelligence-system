"""
sdk_utils.py — Foundry Local SDK Helpers
==========================================
Centralizes SDK initialization and model loading logic so that
ingest.py and app.py don't duplicate boilerplate code.

Usage:
    from sdk_utils import init_sdk, load_model
"""

from foundry_local_sdk import Configuration, FoundryLocalManager


def init_sdk(app_name: str) -> FoundryLocalManager:
    """
    Initializes the Foundry Local SDK singleton and registers GPU execution providers
    if available.

    Safe to call multiple times — if the SDK is already initialized
    (e.g., when Streamlit hot-reloads or switches models in the same
    process), the existing instance is returned without error.

    Parameters:
        app_name : Identifier registered with the Foundry Local service.

    Returns:
        The active FoundryLocalManager singleton instance.
    """
    try:
        config = Configuration(app_name=app_name)
        FoundryLocalManager.initialize(config)
    except Exception as exc:
        # Silently ignore "already initialized" — all other exceptions re-raised.
        if "already been initialized" not in str(exc):
            raise

    manager = FoundryLocalManager.instance

    # Register CUDA execution provider for GPU acceleration if available and not yet registered in this process
    try:
        eps = manager.discover_eps()
        cuda_ep = next((ep for ep in eps if ep.name == "CUDAExecutionProvider"), None)
        if cuda_ep and not cuda_ep.is_registered:
            print("[sdk_utils] CUDAExecutionProvider detected but not registered in this process.")
            print("[sdk_utils] Registering GPU acceleration (~6-7 min process launch cost)...")
            res = manager.download_and_register_eps(["CUDAExecutionProvider"])
            print(f"[sdk_utils] CUDA EP registration complete: success={res.success}, registered={res.registered_eps}")
    except Exception as exc:
        print(f"[sdk_utils] Warning: GPU acceleration registration check encountered an error: {exc}")

    return manager


def load_model(manager: FoundryLocalManager,
               model_name: str,
               model_type: str = "model",
               progress_callback=None):
    """
    Downloads (if not cached) and loads a Foundry Local model into memory.

    Parameters:
        manager           : Active FoundryLocalManager instance.
        model_name        : Catalog alias (e.g. 'qwen3-embedding-0.6b').
        model_type        : Human-readable label used in log output.
        progress_callback : Optional callable(float) receiving download
                            progress as a percentage (0.0–100.0).
                            Pass None to suppress all progress output.

    Returns:
        The loaded Foundry Local model object.
    """
    model = manager.catalog.get_model(model_name)
    
    # Explicitly check for and select the CUDA/GPU variant if available
    try:
        gpu_variant = next(
            (v for v in model.variants if v.info and v.info.runtime and (
                str(v.info.runtime.device_type).upper() in ("GPU", "DEVICETYPE.GPU") or
                getattr(v.info.runtime, "execution_provider", "") == "CUDAExecutionProvider"
            )),
            None
        )
        if gpu_variant:
            model.select_variant(gpu_variant)
            print(f"[sdk_utils] Explicitly selected GPU variant for '{model_name}': {gpu_variant.id}")
        else:
            print(f"[sdk_utils] Warning: No GPU variant found in catalog for '{model_name}'. Active default variant: {model.id}")
    except Exception as exc:
        print(f"[sdk_utils] Note on variant selection: {exc}")

    # Diagnostic log: report selected variant, device type, and execution provider
    try:
        variant = getattr(model, "_selected_variant", None)
        rt = variant.info.runtime if variant and hasattr(variant, "info") else None
        device = rt.device_type if rt else "UNKNOWN"
        ep_name = rt.execution_provider if rt else "UNKNOWN"
        print(f"[sdk_utils] Loading {model_type} '{model_name}' -> variant={model.id} (device={device}, ep={ep_name})")
    except Exception:
        print(f"[sdk_utils] Loading {model_type} '{model_name}' -> variant={model.id}")

    # Check if the model is already on disk
    was_cached = model.is_cached
    
    if not was_cached:
        cb = progress_callback if progress_callback is not None else lambda p: None
        model.download(cb)

    model.load()
    return model


# ------------------------------------------------------------------
# Background Memory Monitor & Comprehensive Offloading Engine
# ------------------------------------------------------------------
import ctypes
import gc
import os
import sys
import threading
import time


def get_process_memory_mb() -> float:
    """Returns the current process memory (RSS / Working Set) in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def unload_all_models() -> dict:
    """
    Executes an aggressive 5-stage memory offloading and cleanup sequence:
      1. Native SDK Model Unloading (Thorough 3-way sweep across ModelLoadManager, Catalog loaded models, and all Catalog models)
      2. Streamlit Cache & Session Eviction (st.cache_resource.clear() & st.cache_data.clear())
      3. Cyclic Garbage Collection (Triple-pass gc.collect())
      4. PyTorch / CUDA / ONNX Memory Flush (torch.cuda.empty_cache() & ipc_collect())
      5. OS Process Working Set Trimming (SetProcessWorkingSetSize & EmptyWorkingSet on Windows / malloc_trim on Linux)

    Returns dict with stats: {"freed_mb": float, "before_mb": float, "after_mb": float, "models_unloaded": int, "unloaded_ids": list[str]}
    """
    before_mb = get_process_memory_mb()
    unloaded_ids = set()

    # Stage 1: Native SDK Model Unloading (Exhaustive multi-pass sweep across loaded models, catalog variants, and aliases)
    if FoundryLocalManager.instance:
        manager = FoundryLocalManager.instance
        load_mgr = getattr(manager, "_model_load_manager", None)

        if load_mgr:
            # 1a. Direct ModelLoadManager list_loaded sweep
            try:
                loaded_ids = load_mgr.list_loaded()
                for model_id in loaded_ids:
                    try:
                        load_mgr.unload(model_id)
                        unloaded_ids.add(model_id)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[sdk_utils] Direct list_loaded sweep note: {exc}")

            # 1b. Catalog exhaustive variant ID & model ID sweep
            if hasattr(manager, "catalog"):
                try:
                    all_models = manager.catalog.list_models()
                    for m in all_models:
                        # Attempt to unload the model itself
                        for candidate_id in [getattr(m, 'id', None), getattr(m, 'alias', None)]:
                            if candidate_id:
                                try:
                                    load_mgr.unload(candidate_id)
                                    unloaded_ids.add(candidate_id)
                                except Exception:
                                    pass
                        # Attempt to unload all variants of the model
                        if hasattr(m, 'variants'):
                            for v in m.variants:
                                v_id = getattr(v, 'id', None)
                                if v_id:
                                    try:
                                        load_mgr.unload(v_id)
                                        unloaded_ids.add(v_id)
                                    except Exception:
                                        pass
                except Exception as exc:
                    print(f"[sdk_utils] Catalog exhaustive variant sweep note: {exc}")

            # 1c. Hardcoded fallback alias sweep for core models
            known_aliases = [
                "qwen3-embedding-0.6b", "phi-3.5-mini", "qwen2.5-0.5b", "phi-4-mini",
                "qwen3-embedding-0.6b-cuda-gpu:1", "Phi-3.5-mini-instruct-cuda-gpu:2",
                "qwen2.5-0.5b-instruct-cuda-gpu:4", "Phi-4-mini-instruct-cuda-gpu:5"
            ]
            for alias in known_aliases:
                try:
                    load_mgr.unload(alias)
                    unloaded_ids.add(alias)
                except Exception:
                    pass

    # Stage 2: Clear Streamlit Resource & Data Caches
    try:
        import streamlit as st
        st.cache_resource.clear()
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            if "model" in k.lower() or "client" in k.lower():
                if k not in ("selected_model", "chat_history_selectbox"):
                    del st.session_state[k]
    except Exception:
        pass

    # Stage 3: Python Cyclic Garbage Collection (Triple pass)
    for _ in range(3):
        gc.collect()

    # Stage 4: GPU / PyTorch CUDA Cache Flushing
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            print(f"[sdk_utils] PyTorch CUDA cleanup warning: {exc}")

    # Stage 5: OS Process Working Set Memory Trimming (Force physical RAM sweep across entire process tree)
    # NOTE: Calling EmptyWorkingSet on GetCurrentProcess() pseudo-handle (-1) fails with Win32 Error 6 (ERROR_INVALID_HANDLE).
    # We acquire real privileged process handles via OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION) across
    # the entire process hierarchy (current PID, parent PID, child worker PIDs) to sweep physical RAM down to ~1-2 MB.
    try:
        if sys.platform == "win32":
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_QUERY_INFORMATION = 0x0400
            target_pids = [os.getpid()]
            try:
                import psutil
                proc = psutil.Process(os.getpid())
                if proc.parent():
                    target_pids.append(proc.parent().pid)
                for child in proc.children(recursive=True):
                    target_pids.append(child.pid)
            except Exception:
                pass

            for pid in set(target_pids):
                try:
                    h_process = ctypes.windll.kernel32.OpenProcess(
                        PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid
                    )
                    if h_process:
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(
                            h_process, ctypes.c_size_t(-1), ctypes.c_size_t(-1)
                        )
                        ctypes.windll.psapi.EmptyWorkingSet(h_process)
                        ctypes.windll.kernel32.CloseHandle(h_process)
                except Exception:
                    pass
        elif sys.platform.startswith("linux"):
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass
    except Exception as exc:
        print(f"[sdk_utils] OS Working Set trim warning: {exc}")

    after_mb = get_process_memory_mb()
    freed_mb = max(0.0, before_mb - after_mb)

    print(f"[sdk_utils] Thorough Memory Offload Complete: Unloaded {len(unloaded_ids)} model(s). RAM: {before_mb:.1f} MB -> {after_mb:.1f} MB (Freed {freed_mb:.1f} MB)")

    return {
        "freed_mb": freed_mb,
        "before_mb": before_mb,
        "after_mb": after_mb,
        "models_unloaded": len(unloaded_ids),
        "unloaded_ids": list(unloaded_ids),
    }



class MemoryMonitor:
    last_query_time = time.time()
    timeout_seconds = 30  # Default to 30 seconds
    _thread = None
    _lock = threading.Lock()
    is_busy = False
    is_offloaded = True  # Starts offloaded until a user query loads models into memory

    @classmethod
    def start(cls):
        """Starts the background watchdog thread if it isn't running already."""
        with cls._lock:
            if cls._thread is None or not cls._thread.is_alive():
                cls._thread = threading.Thread(target=cls._watchdog, daemon=True)
                cls._thread.start()

    @classmethod
    def ping(cls):
        """Resets the idle timer and marks models as active (call whenever a query starts)."""
        cls.last_query_time = time.time()
        cls.is_offloaded = False

    @classmethod
    def set_busy(cls, busy: bool):
        """Sets the busy flag to prevent unloading while actively generating a response."""
        cls.is_busy = busy
        if not busy:
            # When finishing work, reset idle timer for a full timeout window
            cls.ping()

    @classmethod
    def unload_now(cls) -> dict:
        """Executes immediate synchronous multi-stage memory offload."""
        cls.is_offloaded = True
        return unload_all_models()

    @classmethod
    def force_unload_after(cls, delay_seconds: int):
        """Forcefully unloads all models after a specific delay, ignoring standard timeout locks."""
        def _unload():
            time.sleep(delay_seconds)
            cls.is_offloaded = True
            unload_all_models()
        threading.Thread(target=_unload, daemon=True).start()

    @classmethod
    def set_timeout(cls, seconds: int):
        """Updates the auto-free timeout setting."""
        cls.timeout_seconds = seconds

    @classmethod
    def _watchdog(cls):
        """Background loop that checks for idle expiration without infinite offload loop spam."""
        while True:
            time.sleep(5)
            # NOTE: Skip checking if the app is actively processing a prompt OR already offloaded
            # This prevents spammed 30-second offload loops when the system is already idle.
            if cls.is_busy or cls.is_offloaded:
                continue

            if cls.timeout_seconds > 0:
                elapsed = time.time() - cls.last_query_time
                if elapsed >= cls.timeout_seconds:
                    # Idle timeout reached: execute offload ONCE and mark as offloaded
                    cls.is_offloaded = True
                    unload_all_models()
                    cls.last_query_time = time.time()

